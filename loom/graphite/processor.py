from heterograph.hgraph import HGraph
from heterograph.query.qgraph import QGraph
from .match_strategy import IsoMatchStrategy

class GraphProcessor:
    """Match AQL patterns and apply one in-place rewrite pass."""
    def __init__(self, *, match_strategy=None, snapshot=True):
        """Create a processor with an optional match strategy and snapshots."""
        self.match_strategy = (
            IsoMatchStrategy() if match_strategy is None else match_strategy
        )
        self.snapshot = snapshot
        if snapshot:
            from heterograph.webview import WebView
            self.webview = WebView()
        else:
            self.webview = None
        self._snapshot_count = 0

    # where(g, **match) -> bool
    # finalize(g, **match) -> None (in-place modification)
    def run(self, g:HGraph, *, 
            select:str, 
            where=None, 
            rewrite:str|None=None,
            finalize=None, **kwargs):

        snapshot_number = self._snapshot_count
        capture = self.snapshot and rewrite is not None
        if capture:
            self.webview.add_graph(g, title=f"original #{snapshot_number}")

        matches, pattern = self.match_strategy.find_matches(
            g=g, select=select, where=where, **kwargs)

        modified = False

        if rewrite and matches:
            modified = self._rewrite(
                g=g,
                rewrite=rewrite,
                finalize=finalize,
                matches=matches,
                find_pattern_graph=pattern
            )            

        if capture:
            self.webview.add_graph(g, title=f"rewrite #{snapshot_number}")
            self._snapshot_count += 1

        return {'matches': matches, 'pattern': pattern, 'modified': modified}

    def snapshot_view(self):
        """Display the captured snapshots in the processor's WebView."""
        if self.webview is None:
            raise RuntimeError("snapshot_view() requires GraphProcessor(snapshot=True).")
        return self.webview.run()


    def _rewrite(self, *, g, rewrite, finalize, matches, find_pattern_graph) -> bool:
        """Apply a parsed RHS to a disjoint set of LHS bindings.

        The rewrite is performed in stages: create RHS-only vertices, add RHS
        edges, redirect annotated boundary edges, run ``finalize``, remove
        obsolete preserved-preserved edges, and finally delete LHS-only
        vertices. Keeping cleanup last lets ``finalize`` inspect the complete
        replacement while protecting the cleanup targets from accidental
        removal.
        """


        # ---- Overlap detection (mandatory for rewrite) ----
        used = set()
        for m in matches:
            vs = set(m.values())
            overlap = used & vs
            if overlap:
                raise RuntimeError(
                    f"Overlapping matches detected on vertices {overlap}. "
                    "Rewrites require disjoint matches."
                )
            used |= vs


        modified = False
        
        # a{rewire:c}
        # a :=> {in: c, out:c}
        rewire_dict = { } 
        def node_pattern_annotations(qgraph, vx, rewire=None, rewire_in=None, rewire_out=None):
            if all(r is None for r in [rewire, rewire_in, rewire_out]):
                return
            target = qgraph.pmap[vx]['id']
            if target in rewire_dict:
                raise RuntimeError(
                    f"Multiple rewire annotations on the same pattern node [{target}] are not allowed."
                )
            if rewire is not None:
                rewire_dict[target] = { 'in': rewire, 'out': rewire }
            if rewire_in is not None:
                rewire_dict[target] = { 'in': rewire_in, 'out': None }
            if rewire_out is not None:
                rewire_dict[target] = { 'in': None, 'out': rewire_out }
        

        rewrite_pattern_graph = QGraph(rewrite, 
                                       vx_args=node_pattern_annotations) 
          
        # --- Precompute pattern-level sets
        L = set(find_pattern_graph.pmap['ids'].keys())
        R = set(rewrite_pattern_graph.pmap['ids'].keys())

        preserved_ids = L & R
        deleted_ids   = L - R
        created_ids   = R - L

        LE = set([(find_pattern_graph.pmap[s]['id'],
                   find_pattern_graph.pmap[t]['id']) for (s, t) in find_pattern_graph.edges])
        
        RE = set([(rewrite_pattern_graph.pmap[s]['id'],
                   rewrite_pattern_graph.pmap[t]['id']) for (s, t) in rewrite_pattern_graph.edges])    

        add_internal_edges    = RE - LE      
        remove_internal_edges = LE - RE  

        if 0:
            print(f"preserved_ids: {preserved_ids}")
            print(f"deleted_ids: {deleted_ids}")
            print(f"created_ids: {created_ids}")
            print(f"LE: {LE}")
            print(f"RE: {RE}")
            print(f"add_internal_edges: {add_internal_edges}")
            print(f"remove_internal_edges: {remove_internal_edges}")

        
        # --- Validate rewire annotations ---
        # The annotated node is the receiver and must exist in the RHS.
        # The referenced node is the deleted source and must be LHS-only.
        for receiver_id, info in rewire_dict.items():
            if receiver_id not in R:
                raise RuntimeError(
                    f"rewire receiver [{receiver_id}] must exist in the RHS."
                )

            for direction in ("in", "out"):
                source_id = info[direction]
                if source_id is not None and source_id not in deleted_ids:
                    raise RuntimeError(
                        f"rewire source [{source_id}] must be deleted by the rewrite."
                    )        

        # --- Apply rewrites for this single pass
        for m in matches:
            # step 1: Detect invalidated matches (some vx may have been deleted by earlier rewrites)
            # Ensure all LHS ids in m still exist in graph

            if not g.check_vx(list(m.values()), verify=False):
                vx_removed = { vx for vx in m.values() if not g.check_vx(vx, verify=False) }
                
                raise RuntimeError("[x] Match contains vertices that no longer exist in the graph. "
                                   "This indicates integrity issues with the rewrite rules. "
                                   f"Match: {m} / Removed vertices: {vx_removed}")                   

            # step 2: Build initial bindings 
            bind = dict(m)

            # step 3: Add new vertices for RHS-only ids
            for rid in created_ids:
                new_vx = g.add_vx(1)   # returns single vx
                bind[rid] = new_vx
                modified = True   
        
            # add reverse_bind
            matched_vs  = set(bind.values())  # host vertices in match
            # Build reverse lookup: host_vx -> pattern_id
            reverse_bind = {vx: pid for pid, vx in bind.items()}            

            if 0:
                print(f"bind: {bind}")
                print(f"reverse_bind: {reverse_bind}")

            # step 4: adding RHS-only edges: internal edge additions 
            for (sid, tid) in add_internal_edges:
                s_vx = bind[sid]
                t_vx = bind[tid]
                ret = g.add_edge(s_vx, t_vx)                     
                if ret:
                    modified = True   

            # For each deleted id X that is inherited by RHS id Y
            # {'matmul': {'in': 'd', 'out': None}, 'act': {'in': None, 'out': 'd'}}
            for y_id, info in rewire_dict.items():      
                y_vx = bind[y_id]                                
                x_id_in  = info["in"]
                x_id_out = info["out"] 
                                                         

                # ---- Conflict Detection ----
                # Rewiring only redirects edges crossing the match boundary.
                # If the other endpoint is inside the match and is preserved,
                # redirecting it would implicitly create an internal RHS edge.
                # That edge must be written explicitly in RE; otherwise the
                # rewrite would silently change the internal graph structure.
                # E.g., the follwowing is invalid because the internal edge (a, c) is not in the RHS:
                # select="a => b",
                # rewrite="a; c {rewire_in:b}",

                # Incoming edge: preserved source -> deleted source
                # becomes preserved source -> receiver.
                if info["in"] is not None:
                    source_id = info["in"]
                    source_vx = bind[source_id]
                    for src_vx in g.in_vx(source_vx):
                        if src_vx in matched_vs:
                            src_id = reverse_bind[src_vx]
                            if src_id in preserved_ids and src_id != y_id: # we allow self-edges
                                if (src_id, y_id) not in RE:
                                    raise RuntimeError(
                                        f"rewire_in conflict: deleting '{source_id}' and rewiring "
                                        f"would create internal edge {(src_id, y_id)}, "
                                        f"but RHS does not contain this edge."
                                    )

                # Outgoing edge: deleted source -> preserved destination
                # becomes receiver -> preserved destination.
                if info["out"] is not None:
                    source_id = info["out"]
                    source_vx = bind[source_id]
                    for dst_vx in g.out_vx(source_vx):
                        if dst_vx in matched_vs:
                            dst_id = reverse_bind[dst_vx]
                            if dst_id in preserved_ids and dst_id != y_id: # we allow self-edges
                                if (y_id, dst_id) not in RE:
                                    raise RuntimeError(
                                        f"rewire_out conflict: deleting '{source_id}' and rewiring "
                                        f"would create internal edge {(y_id, dst_id)}, "
                                        f"but RHS does not contain this edge."
                                    )                

                # ---- MUTATION PASS ----
                # Conflict detection has passed. Redirect only edges that
                # cross the match boundary. The old edges remain until the
                # deleted source vertices are removed below.

                # Incoming rewiring:
                #   external_source -> deleted_source
                # becomes:
                #   external_source -> receiver
                if x_id_in is not None:
                    x_vx = bind[x_id_in]
                    for src_vx in g.in_vx(x_vx):
                        # Internal edges are controlled explicitly by the RHS;
                        # they must not be redirected in this pass.
                        if src_vx not in matched_vs:
                            ret = g.add_edge(src_vx, y_vx)
                            if ret:
                                # Preserve edge metadata such as ports,
                                # channels, and indices.
                                g.pmap[ret[0]] = g.pmap[(src_vx, x_vx)].copy()  # copy properties from source vertex to new edge

                # Outgoing rewiring:
                #   deleted_source -> external_destination
                # becomes:
                #   receiver -> external_destination
                if x_id_out is not None:
                    x_vx = bind[x_id_out]
                    for dst_vx in g.out_vx(x_vx):
                        # As above, leave internal edges to the explicit RHS
                        # edge additions and removals.
                        if dst_vx not in matched_vs:
                            ret = g.add_edge(y_vx, dst_vx)
                            if ret:                           
                                # Copy metadata before the old source vertex
                                # is removed together with its original edge.
                                g.pmap[ret[0]] = g.pmap[(x_vx, dst_vx)].copy()


            if finalize:
                deleted_vxs = {bind[did] for did in deleted_ids}
                planned_removed_edges = {
                    edge for edge in g.edges
                    if edge[0] in deleted_vxs or edge[1] in deleted_vxs
                }
                planned_removed_edges.update(
                    (bind[sid], bind[tid])
                    for sid, tid in remove_internal_edges
                    if sid in preserved_ids and tid in preserved_ids
                )

                finalize_result = finalize(g, **bind)

                # finalize may perform other graph updates, but it must not
                # preempt this rewrite's cleanup. Deleted vertices must still
                # exist so the rewrite can remove them, and LHS-only edges
                # between preserved vertices must still exist so they can be
                # removed below.
                missing_deleted = [
                    did for did in deleted_ids
                    if not g.check_vx(bind[did], verify=False)
                ]
                if missing_deleted:
                    raise RuntimeError(
                        "finalize removed vertices that the rewrite must remove: "
                        f"{missing_deleted}."
                    )

                missing_edges = [
                    edge for edge in planned_removed_edges
                    if not g.check_edge(edge)
                ]
                if missing_edges:
                    raise RuntimeError(
                        "finalize removed edges that the rewrite must remove: "
                        f"{missing_edges}."
                    )

                if type(finalize_result) is not bool:
                    raise RuntimeError(
                        "finalize callback must return a bool, "
                        f"got {type(finalize_result).__name__}."
                    )
                if finalize_result:
                    modified = True

            # -- remove internal edges (LHS edges not in RHS)
            # -- also, we want to make sure that both source and target
            # -- nodes are preserved
            for (sid, tid) in remove_internal_edges:
                if sid in preserved_ids and tid in preserved_ids:
                    s_vx = bind[sid]
                    t_vx = bind[tid]
                    if g.check_edge((s_vx, t_vx)):
                        g.rm_edge((s_vx, t_vx))
                        modified = True                           

            # -- remove vertices marked to be removed
            for did in deleted_ids:
                vx = bind[did]
                if g.check_vx(vx, verify=False):
                    g.rm_vx(vx)
                    modified = True

        return modified

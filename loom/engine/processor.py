from heterograph.hgraph import HGraph
from heterograph.query.qgraph import QGraph
from .match_policy import IsoMatchPolicy

class Processor:
    def __init__(self, *, match_policy=IsoMatchPolicy(), snapshot=True):
        self.match_policy = match_policy
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

        matches, pattern = self.match_policy.find_matches(
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
            raise RuntimeError("snapshot_view() requires Processor(snapshot=True).")
        return self.webview.run()


    def _rewrite(self, *, g, rewrite, finalize, matches, find_pattern_graph) -> bool:


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

        
        # validate rewire constraints
        # ensure that the rewire sources are all in the deleted set
        # and that their targets are preserved nodes
        # for x_id, info in rewire_dict.items():
        #     if x_id not in deleted_ids:
        #         raise RuntimeError(
        #             "rewire id [%s] must be a removed node [%s]" % (x_id, deleted_ids)
        #         )
        #     if info["target"] not in created_ids:
        #         raise RuntimeError(
        #             "rewire target [%s] must be a created node [%s]" % (info["target"], created_ids)
        #         )

        # --- Apply rewrites for this single pass
        for m in matches:
            # Skip invalidated matches (some vx may have been deleted by earlier rewrites)
            # Ensure all LHS ids in m still exist in graph
            if not g.check_vx(list(m.values()), verify=False):
                vx_removed = { vx for vx in m.values() if not g.check_vx(vx, verify=False) }
                
                raise RuntimeError("[x] Match contains vertices that no longer exist in the graph. "
                                   "This indicates integrity issues with the rewrite rules. "
                                   f"Match: {m} / Removed vertices: {vx_removed}")                   

            # --- Build full bindings dict
            # Start with LHS bindings
            bind = dict(m)

            # Create new vertices for created RHS IDs
            for rid in created_ids:
                new_vx = g.add_vx(1)   # returns single vx
                bind[rid] = new_vx
                modified = True   
        
            matched_vs  = set(bind.values())  # host vertices in match
            # Build reverse lookup: host_vx -> pattern_id
            reverse_bind = {vx: pid for pid, vx in bind.items()}            

            # --- Internal edge additions (RHS edges not in LHS)
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

                # # Check incoming edges of x
                # if rewire_in:
                #     for src_vx in g.in_vx(x_vx):
                #         if src_vx in matched_vs:
                #             src_id = reverse_bind[src_vx]
                #             implied_edge = (src_id, y_id)
                #             if src_id != y_id: # we allow self-edges
                #                 if implied_edge not in RE:
                #                     raise RuntimeError(
                #                         f"rewire_in conflict: deleting '{x_id}' and rewiring "
                #                         f"would create internal edge {implied_edge}, "
                #                         f"but RHS does not contain this edge."
                #                     )

                # # Check outgoing edges of x
                # if rewire_out:
                #     for dst_vx in g.out_vx(x_vx):
                #         if dst_vx in matched_vs:
                #             dst_id = reverse_bind[dst_vx]
                #             implied_edge = (y_id, dst_id)
                #             if y_id != dst_id:
                #                 if implied_edge not in RE:
                #                     raise RuntimeError(
                #                         f"rewire_out conflict: deleting '{x_id}' and rewiring "
                #                         f"would create internal edge {implied_edge}, "
                #                         f"but RHS does not contain this edge."
                #                     )                

                # ---- MUTATION PASS ----
                if x_id_in:
                    x_vx = bind[x_id_in]
                    for src_vx in g.in_vx(x_vx):
                        # we only add the edge if src_vx is outside the match boundary
                        if src_vx not in matched_vs:
                            ret = g.add_edge(src_vx, y_vx)
                            if ret:
                                g.pmap[ret[0]] = g.pmap[(src_vx, x_vx)].copy()  # copy properties from source vertex to new edge
                
                if x_id_out:
                    x_vx = bind[x_id_out]
                    for dst_vx in g.out_vx(x_vx):
                        # we only add the edge if dst_vx is outside the match boundary
                        if dst_vx not in matched_vs:
                            ret = g.add_edge(y_vx, dst_vx)
                            if ret:                           
                                g.pmap[ret[0]] = g.pmap[(x_vx, dst_vx)].copy()


            if finalize:                
                if finalize(g, **bind) is True:
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

            # -- remove deleted vertices 
            for did in deleted_ids:
                vx = bind[did]
                if g.check_vx(vx, verify=False):
                    g.rm_vx(vx)
                    modified = True

         

        return modified
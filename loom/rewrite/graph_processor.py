from ast import pattern
import inspect
from heterograph.query.qgraph import QGraph
from heterograph.query.processor_dfs import QueryProcessorDFS
from heterograph.hgraph import HGraph

from graph_tool.topology import subgraph_isomorphism

class MatchPolicy:
    @staticmethod
    def vx_annotation(qgraph, vx, **kwargs):
        pass

    @staticmethod
    def eg_annotation(qgraph, eg,  **kwargs):
        pass

    @staticmethod
    def vx_check(g, qgraph, vx, qvx):
        return True

    @staticmethod
    def eg_check(g, qgraph, eg, qeg):
        return True
    
    @staticmethod
    def check(g, qgraph, match):
        return True

class GraphProcessor:
    match_algorithms = {
        # 'dfs': '_find_dfs_matches',
        'iso': '_find_iso_matches'
    }

    def __init__(self, *, 
                 mode='iso', 
                 deduplicate=False, 
                 match_policy=None):
        if mode not in self.match_algorithms:
            raise ValueError(f"Unknown mode: {mode}")
        
        self.mode = mode        
        self.deduplicate = deduplicate
        self.match_policy = match_policy or MatchPolicy

    def run(self, g:HGraph, *, find:str, where=None, rewrite:str|None=None, post=None, max_n=0):
        # where(g, **match) -> bool
        # post(g, **match) -> None (in-place modification)

        search_alg = getattr(self,  self.match_algorithms[self.mode])

        matches, pattern = search_alg(g=g, find=find, where=where, max_n=max_n)

        if self.deduplicate and len(matches) > 1:
            matches = self._deduplicate_matches(matches)

   
        modified = False


        if rewrite and matches:
            modified = self._rewrite(
                g=g,
                rewrite=rewrite,
                post=post,
                matches=matches,
                find_pattern_graph=pattern
            )            

        return {'matches': matches, 'pattern': pattern, 'modified': modified}


    def _deduplicate_matches(self, matches):
        # depuplicate matches if requested
        # e.g. match {x:1, y:2} and {x:1, y:2} are duplicates
            
        unique = {}
        for m in matches:
            key = frozenset(m.values())
            if key not in unique:
                unique[key] = m
        matches = list(unique.values())   
        return matches     
    
    def _find_dfs_matches(self, *, g:HGraph, find, where):

        def node_pattern_annotations(qgraph:QGraph, vx, type=None, **kwargs):
            qgraph.pmap[vx]['type'] = type

        def edge_pattern_annotations(qgraph:QGraph, eg, dist=1, **kwargs):
            qgraph.pmap[eg]['xdist'] = dist        


        find_pattern_graph = QGraph(find, 
                                    vx_args=self.match_node_annotations, 
                                    eg_args=self.match_edge_annotations)

        # ---- Default where
        if where is None:
            match_filter  = lambda g, qg, match: True
        else:
            params = inspect.signature(where).parameters
            params_in_signature = set(params.keys()) - {"g"}        
            match_filter = lambda g, qg, match: where(g, **{k: v for k, v in match.items() if k in params_in_signature})            

        # ---- Run query
        qp = QueryProcessorDFS()

        def distance_check(g, qgraph, chain, qchain):
            qvx0 = qchain[0]
            qvx1 = qchain[1]
            

            if qvx0 is not None and qvx1 is not None:
                depths = g.pmap['dfs.depths']
                dist = depths[chain[1]] - depths[chain[0]]
                dist_constraint = qgraph.pmap[qchain]['xdist']
                if dist_constraint == 'any':
                    return True
                else:                
                    return dist == dist_constraint
            else:
                return True
        
        matches = qp.run(
            g=g,
            qgraph=find_pattern_graph,
            path_check=distance_check,
            match_filter=match_filter
        )    

        return (matches, find_pattern_graph)
    
    def _find_iso_matches(self, *, g, find, where, max_n=0):
        find_pattern_graph = QGraph(find,                                     
                                    vx_args=self.match_policy.vx_annotation, 
                                    eg_args=self.match_policy.eg_annotation)

        # --- Get underlying graph-tool graphs
        host_gt = g.igraph
        pattern_gt = find_pattern_graph.igraph

        maps = subgraph_isomorphism(
            pattern_gt,
            host_gt,
            induced=True,
            subgraph=True,
            generator=True
        )     

        matches = []

        nmatches = 0
        for vmap in maps:

            bind = {}

            found_match = True

            qvx_to_vx = {}

            for q_ivx in pattern_gt.vertices():
                host_ivx = vmap[q_ivx]

                q_vx = find_pattern_graph.to_vx[int(q_ivx)]
                host_vx = g.to_vx[int(host_ivx)]

                if not self.match_policy.vx_check(g, find_pattern_graph, host_vx, q_vx):
                    found_match = False
                    break

                pid = find_pattern_graph.pmap[q_vx]['id']
                bind[pid] = host_vx

                qvx_to_vx[q_vx] = host_vx

            if found_match:
                for q_eg in find_pattern_graph.edges:
                    host_eg = (qvx_to_vx[q_eg[0]], qvx_to_vx[q_eg[1]])
                    if not self.match_policy.eg_check(g, find_pattern_graph, host_eg, q_eg):
                        found_match = False
                        break

            # If found match, apply policy and where filter when applicable
            if found_match and  \
               self.match_policy.check(g, find_pattern_graph, bind) \
               and (where is None or where(g, **bind)):
                matches.append(bind)
                nmatches += 1
                if max_n > 0 and nmatches >= max_n:
                    break


        return (matches, find_pattern_graph) 

    
    def _rewrite(self, *, g, rewrite, post, matches, find_pattern_graph) -> bool:


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

        print("L IDs:", L)
        print("R IDs:", R)
        print("preserved IDs:", preserved_ids)
        print("deleted IDs:", deleted_ids)
        print("created IDs:", created_ids)

        LE = set([(find_pattern_graph.pmap[s]['id'],
                   find_pattern_graph.pmap[t]['id']) for (s, t) in find_pattern_graph.edges])
        
        RE = set([(rewrite_pattern_graph.pmap[s]['id'],
                   rewrite_pattern_graph.pmap[t]['id']) for (s, t) in rewrite_pattern_graph.edges])    
        print("L edges:", LE)
        print("R edges:", RE)

        add_internal_edges    = RE - LE      
        remove_internal_edges = LE - RE  

        print("add internal edges:", add_internal_edges)
        print("remove internal edges:", remove_internal_edges)
        
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
            print(":::> rewire dict:", rewire_dict)
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
                    print("::> rewire_in: ", x_vx)
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


            if post:                
                if post(g, **bind) is True:
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
    


"""Matching strategies and domain-specific match constraints."""
from __future__ import annotations

import inspect
from abc import ABC, abstractmethod

from graph_tool.topology import subgraph_isomorphism
from heterograph.hgraph import HGraph
from heterograph.query.processor_dfs import QueryProcessorDFS
from heterograph.query.qgraph import QGraph

from .utils import deduplicate_matches


class MatchPolicy(ABC):
    """Interface for matching algorithms and semantic match constraints."""

    @staticmethod
    def vx_annotation(qgraph, vx, **kwargs):
        pass

    @staticmethod
    def eg_annotation(qgraph, eg, **kwargs):
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

    @abstractmethod
    def find_matches(self, *, g: HGraph, select: str, where=None, **kwargs):
        """Return bindings from the host graph and the parsed query graph.

        Implementations may use different search strategies, but both apply
        structural matching first and then semantic `where` filtering.
        """


class IsoMatchPolicy(MatchPolicy):
    """Match patterns using graph-tool subgraph isomorphism."""

    def find_matches(self, *, g, select, where=None, **kwargs):
        deduplicate = kwargs.get('deduplicate', True)
        max_n = kwargs.get('max_n', 0)
        pattern = QGraph(select, vx_args=self.vx_annotation, eg_args=self.eg_annotation)
        maps = subgraph_isomorphism(
            pattern.igraph, g.igraph, induced=True, subgraph=True, generator=True
        )
        matches = []
        for vmap in maps:
            bind, qvx_to_vx = {}, {}
            for q_ivx in pattern.igraph.vertices():
                q_vx = pattern.to_vx[int(q_ivx)]
                host_vx = g.to_vx[int(vmap[q_ivx])]
                if not self.vx_check(g, pattern, host_vx, q_vx):
                    break
                bind[pattern.pmap[q_vx]['id']] = host_vx
                qvx_to_vx[q_vx] = host_vx
            else:
                if all(self.eg_check(g, pattern, (qvx_to_vx[e[0]], qvx_to_vx[e[1]]), e)
                       for e in pattern.edges) and self.check(g, pattern, bind) \
                   and (where is None or where(g, **bind)):
                    matches.append(bind)
        if deduplicate:
            matches = deduplicate_matches(matches)
        if max_n > 0:
            matches = matches[:max_n]
        return matches, pattern


class DfsMatchPolicy(MatchPolicy):
    """Match patterns using Heterograph's DFS query processor."""

    def find_matches(self, *, g, select, where=None, **kwargs):
        deduplicate = kwargs.get('deduplicate', True)
        max_n = kwargs.get('max_n', 0)
        pattern = QGraph(select, vx_args=self.vx_annotation, eg_args=self.eg_annotation)
        params = set(inspect.signature(where).parameters) - {'g'} if where else set()

        def match_filter(graph, qgraph, match):
            if not self.check(graph, qgraph, match):
                return False
            return where(graph, **{k: v for k, v in match.items() if k in params}) if where else True

        def distance_check(graph, qgraph, chain, qchain):
            if qchain[0] is None or qchain[1] is None:
                return True
            constraint = qgraph.pmap[qchain]['xdist']
            return constraint == 'any' or graph.pmap['dfs.depths'][chain[1]] - graph.pmap['dfs.depths'][chain[0]] == constraint

        matches = QueryProcessorDFS().run(g=g, qgraph=pattern, path_check=distance_check,
                                          match_filter=match_filter)
        if deduplicate:
            matches = deduplicate_matches(matches)
        if max_n > 0:
            matches = matches[:max_n]
        return matches, pattern

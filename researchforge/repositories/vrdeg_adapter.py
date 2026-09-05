"""researchforge/repositories/vrdeg_adapter.py — VRDEG Graph Repository adapters.

RF-1.0.0-alpha.3 (Phase 8A): In-memory reference and SQL relational persistence for VRDEG graphs.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..vrdeg.edge import GraphEdge, RelationType
from ..vrdeg.graph import VRDEG
from ..vrdeg.node import GraphNode


class InMemoryVRDEGRepository:
    """Adapts the reference in-process VRDEG implementation to ResearchGraphRepository."""

    def __init__(self, graph: Optional[VRDEG] = None) -> None:
        self.graph = graph or VRDEG()

    def add_node(self, node: GraphNode) -> None:
        self.graph.add_node(node)

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        return self.graph.get_node(node_id)

    def add_edge(self, edge: GraphEdge) -> None:
        self.graph.add_edge(edge)

    def get_edge(self, edge_id: str) -> Optional[GraphEdge]:
        return self.graph.get_edge(edge_id)

    def get_edges_for_node(self, node_id: str) -> List[GraphEdge]:
        return self.graph.edges_for_node(node_id)

    def trace_lineage(self, start_node_id: str) -> List[GraphNode]:
        """Trace lineage back through causal/preceding relations."""
        visited = set()
        lineage = []
        queue = [start_node_id]

        while queue:
            curr_id = queue.pop(0)
            if curr_id in visited:
                continue
            visited.add(curr_id)

            node = self.graph.get_node(curr_id)
            if node is not None:
                lineage.append(node)

            # Follow incoming edges whose relations signify ancestry
            for edge in self.graph.edges_for_node(curr_id):
                if edge.target_id == curr_id:
                    if edge.source_id not in visited:
                        queue.append(edge.source_id)

        return lineage

    def trace_branch(self, branch_id: str) -> List[GraphNode]:
        """Find all nodes belonging to or connected to a research branch."""
        nodes = []
        for edge in self.graph.all_edges():
            if edge.relation == RelationType.BRANCH_OF.value and edge.target_id == branch_id:
                src_node = self.graph.get_node(edge.source_id)
                if src_node:
                    nodes.append(src_node)
        return nodes

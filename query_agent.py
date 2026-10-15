"""
QueryAgent — Day 9
Traverses the live NetworkX DiGraph to find relevant nodes and subgraphs
for a natural language question.
"""

import logging
import networkx as nx
from collections import defaultdict

logger = logging.getLogger(__name__)


class QueryAgent:

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph

    # ── Public API ─────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 10) -> list:
        """
        Keyword search over node labels and descriptions.
        Returns list of (node_id, data, score) sorted by relevance.
        """
        q_terms = [t.lower().strip() for t in query.split() if len(t) > 2]
        if not q_terms:
            return []

        results = []
        for nid, data in self.graph.nodes(data=True):
            label = data.get("label", "").lower()
            desc  = data.get("description", "").lower()
            text  = label + " " + desc

            score = 0
            for term in q_terms:
                if term in label:
                    score += 3          # label match worth more
                elif term in desc:
                    score += 1
            # bonus: node appears in many sources = more central
            n_sources = len(data.get("sources", [data.get("source", "")]))
            score += min(n_sources - 1, 3) * 0.5

            if score > 0:
                results.append((nid, data, score))

        results.sort(key=lambda x: -x[2])
        return results[:top_k]

    def get_subgraph(self, node_ids: list, radius: int = 1) -> nx.DiGraph:
        """
        Return an ego subgraph around each node_id up to `radius` hops.
        Merges all ego graphs into one DiGraph.
        """
        if not node_ids or self.graph.number_of_nodes() == 0:
            return nx.DiGraph()

        combined_nodes = set()
        for nid in node_ids:
            if self.graph.has_node(nid):
                ego = nx.ego_graph(self.graph, nid, radius=radius, undirected=True)
                combined_nodes.update(ego.nodes())

        return self.graph.subgraph(combined_nodes).copy()

    def find_path(self, source_label: str, target_label: str) -> list:
        """
        Find the shortest path between two concepts by label.
        Returns list of (node_id, label, relation) tuples.
        """
        src_id = self._label_to_id(source_label)
        tgt_id = self._label_to_id(target_label)

        if not src_id or not tgt_id:
            return []

        try:
            path = nx.shortest_path(self.graph.to_undirected(), src_id, tgt_id)
        except nx.NetworkXNoPath:
            return []
        except nx.NodeNotFound:
            return []

        result = []
        for i, nid in enumerate(path):
            data  = self.graph.nodes[nid]
            label = data.get("label", nid)
            rel   = ""
            if i < len(path) - 1:
                edge_data = self.graph.edges.get((nid, path[i+1]), {})
                if not edge_data:
                    edge_data = self.graph.edges.get((path[i+1], nid), {})
                rel = edge_data.get("relation", "→")
            result.append({"node_id": nid, "label": label, "relation": rel})
        return result

    def get_neighbors(self, node_id: str, depth: int = 1) -> dict:
        """
        Return predecessors and successors of a node up to depth hops.
        """
        if not self.graph.has_node(node_id):
            return {"in": [], "out": []}

        out_edges = []
        for _, tgt, data in self.graph.out_edges(node_id, data=True):
            tgt_label = self.graph.nodes[tgt].get("label", tgt)
            out_edges.append({
                "node_id": tgt,
                "label": tgt_label,
                "relation": data.get("relation", "related_to"),
            })

        in_edges = []
        for src, _, data in self.graph.in_edges(node_id, data=True):
            src_label = self.graph.nodes[src].get("label", src)
            in_edges.append({
                "node_id": src,
                "label": src_label,
                "relation": data.get("relation", "related_to"),
            })

        return {"in": in_edges, "out": out_edges}

    def get_node_by_label(self, label: str) -> tuple:
        """Find a node by its exact or fuzzy label. Returns (node_id, data) or (None, None)."""
        label_lower = label.lower().strip()
        # exact match first
        for nid, data in self.graph.nodes(data=True):
            if data.get("label", "").lower() == label_lower:
                return nid, data
        # partial match fallback
        for nid, data in self.graph.nodes(data=True):
            if label_lower in data.get("label", "").lower():
                return nid, data
        return None, None

    def multi_source_nodes(self, min_sources: int = 2) -> list:
        """Return nodes that appear in at least min_sources articles."""
        result = []
        for nid, data in self.graph.nodes(data=True):
            srcs = data.get("sources", [data.get("source", "")])
            if len(srcs) >= min_sources:
                result.append({
                    "node_id": nid,
                    "label": data.get("label", nid),
                    "sources": srcs,
                    "n_sources": len(srcs),
                })
        result.sort(key=lambda x: -x["n_sources"])
        return result

    def get_stats(self) -> dict:
        """Quick graph statistics for display."""
        if self.graph.number_of_nodes() == 0:
            return {}
        degrees = dict(self.graph.degree())
        top = sorted(degrees.items(), key=lambda x: -x[1])[:5]
        top_nodes = [
            {"node_id": nid, "label": self.graph.nodes[nid].get("label", nid), "degree": d}
            for nid, d in top
        ]
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "top_nodes": top_nodes,
            "avg_degree": round(sum(degrees.values()) / max(len(degrees), 1), 2),
        }

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _label_to_id(self, label: str) -> str:
        """Convert a human label to a node_id, with fuzzy fallback."""
        nid, _ = self.get_node_by_label(label)
        if nid:
            return nid
        # try direct id format
        candidate = label.lower().replace(" ", "_")
        if self.graph.has_node(candidate):
            return candidate
        return None

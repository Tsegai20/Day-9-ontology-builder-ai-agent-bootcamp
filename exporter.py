"""
Exporter — Day 9
Exports the knowledge graph in multiple formats:
  - JSON  (nodes + edges, human-readable)
  - CSV   (two files: nodes.csv, edges.csv)
  - RDF   (N-Triples plain text)
"""

import json
import csv
import io
import logging
import networkx as nx

logger = logging.getLogger(__name__)


class GraphExporter:

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph

    # ── JSON export ────────────────────────────────────────────────────────────

    def to_json(self, pretty: bool = True) -> str:
        """Export as a JSON object with nodes and edges arrays."""
        nodes = []
        for nid, data in self.graph.nodes(data=True):
            nodes.append({
                "id":          nid,
                "label":       data.get("label", nid),
                "type":        data.get("type", "concept"),
                "description": data.get("description", ""),
                "source":      data.get("source", ""),
                "sources":     data.get("sources", [data.get("source", "")]),
            })

        edges = []
        for src, tgt, data in self.graph.edges(data=True):
            edges.append({
                "source":   src,
                "target":   tgt,
                "relation": data.get("relation", "related_to"),
                "source_article": data.get("source", ""),
            })

        payload = {
            "meta": {
                "nodes": len(nodes),
                "edges": len(edges),
                "format": "ontology-builder-v1",
            },
            "nodes": nodes,
            "edges": edges,
        }

        indent = 2 if pretty else None
        return json.dumps(payload, indent=indent, ensure_ascii=False)

    # ── CSV export ─────────────────────────────────────────────────────────────

    def to_csv_nodes(self) -> str:
        """Export nodes as CSV string."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["id", "label", "type", "description", "source", "n_sources"])
        for nid, data in self.graph.nodes(data=True):
            srcs = data.get("sources", [data.get("source", "")])
            writer.writerow([
                nid,
                data.get("label", nid),
                data.get("type", "concept"),
                data.get("description", "")[:200],
                data.get("source", ""),
                len(srcs),
            ])
        return output.getvalue()

    def to_csv_edges(self) -> str:
        """Export edges as CSV string."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["source_id", "source_label", "relation", "target_id", "target_label", "article"])
        for src, tgt, data in self.graph.edges(data=True):
            src_label = self.graph.nodes[src].get("label", src)
            tgt_label = self.graph.nodes[tgt].get("label", tgt)
            writer.writerow([
                src, src_label,
                data.get("relation", "related_to"),
                tgt, tgt_label,
                data.get("source", ""),
            ])
        return output.getvalue()

    # ── RDF N-Triples export ───────────────────────────────────────────────────

    def to_rdf_ntriples(self) -> str:
        """
        Export as RDF N-Triples (plain text, no rdflib dependency).
        Each triple: <subject> <predicate> <object> .
        """
        BASE = "http://ontologybuilder.ai/entity/"
        PROP = "http://ontologybuilder.ai/property/"
        RDF  = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        RDFS = "http://www.w3.org/2000/01/rdf-schema#"

        def uri(s: str) -> str:
            safe = s.replace(" ", "_").replace('"', "").replace("<", "").replace(">", "")
            return f"<{BASE}{safe}>"

        def prop(s: str) -> str:
            safe = s.replace(" ", "_")
            return f"<{PROP}{safe}>"

        def lit(s: str) -> str:
            escaped = s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
            return f'"{escaped}"'

        lines = []
        # Node triples
        for nid, data in self.graph.nodes(data=True):
            subj  = uri(nid)
            label = data.get("label", nid)
            ntype = data.get("type", "concept")
            desc  = data.get("description", "")

            lines.append(f"{subj} <{RDFS}label> {lit(label)} .")
            lines.append(f"{subj} <{RDFS}type> {prop(ntype)} .")
            if desc:
                lines.append(f"{subj} <{RDFS}comment> {lit(desc[:300])} .")

        # Edge triples
        for src, tgt, data in self.graph.edges(data=True):
            rel = data.get("relation", "related_to")
            lines.append(f"{uri(src)} {prop(rel)} {uri(tgt)} .")

        return "\n".join(lines)

    # ── Summary stats ──────────────────────────────────────────────────────────

    def summary(self) -> dict:
        type_counts = {}
        source_counts = {}
        for _, data in self.graph.nodes(data=True):
            t = data.get("type", "concept")
            type_counts[t] = type_counts.get(t, 0) + 1
            for s in data.get("sources", [data.get("source", "")]):
                source_counts[s] = source_counts.get(s, 0) + 1

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types":  type_counts,
            "top_sources": sorted(source_counts.items(), key=lambda x: -x[1])[:10],
        }

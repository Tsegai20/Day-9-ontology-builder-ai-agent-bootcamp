"""
CoverageAnalyzer — Day 8 Part 2
Analyzes the knowledge graph for completeness, cluster density, and gaps.
"""

import logging
from collections import defaultdict
import networkx as nx

logger = logging.getLogger(__name__)

# Category colors for the dashboard
CATEGORY_COLORS = {
    "core":        "#4FC3F7",
    "method":      "#CE93D8",
    "application": "#81C784",
    "theory":      "#FFB74D",
    "related":     "#F06292",
    "unknown":     "#B0BEC5",
}


def analyze_coverage(graph: nx.DiGraph, plan_topics: list = None,
                     sources: list = None) -> dict:
    """
    Analyze graph coverage by source and node type.
    Returns structured coverage report.

    sources: list of {"source": name, "chunks": n, "entities": n, "relationships": n}
             from builder.sources — used to show all ingested sources even when
             nodes are shared/deduped across articles.
    """
    if graph.number_of_nodes() == 0:
        return {"clusters": [], "gaps": [], "score": 0,
                "total_nodes": 0, "total_edges": 0, "by_type": {}}

    # Group nodes by source attribute
    by_source = defaultdict(list)
    for nid, data in graph.nodes(data=True):
        source = data.get("source", "unknown")
        by_source[source].append(nid)

    # If builder.sources gives us more sources than node attributes captured,
    # add them with zero nodes (so they still show in the dashboard)
    if sources:
        for s in sources:
            name = s.get("source", "")
            if name and name not in by_source:
                by_source[name] = []

    # Group nodes by type
    by_type = defaultdict(list)
    for nid, data in graph.nodes(data=True):
        ntype = data.get("type", "concept")
        by_type[ntype].append(nid)

    # Per-source cluster stats
    clusters = []
    for source, nodes in sorted(by_source.items(), key=lambda x: -len(x[1])):
        subgraph = graph.subgraph(nodes)
        internal_edges = subgraph.number_of_edges()
        # Connectivity = internal edges / max possible edges
        n = len(nodes)
        max_edges = n * (n - 1) if n > 1 else 1
        density = internal_edges / max_edges
        # Cross-connections to other sources
        external = sum(
            1 for u, v in graph.edges()
            if (u in nodes) != (v in nodes)
        )
        # Completeness score (0-100)
        score = min(100, int(
            (min(n, 30) / 30) * 50 +       # node count (up to 50 pts)
            (min(density * 100, 30)) +       # density (up to 30 pts)
            (min(external, 20))              # cross-connections (up to 20 pts)
        ))
        clusters.append({
            "source": source,
            "nodes": n,
            "internal_edges": internal_edges,
            "external_edges": external,
            "density": round(density * 100, 1),
            "score": score,
            "node_ids": nodes[:5],  # sample for display
        })

    # Identify gaps — sources with low scores
    gaps = [
        {
            "topic": c["source"],
            "score": c["score"],
            "nodes": c["nodes"],
            "reason": _gap_reason(c),
        }
        for c in clusters
        if c["score"] < 60
    ]
    gaps.sort(key=lambda x: x["score"])

    # Overall completeness score
    overall = int(sum(c["score"] for c in clusters) / len(clusters)) if clusters else 0

    return {
        "clusters": clusters,
        "by_type": {k: len(v) for k, v in by_type.items()},
        "gaps": gaps,
        "score": overall,
        "total_nodes": graph.number_of_nodes(),
        "total_edges": graph.number_of_edges(),
    }


def _gap_reason(cluster: dict) -> str:
    if cluster["nodes"] == 0:
        return "No nodes attributed to this source yet — concepts may be shared with another source"
    if cluster["nodes"] < 8:
        return f"Only {cluster['nodes']} nodes — needs more content"
    if cluster["density"] < 5:
        return f"Low connectivity ({cluster['density']}%) — concepts not well linked"
    if cluster["external_edges"] < 3:
        return "Isolated — not connected to rest of graph"
    return "Below completeness threshold"


def build_coverage_chart(coverage: dict):
    """Build a Plotly bar chart showing coverage scores per source."""
    try:
        import plotly.graph_objects as go

        clusters = coverage.get("clusters", [])
        if not clusters:
            return None

        sources = [c["source"][:20] for c in clusters]
        scores  = [c["score"]      for c in clusters]
        nodes   = [c["nodes"]      for c in clusters]
        colors  = ["#4FC3F7" if s >= 70 else "#FFB74D" if s >= 40 else "#F06292"
                   for s in scores]

        fig = go.Figure()

        # Score bars
        fig.add_trace(go.Bar(
            name="Completeness Score",
            x=sources, y=scores,
            marker_color=colors,
            text=[f"{s}%" for s in scores],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Score: %{y}%<extra></extra>",
        ))

        # Node count line
        fig.add_trace(go.Scatter(
            name="Node Count",
            x=sources, y=nodes,
            mode="lines+markers",
            yaxis="y2",
            line=dict(color="#CE93D8", width=2),
            marker=dict(size=8),
            hovertemplate="<b>%{x}</b><br>Nodes: %{y}<extra></extra>",
        ))

        fig.update_layout(
            title=dict(text="Knowledge Base Coverage by Source",
                       font=dict(color="#fff", size=15)),
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="#ccc"),
            xaxis=dict(tickangle=-30, gridcolor="#333"),
            yaxis=dict(title="Completeness %", range=[0, 110], gridcolor="#333"),
            yaxis2=dict(title="Node Count", overlaying="y", side="right",
                        gridcolor="#333"),
            legend=dict(bgcolor="#1a1a2e", bordercolor="#444"),
            height=400,
            margin=dict(l=40, r=40, t=50, b=80),
            barmode="group",
        )
        return fig
    except ImportError:
        return None


def build_type_chart(coverage: dict):
    """Donut chart of node types."""
    try:
        import plotly.graph_objects as go

        by_type = coverage.get("by_type", {})
        if not by_type:
            return None

        TYPE_COLORS = {
            "concept":      "#4FC3F7",
            "technology":   "#81C784",
            "organization": "#FFB74D",
            "person":       "#F06292",
            "method":       "#CE93D8",
            "application":  "#80DEEA",
            "challenge":    "#FFCC80",
        }

        labels = list(by_type.keys())
        values = list(by_type.values())
        colors = [TYPE_COLORS.get(l, "#B0BEC5") for l in labels]

        fig = go.Figure(go.Pie(
            labels=labels, values=values,
            hole=0.5,
            marker=dict(colors=colors),
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>%{value} nodes (%{percent})<extra></extra>",
        ))
        fig.update_layout(
            title=dict(text="Node Types", font=dict(color="#fff", size=15)),
            paper_bgcolor="#0e1117",
            font=dict(color="#ccc"),
            height=350,
            margin=dict(l=20, r=20, t=50, b=20),
            showlegend=True,
            legend=dict(bgcolor="#1a1a2e"),
        )
        return fig
    except ImportError:
        return None

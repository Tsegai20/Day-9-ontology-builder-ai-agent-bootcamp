"""
Multi-Agent Ontology Builder — Day 8 + Day 9 AI Skills Bootcamp
Streamlit app with 8 tabs:
  1. 📝 Text       — paste/type text to ingest
  2. 🌐 URL        — ingest any URL or Wikipedia article
  3. 🤖 Research   — AI agent plans & fetches a whole domain automatically
  4. 📊 Coverage   — completeness dashboard + gap analysis
  5. 🕸️ Graph      — interactive 3-D knowledge graph
  6. 🔍 Search     — explore nodes and edges
  7. 💬 Ask        — RAG chat over the knowledge graph  [Day 9]
  8. 📤 Export     — download JSON / CSV / RDF          [Day 9]

Colab launch:
    !streamlit run /content/ontology_agent/app.py \
        --server.enableWebsocketCompression=false \
        --server.enableCORS=false \
        --server.enableXsrfProtection=false &
    from google.colab.output import eval_js
    print(eval_js("google.colab.kernel.proxyPort(8501)"))
"""

import os
import sys
import time
import streamlit as st

# ── path setup ────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from builder  import OntologyBuilder
from planner  import ResearchPlanner
from coverage import analyze_coverage, build_coverage_chart, build_type_chart

# Day 9 imports — graceful fallback if files not present yet
try:
    from rag_agent import RAGAgent
    from query_agent import QueryAgent
    from exporter import GraphExporter
    DAY9_AVAILABLE = True
except ImportError:
    DAY9_AVAILABLE = False

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent Ontology Builder",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── dark theme CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background: #0e1117; color: #e0e0e0; }
  .stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: #1a1a2e; border-radius: 10px; padding: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px; padding: 8px 18px; color: #aaa; font-size: 14px;
  }
  .stTabs [aria-selected="true"] {
    background: #4FC3F7 !important; color: #000 !important; font-weight: 700;
  }
  .metric-card {
    background: #1a1a2e; border: 1px solid #333; border-radius: 12px;
    padding: 18px; text-align: center; margin: 4px;
  }
  .metric-card h2 { color: #4FC3F7; margin: 0; font-size: 2em; }
  .metric-card p  { color: #aaa; margin: 4px 0 0 0; font-size: 0.85em; }
  .plan-card {
    background: #12122a; border-left: 4px solid #4FC3F7;
    border-radius: 8px; padding: 14px 18px; margin: 8px 0;
  }
  .plan-card.method      { border-color: #CE93D8; }
  .plan-card.application { border-color: #81C784; }
  .plan-card.theory      { border-color: #FFB74D; }
  .plan-card.related     { border-color: #F06292; }
  .gap-card {
    background: #1a0f0f; border-left: 4px solid #F06292;
    border-radius: 8px; padding: 14px 18px; margin: 8px 0;
  }
  .gap-card.warn { border-color: #FFB74D; }
  .source-badge {
    display: inline-block; background: #1a1a2e; border: 1px solid #4FC3F7;
    border-radius: 20px; padding: 2px 12px; font-size: 0.75em;
    color: #4FC3F7; margin: 2px;
  }
  /* Day 9 chat styles */
  .chat-user {
    background: #1a2a3a; border-left: 3px solid #4FC3F7;
    border-radius: 8px; padding: 10px 14px; margin: 6px 0;
  }
  .chat-ai {
    background: #12122a; border-left: 3px solid #81C784;
    border-radius: 8px; padding: 10px 14px; margin: 6px 0;
  }
  .chat-sources {
    font-size: 0.78em; color: #888; margin-top: 6px;
  }
  .export-card {
    background: #12122a; border: 1px solid #333; border-radius: 10px;
    padding: 16px 20px; margin: 8px 0;
  }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Session helpers
# ══════════════════════════════════════════════════════════════════════════════

def _init_builder(api_key: str) -> OntologyBuilder:
    """Create or reload builder; always reload graph from disk."""
    if ("builder" not in st.session_state
            or st.session_state.get("api_key") != api_key):
        st.session_state.builder = OntologyBuilder.load_graph(api_key)
        st.session_state.api_key  = api_key
    else:
        b = OntologyBuilder.load_graph(api_key)
        st.session_state.builder.graph   = b.graph
        st.session_state.builder.sources = b.sources
    return st.session_state.builder


def _get_builder() -> OntologyBuilder:
    return st.session_state.get("builder")


def _get_rag_agent(api_key: str, graph) -> "RAGAgent":
    """Get or create RAGAgent; always synced to the live graph."""
    if "rag_agent" not in st.session_state:
        st.session_state.rag_agent = RAGAgent(api_key, graph)
    else:
        st.session_state.rag_agent.update_graph(graph)
    return st.session_state.rag_agent


# ══════════════════════════════════════════════════════════════════════════════
#  Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🧠 Ontology Builder")
    st.markdown("*Day 8+9 — AI Skills Bootcamp*")
    st.divider()

    api_key = st.text_input(
        "Groq API Key", type="password",
        placeholder="gsk_...",
        help="Get your key at console.groq.com",
    )

    if api_key:
        builder = _init_builder(api_key)
        stats = builder.get_stats()

        st.markdown("### 📈 Graph Stats")
        col1, col2 = st.columns(2)
        col1.metric("Nodes",   stats["nodes"])
        col2.metric("Edges",   stats["edges"])
        col1.metric("Sources", stats["sources"])
        col2.metric("Chunks",  stats["chunks"])

        st.divider()
        if st.button("🗑️ Clear Graph", use_container_width=True,
                     help="Delete all nodes and start fresh"):
            builder.graph.clear()
            builder.sources.clear()
            builder._save_graph()
            # also reset RAG agent
            st.session_state.pop("rag_agent", None)
            st.session_state.pop("chat_history", None)
            st.success("Graph cleared!")
            st.rerun()
    else:
        st.info("Enter your Groq API key to begin.")

    st.divider()
    st.markdown("""
**Quick Start**
1. Enter API key above
2. Go to 🤖 Research tab
3. Describe your domain
4. Let the AI build your knowledge base
5. Ask questions in 💬 Ask tab
    """)


# ── guard: no API key ─────────────────────────────────────────────────────────
if not api_key:
    st.title("🧠 Multi-Agent Ontology Builder")
    st.info("👈 Enter your **Groq API key** in the sidebar to get started.")
    st.stop()

builder = _get_builder()


# ══════════════════════════════════════════════════════════════════════════════
#  Tabs
# ══════════════════════════════════════════════════════════════════════════════

(tab_text, tab_url, tab_research, tab_coverage,
 tab_graph, tab_search, tab_ask, tab_export) = st.tabs([
    "📝 Text",
    "🌐 URL / Wikipedia",
    "🤖 Research Planner",
    "📊 Coverage",
    "🕸️ Graph",
    "🔍 Search",
    "💬 Ask the Graph",
    "📤 Export",
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — TEXT
# ══════════════════════════════════════════════════════════════════════════════

with tab_text:
    st.header("📝 Ingest Text")
    source_name = st.text_input(
        "Source name", value="Manual Text",
        placeholder="e.g. My Research Notes",
    )
    text_input = st.text_area(
        "Paste text here", height=280,
        placeholder="Paste any text — articles, notes, reports…",
    )
    if st.button("⚡ Extract Knowledge", use_container_width=True,
                 disabled=not text_input.strip()):
        with st.spinner("Extracting knowledge…"):
            result = builder.ingest_text(text_input.strip(), source_name)
        st.success(
            f"✅ Done — {result['entities']} entities, "
            f"{result['relationships']} relationships across {result['chunks']} chunks"
        )
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — URL
# ══════════════════════════════════════════════════════════════════════════════

with tab_url:
    st.header("🌐 Ingest URL / Wikipedia")
    st.markdown(
        "Paste any URL. Wikipedia articles are fetched as full plaintext (~34 k chars)."
    )

    url_col, btn_col = st.columns([4, 1])
    with url_col:
        url_input = st.text_input(
            "URL", label_visibility="collapsed",
            placeholder="https://en.wikipedia.org/wiki/Federated_learning",
        )
    with btn_col:
        ingest_url_btn = st.button(
            "⚡ Ingest", use_container_width=True,
            disabled=not url_input.strip(),
        )

    st.caption(
        "💡 Example: `https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)`"
    )

    if ingest_url_btn and url_input.strip():
        with st.spinner(f"Fetching {url_input.strip()}…"):
            result = builder.ingest_url(url_input.strip())
        if result.get("error"):
            st.error(f"❌ {result['error']}")
        else:
            st.success(
                f"✅ Ingested **{result['source']}** — "
                f"{result['entities']} entities, {result['relationships']} rels"
            )
        st.rerun()

    if builder.sources:
        st.divider()
        st.markdown("### Recently Ingested")
        for s in reversed(builder.sources[-8:]):
            st.markdown(
                f"- **{s['source']}** — "
                f"{s.get('entities',0)} entities · "
                f"{s.get('relationships',0)} rels · "
                f"{s.get('chunks',0)} chunks"
            )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — RESEARCH PLANNER
# ══════════════════════════════════════════════════════════════════════════════

with tab_research:
    st.header("🤖 Research Planner Agent")
    st.markdown(
        "Describe your domain. The AI will choose which Wikipedia articles to fetch, "
        "show you the plan, and — once you approve — build the knowledge base automatically."
    )

    domain_input = st.text_area(
        "Domain description",
        height=100,
        placeholder=(
            "e.g. Federated learning and privacy-preserving machine learning — "
            "I want to understand the core methods, key applications in healthcare "
            "and finance, and the main privacy techniques like differential privacy."
        ),
    )

    if st.button(
        "🧠 Generate Research Plan", use_container_width=True,
        disabled=not domain_input.strip(),
    ):
        planner = ResearchPlanner(api_key)
        with st.spinner("Thinking about what to fetch…"):
            plan = planner.generate_plan(domain_input.strip())

        if not plan.get("topics"):
            st.error("Could not generate a plan. Try rephrasing your domain description.")
        else:
            with st.spinner("Verifying Wikipedia articles exist…"):
                plan["topics"] = planner.verify_topics(plan["topics"])
            st.session_state["research_plan"]   = plan
            st.session_state["research_domain"] = domain_input.strip()

    # ── display plan ──────────────────────────────────────────────────────────
    plan = st.session_state.get("research_plan")
    if plan:
        st.divider()
        st.markdown("### 📋 Research Plan")
        st.info(plan.get("summary", ""))

        topics  = plan.get("topics", [])
        valid   = [t for t in topics if t.get("exists", True)]
        invalid = [t for t in topics if not t.get("exists", True)]

        CAT_COLORS = {
            "core":        "#4FC3F7",
            "method":      "#CE93D8",
            "application": "#81C784",
            "theory":      "#FFB74D",
            "related":     "#F06292",
        }

        cols = st.columns(2)
        for i, t in enumerate(valid):
            cat   = t.get("category", "core")
            color = CAT_COLORS.get(cat, "#aaa")
            with cols[i % 2]:
                st.markdown(
                    f'<div class="plan-card {cat}">'
                    f'<span style="color:{color};font-weight:700">[{cat.upper()}]</span>'
                    f'<span style="color:#fff;font-size:1.05em;margin-left:8px">'
                    f'{t["title"].replace("_"," ")}</span><br>'
                    f'<span style="color:#aaa;font-size:0.85em">'
                    f'{t.get("reason","")}</span></div>',
                    unsafe_allow_html=True,
                )

        if invalid:
            with st.expander(f"⚠️ {len(invalid)} articles not found (will be skipped)"):
                for t in invalid:
                    st.markdown(f"- ~~{t['title'].replace('_',' ')}~~")

        st.markdown(f"**{len(valid)} articles ready to fetch.**")

        st.divider()
        if st.button(
            f"🚀 Approve & Fetch All {len(valid)} Articles",
            use_container_width=True,
            type="primary",
            disabled=len(valid) == 0,
        ):
            progress_bar = st.progress(0, text="Starting…")
            status_box   = st.empty()

            for idx, topic in enumerate(valid):
                url   = topic["url"]
                title = topic["title"].replace("_", " ")
                frac  = idx / len(valid)
                progress_bar.progress(frac,
                    text=f"Fetching {idx+1}/{len(valid)}: {title}")
                status_box.info(f"⏳ Ingesting **{title}**…")

                res = builder.ingest_url(url, source_name=title)
                if res.get("error"):
                    status_box.warning(f"⚠️ {title}: {res['error']}")
                else:
                    status_box.success(
                        f"✅ {title} — "
                        f"{res.get('entities',0)} entities, "
                        f"{res.get('relationships',0)} rels"
                    )
                time.sleep(0.3)  # polite crawl delay

            progress_bar.progress(1.0, text="✅ Complete!")
            status_box.empty()
            st.success(f"✅ Ingested {len(valid)} articles into the knowledge graph!")
            st.session_state.pop("research_plan", None)
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — COVERAGE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

with tab_coverage:
    st.header("📊 Coverage Dashboard")

    if builder.node_count == 0:
        st.info("No graph yet — add some sources in the other tabs first.")
    else:
        coverage = analyze_coverage(builder.graph, sources=builder.sources)

        # ── top metrics ───────────────────────────────────────────────────────
        score_color = (
            "#81C784" if coverage["score"] >= 70
            else "#FFB74D" if coverage["score"] >= 40
            else "#F06292"
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(
            f'<div class="metric-card">'
            f'<h2>{coverage["total_nodes"]}</h2><p>Total Nodes</p></div>',
            unsafe_allow_html=True,
        )
        c2.markdown(
            f'<div class="metric-card">'
            f'<h2>{coverage["total_edges"]}</h2><p>Total Edges</p></div>',
            unsafe_allow_html=True,
        )
        c3.markdown(
            f'<div class="metric-card">'
            f'<h2>{len(coverage["clusters"])}</h2><p>Source Clusters</p></div>',
            unsafe_allow_html=True,
        )
        c4.markdown(
            f'<div class="metric-card">'
            f'<h2 style="color:{score_color}">{coverage["score"]}%</h2>'
            f'<p>Overall Score</p></div>',
            unsafe_allow_html=True,
        )

        st.divider()

        # ── charts ────────────────────────────────────────────────────────────
        chart_col, type_col = st.columns([3, 2])
        with chart_col:
            fig = build_coverage_chart(coverage)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        with type_col:
            fig2 = build_type_chart(coverage)
            if fig2:
                st.plotly_chart(fig2, use_container_width=True)

        # ── per-cluster table ─────────────────────────────────────────────────
        st.divider()
        st.markdown("### 🗂️ Cluster Details")
        for c in coverage["clusters"]:
            sc  = c["score"]
            col = "🟢" if sc >= 70 else "🟡" if sc >= 40 else "🔴"
            with st.expander(
                f"{col} **{c['source']}** — {sc}% · "
                f"{c['nodes']} nodes · {c['internal_edges']} internal · "
                f"{c['external_edges']} cross-links"
            ):
                cols = st.columns(3)
                cols[0].metric("Nodes",       c["nodes"])
                cols[1].metric("Density",     f"{c['density']}%")
                cols[2].metric("Cross-links", c["external_edges"])
                st.progress(
                    sc / 100,
                    text=(
                        f"Completeness: {sc}%  "
                        + ("✅ Strong" if sc >= 70
                           else "⚠️ Needs more content" if sc >= 40
                           else "🔴 Sparse")
                    ),
                )

        # ── gap analysis ──────────────────────────────────────────────────────
        gaps = coverage.get("gaps", [])
        if gaps:
            st.divider()
            st.markdown("### 🕳️ Knowledge Gaps")
            st.caption(
                "These clusters have low completeness scores "
                "and would benefit from more content."
            )
            planner = ResearchPlanner(api_key)

            for gap in gaps:
                sev_cls = "gap-card warn" if gap["score"] >= 30 else "gap-card"
                st.markdown(
                    f'<div class="{sev_cls}">'
                    f'<b style="color:#F06292">{gap["topic"]}</b>'
                    f' — score {gap["score"]}% · {gap["nodes"]} nodes<br>'
                    f'<span style="color:#aaa;font-size:0.85em">'
                    f'{gap["reason"]}</span></div>',
                    unsafe_allow_html=True,
                )

                fill_key = f"fill__{gap['topic'].replace(' ','_')}"
                if st.button(
                    f"🔧 Fill this gap: {gap['topic'][:40]}",
                    key=fill_key,
                ):
                    existing_nodes = [
                        d.get("label", n)
                        for n, d in builder.graph.nodes(data=True)
                    ][:30]
                    with st.spinner("Planning gap-fill articles…"):
                        try:
                            suggestions = planner.generate_gap_fill_plan(
                                gap["topic"], existing_nodes, n=4
                            )
                        except Exception:
                            suggestions = []
                    if not suggestions:
                        st.warning("Could not generate gap-fill suggestions.")
                    else:
                        prog = st.progress(0)
                        for i, s in enumerate(suggestions):
                            prog.progress(
                                (i + 1) / len(suggestions),
                                text=f"Fetching {s['title'].replace('_',' ')}…",
                            )
                            res = builder.ingest_url(
                                s["url"],
                                source_name=s["title"].replace("_", " "),
                            )
                            if res.get("error"):
                                st.warning(f"⚠️ {s['title']}: {res['error']}")
                            else:
                                st.markdown(
                                    f"- ✅ **{s['title'].replace('_',' ')}** — "
                                    f"{res.get('entities',0)} entities  "
                                    f"*({s.get('reason','')})*"
                                )
                        prog.progress(1.0, text="Done!")
                        st.rerun()
        else:
            st.success(
                "🎉 No significant gaps — your knowledge base looks well-covered!"
            )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — GRAPH
# ══════════════════════════════════════════════════════════════════════════════

with tab_graph:
    st.header("🕸️ Knowledge Graph")

    if builder.node_count == 0:
        st.info("Graph is empty. Add some sources to visualize.")
    else:
        st.markdown(
            f"**{builder.node_count} nodes · {builder.edge_count} edges** "
            f"from {len(builder.sources)} sources"
        )
        try:
            import plotly.graph_objects as go
            import networkx as nx

            g   = builder.graph
            pos = nx.spring_layout(g, dim=3, seed=42, k=0.5)

            TYPE_COLORS = {
                "concept":      "#4FC3F7",
                "technology":   "#81C784",
                "organization": "#FFB74D",
                "person":       "#F06292",
                "method":       "#CE93D8",
                "application":  "#80DEEA",
                "challenge":    "#FFCC80",
            }

            nodes  = list(g.nodes(data=True))
            xs     = [pos[n][0] for n, _ in nodes]
            ys     = [pos[n][1] for n, _ in nodes]
            zs     = [pos[n][2] for n, _ in nodes]
            colors = [TYPE_COLORS.get(d.get("type","concept"), "#aaa") for _, d in nodes]
            sizes  = [6 + min(g.degree(n) * 1.5, 20) for n, _ in nodes]
            hover  = [
                f"<b>{d.get('label',n)}</b><br>"
                f"Type: {d.get('type','?')}<br>"
                f"Degree: {g.degree(n)}<br>"
                f"Source: {d.get('source','?')}<br>"
                f"{d.get('description','')[:100]}"
                for n, d in nodes
            ]

            edge_x, edge_y, edge_z = [], [], []
            for u, v in g.edges():
                if u in pos and v in pos:
                    edge_x += [pos[u][0], pos[v][0], None]
                    edge_y += [pos[u][1], pos[v][1], None]
                    edge_z += [pos[u][2], pos[v][2], None]

            fig = go.Figure()
            fig.add_trace(go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode="lines",
                line=dict(width=1, color="#444"),
                hoverinfo="none",
                name="edges",
            ))
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="markers",
                marker=dict(
                    size=sizes, color=colors, opacity=0.85,
                    line=dict(width=0.5, color="#222"),
                ),
                text=hover,
                hovertemplate="%{text}<extra></extra>",
                name="nodes",
            ))
            fig.update_layout(
                scene=dict(
                    bgcolor="#0e1117",
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                ),
                paper_bgcolor="#0e1117",
                margin=dict(l=0, r=0, t=0, b=0),
                height=620,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Legend
            legend_cols = st.columns(len(TYPE_COLORS))
            for col, (t, c) in zip(legend_cols, TYPE_COLORS.items()):
                col.markdown(
                    f'<span style="color:{c}">⬤</span> '
                    f'<span style="font-size:0.75em">{t}</span>',
                    unsafe_allow_html=True,
                )

        except ImportError:
            st.error("Plotly not installed. Run: `pip install plotly`")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 6 — SEARCH
# ══════════════════════════════════════════════════════════════════════════════

with tab_search:
    st.header("🔍 Search & Explore")

    if builder.node_count == 0:
        st.info("Graph is empty.")
    else:
        query = st.text_input(
            "Search nodes",
            placeholder="e.g. federated learning, privacy…",
        )

        all_nodes = list(builder.graph.nodes(data=True))

        if query.strip():
            q = query.lower()
            results = [
                (nid, data) for nid, data in all_nodes
                if (q in nid
                    or q in data.get("label", "").lower()
                    or q in data.get("description", "").lower()
                    or q in data.get("type", "").lower())
            ]
        else:
            results = sorted(
                all_nodes,
                key=lambda x: builder.graph.degree(x[0]),
                reverse=True,
            )[:30]

        st.caption(
            f"Showing {len(results)} "
            f"{'results' if query.strip() else 'top nodes by degree'}"
        )

        TYPE_COLORS = {
            "concept":      "#4FC3F7",
            "technology":   "#81C784",
            "organization": "#FFB74D",
            "person":       "#F06292",
            "method":       "#CE93D8",
            "application":  "#80DEEA",
        }

        for nid, data in results:
            deg   = builder.graph.degree(nid)
            label = data.get("label", nid)
            ntype = data.get("type", "concept")
            desc  = data.get("description", "")
            src   = data.get("source", "")
            color = TYPE_COLORS.get(ntype, "#aaa")

            with st.expander(f"[{ntype}] **{label}** — degree {deg}"):
                st.markdown(
                    f'<span style="color:{color}">●</span> '
                    f'**Type:** {ntype} &nbsp;·&nbsp; '
                    f'**Degree:** {deg} &nbsp;·&nbsp; '
                    f'**Source:** {src}',
                    unsafe_allow_html=True,
                )
                if desc:
                    st.markdown(f"*{desc}*")

                neighbors = (
                    list(builder.graph.successors(nid)) +
                    list(builder.graph.predecessors(nid))
                )
                if neighbors:
                    st.markdown("**Connected to:**")
                    nb_labels = [
                        builder.graph.nodes[nb].get("label", nb)
                        for nb in neighbors[:10]
                    ]
                    st.markdown(", ".join(f"`{l}`" for l in nb_labels))

                out_edges = list(builder.graph.out_edges(nid, data=True))
                if out_edges:
                    st.markdown("**Outgoing relations:**")
                    for _, t, ed in out_edges[:8]:
                        tgt_label = builder.graph.nodes[t].get("label", t)
                        st.markdown(
                            f"→ **{ed.get('relation','related_to')}** → `{tgt_label}`"
                        )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 7 — ASK THE GRAPH  (Day 9)
# ══════════════════════════════════════════════════════════════════════════════

with tab_ask:
    st.header("💬 Ask the Graph")
    st.markdown(
        "Ask any question about your knowledge base. "
        "The RAG agent retrieves relevant concepts and relationships, "
        "then answers with citations from the graph."
    )

    if not DAY9_AVAILABLE:
        st.warning(
            "⚠️ Day 9 modules not found. "
            "Make sure `rag_agent.py`, `query_agent.py`, and `exporter.py` "
            "are in the same folder as `app.py`."
        )
        st.stop()

    if builder.node_count == 0:
        st.info("📭 Your graph is empty. Add articles first using the Research Planner or URL tab.")
    else:
        # ── RAG settings ──────────────────────────────────────────────────────
        with st.expander("⚙️ RAG Settings", expanded=False):
            radius = st.slider(
                "Subgraph expansion radius",
                min_value=1, max_value=3, value=1,
                help="How many hops to expand around matched nodes. "
                     "Higher = more context but slower.",
            )
            show_context = st.checkbox(
                "Show retrieved context nodes", value=False,
                help="Show which graph nodes were used to build the answer.",
            )

        # ── chat history init ─────────────────────────────────────────────────
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        rag = _get_rag_agent(api_key, builder.graph)

        # ── render existing chat ──────────────────────────────────────────────
        chat_container = st.container()
        with chat_container:
            if not st.session_state.chat_history:
                st.markdown(
                    '<div style="color:#888;font-style:italic;padding:20px 0;">'
                    "No questions yet. Ask anything about your knowledge graph below!"
                    "</div>",
                    unsafe_allow_html=True,
                )
            for entry in st.session_state.chat_history:
                # User bubble
                st.markdown(
                    f'<div class="chat-user">'
                    f'<span style="color:#4FC3F7;font-weight:700">You</span><br>'
                    f'{entry["question"]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                # AI bubble
                st.markdown(
                    f'<div class="chat-ai">'
                    f'<span style="color:#81C784;font-weight:700">🧠 Graph Agent</span><br>'
                    f'{entry["answer"]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                # Sources
                if entry.get("sources"):
                    src_badges = "".join(
                        f'<span class="source-badge">{s}</span>'
                        for s in entry["sources"]
                    )
                    st.markdown(
                        f'<div class="chat-sources">📚 Sources: {src_badges}</div>',
                        unsafe_allow_html=True,
                    )
                # Context nodes (optional)
                if show_context and entry.get("context_nodes"):
                    with st.expander(
                        f"🔎 {entry['nodes_used']} context nodes retrieved", expanded=False
                    ):
                        node_labels = [label for _, label in entry["context_nodes"][:20]]
                        st.markdown(", ".join(f"`{l}`" for l in node_labels))

        st.divider()

        # ── input bar ─────────────────────────────────────────────────────────
        q_col, btn_col = st.columns([5, 1])
        with q_col:
            user_question = st.text_input(
                "Ask a question",
                label_visibility="collapsed",
                placeholder="e.g. How does federated learning preserve privacy?",
                key="ask_input",
            )
        with btn_col:
            ask_btn = st.button(
                "Ask ➤", use_container_width=True,
                disabled=not user_question.strip(),
                type="primary",
            )

        # ── quick example buttons ─────────────────────────────────────────────
        st.caption("💡 Try:")
        ex_col1, ex_col2, ex_col3 = st.columns(3)
        ex1 = ex_col1.button("What are the main concepts?",   key="ex1")
        ex2 = ex_col2.button("How are these topics related?", key="ex2")
        ex3 = ex_col3.button("What are the key challenges?",  key="ex3")

        question_to_ask = None
        if ask_btn and user_question.strip():
            question_to_ask = user_question.strip()
        elif ex1:
            question_to_ask = "What are the main concepts in the knowledge graph?"
        elif ex2:
            question_to_ask = "How are the main topics in this knowledge graph related to each other?"
        elif ex3:
            question_to_ask = "What are the key challenges and open problems described in the graph?"

        if question_to_ask:
            with st.spinner("🔍 Searching graph and generating answer…"):
                result = rag.ask(question_to_ask, radius=radius)

            entry = {
                "question":     question_to_ask,
                "answer":       result["answer"],
                "sources":      result["sources"],
                "nodes_used":   result["nodes_used"],
                "context_nodes": result["context_nodes"],
            }
            st.session_state.chat_history.append(entry)
            st.rerun()

        # ── clear chat ────────────────────────────────────────────────────────
        if st.session_state.chat_history:
            st.divider()
            if st.button("🗑️ Clear conversation", use_container_width=False):
                st.session_state.chat_history = []
                rag.clear_history()
                st.rerun()

        # ── graph stats strip ─────────────────────────────────────────────────
        st.divider()
        st.markdown("##### 🔬 Graph Quick Stats")
        if DAY9_AVAILABLE:
            qagent = QueryAgent(builder.graph)
            stats  = qagent.get_stats()
            if stats:
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("Nodes", stats["nodes"])
                sc2.metric("Edges", stats["edges"])
                sc3.metric("Avg Degree", stats["avg_degree"])

                st.markdown("**Most connected concepts:**")
                for item in stats.get("top_nodes", []):
                    st.markdown(
                        f"- **{item['label']}** — degree {item['degree']}"
                    )

            # multi-source nodes
            multi = qagent.multi_source_nodes(min_sources=2)
            if multi:
                with st.expander(
                    f"🌐 {len(multi)} concepts appear in multiple sources"
                ):
                    for item in multi[:15]:
                        srcs = ", ".join(item["sources"][:4])
                        st.markdown(
                            f"- **{item['label']}** — "
                            f"{item['n_sources']} sources: *{srcs}*"
                        )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 8 — EXPORT  (Day 9)
# ══════════════════════════════════════════════════════════════════════════════

with tab_export:
    st.header("📤 Export Knowledge Graph")
    st.markdown(
        "Download your knowledge graph in multiple formats "
        "for use in other tools, databases, or research pipelines."
    )

    if not DAY9_AVAILABLE:
        st.warning(
            "⚠️ Day 9 modules not found. "
            "Make sure `exporter.py` is in the same folder as `app.py`."
        )
        st.stop()

    if builder.node_count == 0:
        st.info("📭 Graph is empty. Add articles first.")
    else:
        exporter = GraphExporter(builder.graph)
        summary  = exporter.summary()

        # ── summary metrics ───────────────────────────────────────────────────
        e1, e2, e3 = st.columns(3)
        e1.metric("Nodes", summary["total_nodes"])
        e2.metric("Edges", summary["total_edges"])
        e3.metric("Node Types", len(summary["node_types"]))

        # Type breakdown
        if summary["node_types"]:
            type_md = " · ".join(
                f"**{k}** {v}" for k, v in sorted(
                    summary["node_types"].items(), key=lambda x: -x[1]
                )
            )
            st.caption(f"Types: {type_md}")

        st.divider()

        # ── JSON export ───────────────────────────────────────────────────────
        st.markdown(
            '<div class="export-card">'
            '<h4 style="color:#4FC3F7;margin:0 0 6px 0">📄 JSON</h4>'
            '<p style="color:#aaa;margin:0 0 10px 0;font-size:0.85em">'
            'Complete graph with metadata — nodes, edges, descriptions, sources. '
            'Best for loading into Python, JavaScript, or other graph tools.'
            '</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        json_pretty = st.checkbox("Pretty-print JSON (readable but larger)", value=True)
        json_data   = exporter.to_json(pretty=json_pretty)
        st.download_button(
            label="⬇️ Download graph.json",
            data=json_data,
            file_name="ontology_graph.json",
            mime="application/json",
            use_container_width=True,
        )

        st.divider()

        # ── CSV export ────────────────────────────────────────────────────────
        st.markdown(
            '<div class="export-card">'
            '<h4 style="color:#81C784;margin:0 0 6px 0">📊 CSV (2 files)</h4>'
            '<p style="color:#aaa;margin:0 0 10px 0;font-size:0.85em">'
            'Separate files for nodes and edges — import into Excel, Google Sheets, '
            'Gephi, or any CSV-compatible tool.'
            '</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        csv_col1, csv_col2 = st.columns(2)
        with csv_col1:
            nodes_csv = exporter.to_csv_nodes()
            st.download_button(
                label="⬇️ Download nodes.csv",
                data=nodes_csv,
                file_name="ontology_nodes.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.caption(
                f"{summary['total_nodes']} rows · "
                "columns: id, label, type, description, source, n_sources"
            )
        with csv_col2:
            edges_csv = exporter.to_csv_edges()
            st.download_button(
                label="⬇️ Download edges.csv",
                data=edges_csv,
                file_name="ontology_edges.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.caption(
                f"{summary['total_edges']} rows · "
                "columns: source_id, source_label, relation, target_id, target_label, article"
            )

        st.divider()

        # ── RDF export ────────────────────────────────────────────────────────
        st.markdown(
            '<div class="export-card">'
            '<h4 style="color:#CE93D8;margin:0 0 6px 0">🔗 RDF N-Triples</h4>'
            '<p style="color:#aaa;margin:0 0 10px 0;font-size:0.85em">'
            'Semantic Web format compatible with SPARQL, Protégé, Wikidata, '
            'and any RDF triple store. No extra library needed.'
            '</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        rdf_data = exporter.to_rdf_ntriples()
        st.download_button(
            label="⬇️ Download graph.nt (RDF N-Triples)",
            data=rdf_data,
            file_name="ontology_graph.nt",
            mime="text/plain",
            use_container_width=True,
        )
        n_triples = rdf_data.count("\n") + 1
        st.caption(
            f"~{n_triples:,} triples · "
            "Base URI: http://ontologybuilder.ai/entity/"
        )

        st.divider()

        # ── preview ───────────────────────────────────────────────────────────
        with st.expander("👁️ Preview exports (first 20 lines each)"):
            st.markdown("**JSON (first 30 lines)**")
            json_lines = json_data.split("\n")[:30]
            st.code("\n".join(json_lines), language="json")

            st.markdown("**Nodes CSV (first 10 rows)**")
            csv_lines = nodes_csv.split("\n")[:11]
            st.code("\n".join(csv_lines), language="text")

            st.markdown("**RDF N-Triples (first 15 triples)**")
            rdf_lines = rdf_data.split("\n")[:15]
            st.code("\n".join(rdf_lines), language="text")

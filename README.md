# 🧠 Multi-Agent Ontology Builder
### AI Skills Bootcamp — Day 8 + Day 9

A multi-agent system that builds a living knowledge graph from any domain,
then lets you query it conversationally using Retrieval-Augmented Generation (RAG).

---

## 🚀 Quick Start (Google Colab)

### Day 9 — Full app (recommended)
1. Open a new [Google Colab](https://colab.research.google.com) notebook
2. Paste the entire contents of `DAY9_DEPLOY.py` into a code cell
3. Run the cell — a Streamlit URL will appear in ~15 seconds
4. Enter your [Groq API key](https://console.groq.com) in the sidebar

### Day 8 — Original build
Same steps using `DAY8_REDEPLOY.py`

---

## 🗂️ What's Inside

| File | Description |
|------|-------------|
| `DAY9_DEPLOY.py` | Single-cell Colab deployer — embeds all 7 source files |
| `DAY8_REDEPLOY.py` | Day 8 deployer (graph builder only) |
| `ontology_agent/app.py` | Main Streamlit app — 8 tabs |
| `ontology_agent/builder.py` | Knowledge graph builder (entity + relationship extraction) |
| `ontology_agent/planner.py` | Research planner agent (Wikipedia article selection) |
| `ontology_agent/coverage.py` | Coverage dashboard + gap analysis |
| `ontology_agent/query_agent.py` | Graph traversal agent (Day 9) |
| `ontology_agent/rag_agent.py` | RAG agent — retrieval + LLM answering (Day 9) |
| `ontology_agent/exporter.py` | Graph export: JSON, CSV, RDF N-Triples (Day 9) |

---

## 🧩 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit App (8 tabs)                │
├──────────┬──────────┬──────────┬──────────┬─────────────┤
│  Text /  │ Research │ Coverage │  Graph   │  Ask (RAG)  │
│   URL    │ Planner  │Dashboard │  3D Viz  │  + Export   │
└────┬─────┴────┬─────┴────┬─────┴────┬─────┴──────┬──────┘
     │          │          │          │             │
     ▼          ▼          ▼          ▼             ▼
 OntologyBuilder        Coverage   NetworkX    RAGAgent
 (Groq LLM extract)     Analyzer   DiGraph     QueryAgent
                                               GraphExporter
```

**Day 8 agents:** OntologyBuilder · ResearchPlanner · CoverageAnalyzer

**Day 9 agents:** QueryAgent · RAGAgent · GraphExporter

All agents share one live `NetworkX DiGraph` via Streamlit session state —
no inter-process communication needed.

---

## 📋 App Tabs

| Tab | What it does |
|-----|-------------|
| 📝 Text | Paste any text to extract knowledge |
| 🌐 URL / Wikipedia | Ingest any URL or Wikipedia article |
| 🤖 Research Planner | AI selects and fetches a full article set for your domain |
| 📊 Coverage | Completeness score, cluster analysis, gap detection |
| 🕸️ Graph | Interactive 3-D knowledge graph visualization |
| 🔍 Search | Keyword search over nodes and edges |
| 💬 Ask the Graph | RAG chat — ask questions, get cited answers from the graph |
| 📤 Export | Download graph as JSON, CSV (nodes + edges), or RDF N-Triples |

---

## 🔧 How RAG Works

```
Question
   │
   ▼
QueryAgent.search()          ← keyword match on node labels + descriptions
   │
   ▼
QueryAgent.get_subgraph()    ← ego-graph expansion (radius 1–3 hops)
   │
   ▼
RAGAgent._build_context()    ← format nodes + edges as text context
   │
   ▼
Groq LLM (qwen3.8-27b)       ← answer strictly from graph context
   │
   ▼
Answer + source citations
```

---

## 📦 Graph Export Formats

- **JSON** — full graph with node metadata, edge relations, source attribution
- **CSV** — `nodes.csv` + `edges.csv` for Excel, Gephi, or any data tool
- **RDF N-Triples** — semantic web format for SPARQL / Protégé / triple stores

---

## 🛠️ Local Installation

```bash
git clone https://github.com/YOUR_USERNAME/ontology-builder-ai-bootcamp.git
cd ontology-builder-ai-bootcamp
pip install -r requirements.txt
streamlit run ontology_agent/app.py
```

Requires a [Groq API key](https://console.groq.com) (free tier available).

---

## 📚 Built With

- [Streamlit](https://streamlit.io) — web UI
- [Groq](https://groq.com) — LLM inference (`qwen/qwen3.8-27b`)
- [NetworkX](https://networkx.org) — graph data structure + algorithms
- [Plotly](https://plotly.com) — 3-D graph visualization
- [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/) — web scraping

---

## 📖 About

Built as part of a 10-day AI Skills Bootcamp exploring multi-agent systems,
knowledge graphs, and retrieval-augmented generation.

- **Day 8** — Multi-agent graph builder with coverage analysis
- **Day 9** — RAG layer: Ask the Graph + multi-format export

---

*MIT License*

"""
RAGAgent — Day 9
Retrieval-Augmented Generation over the live knowledge graph.
Takes a question, retrieves relevant subgraph via QueryAgent,
builds a context string, calls Groq LLM, returns a cited answer.
"""

import logging
import httpx
from groq import Groq
import networkx as nx
from query_agent import QueryAgent

logger = logging.getLogger(__name__)

GROQ_MODEL  = "qwen/qwen3.8-27b"
MAX_CONTEXT = 3000   # chars of graph context sent to LLM
MAX_TOKENS  = 1200


class RAGAgent:

    def __init__(self, groq_api_key: str, graph: nx.DiGraph):
        self.groq_api_key = groq_api_key
        self.graph        = graph
        self.query_agent  = QueryAgent(graph)
        self.history: list = []   # conversation memory [(q, a), ...]

    # ── Public API ─────────────────────────────────────────────────────────────

    def ask(self, question: str, radius: int = 1) -> dict:
        """
        Answer a question using the knowledge graph as context.

        Returns:
          {
            "answer":   str,
            "sources":  [str, ...],   # source article names cited
            "nodes_used": int,
            "context_nodes": [(node_id, label), ...]
          }
        """
        if self.graph.number_of_nodes() == 0:
            return {
                "answer": "The knowledge graph is empty. Please ingest some articles first using the Research Planner or URL tab.",
                "sources": [], "nodes_used": 0, "context_nodes": [],
            }

        # 1. Retrieve relevant nodes
        hits = self.query_agent.search(question, top_k=12)
        if not hits:
            return {
                "answer": "I couldn't find relevant concepts in the graph for that question. Try rephrasing or ingest more articles.",
                "sources": [], "nodes_used": 0, "context_nodes": [],
            }

        node_ids = [h[0] for h in hits]

        # 2. Expand to subgraph
        subgraph = self.query_agent.get_subgraph(node_ids, radius=radius)
        logger.info(f"RAG subgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")

        # 3. Build context
        context, sources, context_nodes = self._build_context(subgraph, hits)

        # 4. Build conversation history snippet
        history_text = ""
        if self.history:
            recent = self.history[-2:]   # last 2 exchanges
            history_text = "\n".join(
                f"Q: {q}\nA: {a[:200]}..." for q, a in recent
            )

        # 5. Call LLM
        answer = self._call_llm(question, context, history_text)

        # 6. Store in history
        self.history.append((question, answer))
        if len(self.history) > 10:
            self.history = self.history[-10:]

        return {
            "answer":        answer,
            "sources":       sources,
            "nodes_used":    subgraph.number_of_nodes(),
            "context_nodes": context_nodes,
        }

    def clear_history(self):
        self.history = []

    def update_graph(self, graph: nx.DiGraph):
        """Call this when the builder adds new articles so RAG sees live data."""
        self.graph       = graph
        self.query_agent = QueryAgent(graph)

    # ── Context builder ────────────────────────────────────────────────────────

    def _build_context(self, subgraph: nx.DiGraph, hits: list) -> tuple:
        """
        Convert subgraph into a compact text context for the LLM.
        Returns (context_str, sources_list, context_nodes_list)
        """
        lines   = []
        sources = set()
        context_nodes = []
        char_count = 0

        # Entities section
        lines.append("=== Key Concepts ===")
        for nid, data in subgraph.nodes(data=True):
            label = data.get("label", nid)
            ntype = data.get("type", "concept")
            desc  = data.get("description", "")
            srcs  = data.get("sources", [data.get("source", "")])
            sources.update(srcs)
            context_nodes.append((nid, label))

            line = f"- {label} ({ntype})"
            if desc:
                line += f": {desc[:120]}"
            lines.append(line)
            char_count += len(line)
            if char_count > MAX_CONTEXT * 0.6:
                break

        # Relationships section
        lines.append("\n=== Relationships ===")
        edge_count = 0
        for src, tgt, data in subgraph.edges(data=True):
            if char_count >= MAX_CONTEXT:
                break
            src_label = subgraph.nodes[src].get("label", src)
            tgt_label = subgraph.nodes[tgt].get("label", tgt)
            rel       = data.get("relation", "related_to")
            line      = f"- {src_label} → [{rel}] → {tgt_label}"
            lines.append(line)
            char_count += len(line)
            edge_count += 1
            if edge_count >= 60:
                break

        context = "\n".join(lines)
        return context, sorted(sources), context_nodes[:20]

    # ── LLM call ──────────────────────────────────────────────────────────────

    def _call_llm(self, question: str, context: str, history: str) -> str:
        prompt_lines = [
            "You are an expert research assistant with access to a structured knowledge graph.",
            "Use ONLY the information provided in the graph context below to answer the question.",
            "If the graph doesn't contain enough information, say so clearly.",
            "Be concise, precise, and cite specific concepts from the graph.",
            "",
        ]

        if history:
            prompt_lines += [
                "=== Recent Conversation ===",
                history,
                "",
            ]

        prompt_lines += [
            "=== Knowledge Graph Context ===",
            context,
            "",
            f"=== Question ===",
            question,
            "",
            "Answer based strictly on the graph context above:",
        ]

        prompt = "\n".join(prompt_lines)

        client = Groq(
            api_key=self.groq_api_key,
            http_client=httpx.Client(timeout=30.0),
        )
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=MAX_TOKENS,
                    temperature=0.2,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"RAG LLM attempt {attempt+1}/3: {e}")

        return "Sorry, I couldn't generate an answer right now. Please try again."

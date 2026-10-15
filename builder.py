"""
OntologyBuilder - Multi-Agent Ontology Builder (Day 8)
Tested and verified working. Clean version for Colab deployment.
"""

import os
import json
import logging
import pickle
import hashlib
from typing import Optional
from pathlib import Path

import httpx
from groq import Groq
import networkx as nx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
GROQ_MODEL   = "qwen/qwen3.8-27b"
CHUNK_SIZE   = 1200
CHUNK_OVERLAP = 200
MAX_TOKENS   = 1000
GRAPH_DIR    = os.environ.get("GRAPH_DIR", "/tmp/ontology_agent/graph_data")
GRAPH_FILE   = os.path.join(GRAPH_DIR, "graph.pkl")


# ─── OntologyBuilder ──────────────────────────────────────────────────────────

class OntologyBuilder:
    """Ingests text/URLs/PDFs → extracts entities & relations → builds knowledge graph."""

    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        self.graph   = nx.DiGraph()
        self.sources = []          # list of {"source": str, "chunks": int, "entities": int, "relationships": int}
        Path(GRAPH_DIR).mkdir(parents=True, exist_ok=True)

    # ── Public ingestion methods ───────────────────────────────────────────────

    def ingest_text(self, text: str, source_name: str = "Text") -> dict:
        chunks = self._chunk_text(text)
        total_entities = 0
        total_rels     = 0
        for chunk in chunks:
            ents, rels = self._extract_knowledge(chunk, source_name)
            self._update_graph(ents, rels, source_name)
            total_entities += len(ents)
            total_rels     += len(rels)
        result = {
            "source":        source_name,
            "chunks":        len(chunks),
            "entities":      total_entities,
            "relationships": total_rels,
        }
        self.sources.append(result)
        self._save_graph()
        return result

    def ingest_url(self, url: str, source_name: Optional[str] = None) -> dict:
        if source_name is None:
            source_name = url.split("/")[-1] or url
        text = self._fetch_url(url)
        if not text:
            return {"source": source_name, "chunks": 0, "entities": 0, "relationships": 0, "error": "fetch failed"}
        return self.ingest_text(text, source_name)

    # ── Graph stats ────────────────────────────────────────────────────────────

    @property
    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self.graph.number_of_edges()

    def get_stats(self) -> dict:
        return {
            "nodes":   self.node_count,
            "edges":   self.edge_count,
            "sources": len(self.sources),
            "chunks":  sum(s.get("chunks", 0) for s in self.sources),
        }

    # ── Disk persistence ───────────────────────────────────────────────────────

    def _save_graph(self):
        try:
            with open(GRAPH_FILE, "wb") as f:
                pickle.dump({"graph": self.graph, "sources": self.sources}, f)
            logger.info(f"Graph saved: {self.node_count} nodes, {self.edge_count} edges → {GRAPH_FILE}")
        except Exception as e:
            logger.error(f"Save failed: {e}")

    @classmethod
    def load_graph(cls, groq_api_key: str) -> "OntologyBuilder":
        """Load a previously saved graph from disk."""
        builder = cls(groq_api_key)
        if os.path.exists(GRAPH_FILE):
            try:
                with open(GRAPH_FILE, "rb") as f:
                    data = pickle.load(f)
                builder.graph   = data["graph"]
                builder.sources = data.get("sources", [])
                logger.info(f"Graph loaded: {builder.node_count} nodes")
            except Exception as e:
                logger.error(f"Load failed: {e}")
        return builder

    # ── LLM call ──────────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
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
                    temperature=0.1,
                )
                raw = resp.choices[0].message.content.strip()
                return raw
            except Exception as e:
                logger.warning(f"LLM attempt {attempt + 1}/3 failed: {e}")
        return ""

    # ── Knowledge extraction ───────────────────────────────────────────────────

    def _extract_knowledge(self, text: str, source: str):
        prompt = (
            "You are a knowledge graph expert. Extract ALL important knowledge from the text.\n\n"
            "Rules:\n"
            "- Extract EVERY named concept, technology, method, organization, person, and application\n"
            "- Decompose complex sentences into simple (subject, relation, object) triples\n"
            "- Infer implicit relationships (e.g. X used in Y implies X enables Y)\n"
            "- Aim for 10-20 entities and 15-25 relationships per chunk\n"
            "- Use short clear relation verbs: is_a, enables, uses, requires, improves, "
            "addresses, developed_by, applied_to, part_of, extends\n\n"
            "Return ONLY a JSON object, no markdown, no explanation:\n"
            "{\n"
            '  "entities": [\n'
            '    {"name": "Entity Name", '
            '"type": "concept|technology|organization|person|method|application|challenge", '
            '"description": "short description"}\n'
            "  ],\n"
            '  "relationships": [\n'
            '    {"source": "Entity A", "relation": "verb", "target": "Entity B"}\n'
            "  ]\n"
            "}\n\n"
            f'Text:\n"""{text[:2000]}"""'
        )
        raw = self._call_llm(prompt)
        if not raw:
            logger.warning("Empty LLM response")
            return [], []

        data = self._parse_json(raw)
        entities      = data.get("entities",      []) if data else []
        relationships = data.get("relationships", []) if data else []

        # Basic validation
        entities      = [e for e in entities      if isinstance(e, dict) and e.get("name")]
        relationships = [r for r in relationships if isinstance(r, dict) and r.get("source") and r.get("target")]

        logger.info(f"Extracted: {len(entities)} entities, {len(relationships)} relationships from '{source}'")
        return entities, relationships

    # ── JSON parsing (robust) ─────────────────────────────────────────────────

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """Attempt multiple strategies to extract JSON from LLM response."""
        if not raw:
            return {}

        # Strip common markdown wrappers
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last fence lines
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        # Strategy 1: direct parse
        try:
            obj = json.loads(cleaned)
            if isinstance(obj, list) and obj:
                obj = obj[0]
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        # Strategy 2: find outermost { … }
        try:
            start = cleaned.index("{")
            end   = cleaned.rindex("}") + 1
            obj   = json.loads(cleaned[start:end])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        # Strategy 3: find outermost [ … ] and take first element
        try:
            start = cleaned.index("[")
            end   = cleaned.rindex("]") + 1
            arr   = json.loads(cleaned[start:end])
            if isinstance(arr, list) and arr and isinstance(arr[0], dict):
                return arr[0]
        except Exception:
            pass

        logger.warning(f"All JSON parse strategies failed. Raw prefix: {raw[:200]}")
        return {}

    # ── Graph update ───────────────────────────────────────────────────────────

    def _update_graph(self, entities: list, relationships: list, source: str):
        for ent in entities:
            name = ent.get("name", "").strip()
            if not name:
                continue
            node_id = name.lower().replace(" ", "_")
            if not self.graph.has_node(node_id):
                self.graph.add_node(
                    node_id,
                    label=name,
                    type=ent.get("type", "concept"),
                    description=ent.get("description", ""),
                    source=source,
                    sources=[source],   # track all sources this node appears in
                )
            else:
                # Node exists — add this source to its sources list
                existing = self.graph.nodes[node_id]
                srcs = existing.get("sources", [existing.get("source", "")])
                if source not in srcs:
                    srcs = srcs + [source]
                    self.graph.nodes[node_id]["sources"] = srcs

        for rel in relationships:
            src_name  = rel.get("source", "").strip()
            tgt_name  = rel.get("target", "").strip()
            relation  = rel.get("relation", "related_to").strip()
            if not src_name or not tgt_name:
                continue
            src_id = src_name.lower().replace(" ", "_")
            tgt_id = tgt_name.lower().replace(" ", "_")
            # Auto-create nodes if missing
            for nid, nname in [(src_id, src_name), (tgt_id, tgt_name)]:
                if not self.graph.has_node(nid):
                    self.graph.add_node(nid, label=nname, type="concept",
                                        source=source, sources=[source])
            self.graph.add_edge(src_id, tgt_id, relation=relation, source=source)

    # ── Text chunking ──────────────────────────────────────────────────────────

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
        text = text.strip()
        if not text:
            return []
        if len(text) <= chunk_size:
            return [text]
        chunks = []
        start  = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start = end - overlap
        return chunks

    # ── URL fetching ───────────────────────────────────────────────────────────

    @staticmethod
    def _fetch_url(url: str) -> str:
        import re as _re
        # Wikipedia bot policy requires descriptive User-Agent with contact info
        headers = {
            "User-Agent": (
                "OntologyBuilder/1.0 (research bot; contact@research.edu) python-httpx"
            )
        }
        try:
            # Wikipedia: use Mediawiki API for FULL article (~34k chars)
            if "wikipedia.org/wiki/" in url:
                title  = url.split("/wiki/")[-1]
                mw_url = (
                    "https://en.wikipedia.org/w/api.php"
                    "?action=query&prop=extracts&explaintext=1&format=json&titles="
                    + title
                )
                resp = httpx.get(mw_url, timeout=30, headers=headers)
                resp.raise_for_status()
                pages = resp.json().get("query", {}).get("pages", {})
                text  = next(iter(pages.values())).get("extract", "")
                if text:
                    logger.info(
                        f"Wikipedia Mediawiki API: {len(text)} chars for '{title}'"
                    )
                    return text[:12000]
                return ""
            # Generic URL: fetch HTML and strip tags
            resp = httpx.get(url, follow_redirects=True, timeout=15, headers=headers)
            resp.raise_for_status()
            text = _re.sub(r"<[^>]+>", " ", resp.text)
            text = _re.sub(r"\s+", " ", text).strip()
            return text[:12000]
        except Exception as e:
            logger.error(f"URL fetch failed for {url}: {e}")
            return ""

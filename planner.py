"""
ResearchPlanner — Day 8 Part 2
Takes a domain description, uses Groq to plan which Wikipedia articles to fetch,
then ingests them one by one into the OntologyBuilder.
"""

import json
import logging
import httpx
from groq import Groq

logger = logging.getLogger(__name__)

GROQ_MODEL = "qwen/qwen3.8-27b"


class ResearchPlanner:

    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key

    def _call_llm(self, prompt: str, max_tokens: int = 1500) -> str:
        client = Groq(
            api_key=self.groq_api_key,
            http_client=httpx.Client(timeout=30.0),
        )
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.2,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"LLM attempt {attempt+1}/3: {e}")
        return ""

    def generate_plan(self, domain: str) -> dict:
        """
        Given a domain description, return a structured research plan.
        Returns:
          {
            "summary": "one paragraph explaining the plan",
            "topics": [
              {"title": "Federated_learning", "reason": "core method", "url": "https://..."},
              ...
            ]
          }
        """
        prompt = (
            f"You are a research knowledge-base architect.\n\n"
            f"Domain: {domain}\n\n"
            f"Your job: identify the 8-12 most important Wikipedia articles that together "
            f"would build a comprehensive knowledge base for this domain.\n\n"
            f"Think about:\n"
            f"- Core concepts and methods\n"
            f"- Key technologies and tools\n"
            f"- Application areas\n"
            f"- Related fields that connect to this domain\n"
            f"- Foundational theory\n\n"
            f"Return ONLY a JSON object, no markdown:\n"
            f"{{\n"
            f'  "summary": "2-3 sentences explaining your research strategy",\n'
            f'  "topics": [\n'
            f'    {{"title": "Wikipedia_article_title", "reason": "why this is important", "category": "core|method|application|theory|related"}}\n'
            f"  ]\n"
            f"}}\n\n"
            f"Use exact Wikipedia article titles (underscores, correct capitalization).\n"
            f"Verify mentally that each article exists on Wikipedia before including it."
        )
        raw = self._call_llm(prompt)
        if not raw:
            return {"summary": "Plan generation failed.", "topics": []}

        # Parse JSON
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                lines = [l for l in cleaned.split("\n") if not l.strip().startswith("```")]
                cleaned = "\n".join(lines).strip()
            data = json.loads(cleaned)
        except Exception:
            try:
                s = cleaned.index("{"); e = cleaned.rindex("}") + 1
                data = json.loads(cleaned[s:e])
            except Exception:
                return {"summary": raw[:300], "topics": []}

        # Add Wikipedia URLs
        topics = data.get("topics", [])
        for t in topics:
            title = t.get("title", "").replace(" ", "_")
            t["title"] = title
            t["url"] = f"https://en.wikipedia.org/wiki/{title}"

        return {
            "summary": data.get("summary", ""),
            "topics": topics,
        }

    def verify_topics(self, topics: list) -> list:
        """
        Check each Wikipedia article actually exists (quick HEAD request).
        Returns topics with 'exists': True/False added.
        """
        headers = {"User-Agent": "OntologyBuilder/1.0 (research bot; contact@research.edu) python-httpx"}
        verified = []
        for t in topics:
            url = t.get("url", "")
            try:
                r = httpx.head(url, follow_redirects=True, timeout=8, headers=headers)
                t["exists"] = r.status_code == 200
            except Exception:
                t["exists"] = False
            verified.append(t)
        return verified

    def generate_gap_fill_plan(self, gap_topic: str, existing_nodes: list, n: int = 4) -> list:
        """
        Given a gap topic and existing node labels, suggest n Wikipedia articles to fill it.
        Returns list of {"title": ..., "url": ..., "reason": ...}
        """
        existing_sample = ", ".join(existing_nodes[:20])
        prompt = (
            f"The knowledge base already contains these concepts: {existing_sample}\n\n"
            f"Gap identified: '{gap_topic}' is underrepresented.\n\n"
            f"Suggest exactly {n} Wikipedia articles that would best fill this gap "
            f"without duplicating what already exists.\n\n"
            f"Return ONLY a JSON array, no markdown:\n"
            f'[{{"title": "Wikipedia_article_title", "reason": "what gap it fills"}}, ...]\n\n'
            f"Use exact Wikipedia article titles with underscores."
        )
        raw = self._call_llm(prompt, max_tokens=600)
        if not raw:
            return []
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                lines = [l for l in cleaned.split("\n") if not l.strip().startswith("```")]
                cleaned = "\n".join(lines).strip()
            s = cleaned.index("["); e = cleaned.rindex("]") + 1
            topics = json.loads(cleaned[s:e])
            for t in topics:
                title = t.get("title", "").replace(" ", "_")
                t["title"] = title
                t["url"] = f"https://en.wikipedia.org/wiki/{title}"
            return topics
        except Exception as e:
            logger.error(f"Gap fill parse failed: {e}")
            return []

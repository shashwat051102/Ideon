from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import os

from .task import CollectiveIdeaTask
from core.database.chroma_manager import ChromaManager
from core.models.embeddings import EmbeddingModel


@dataclass
class _Result:
    raw: str


class CollectiveCrew:
    """
    Minimal crew wrapper that matches the `crew().kickoff(inputs=...)` call style.
    It directly executes the CollectiveIdeaTask to synthesize a collective idea
    from a graph seed and its connected neighbors (autolinking if needed).
    """

    def __init__(self):
        self._task = CollectiveIdeaTask()

    # Keep the API compatible with examples: ResearchCrew().crew().kickoff(inputs=...)
    def crew(self) -> "CollectiveCrew":
        return self

    def kickoff(self, inputs: Optional[Dict[str, Any]] = None) -> _Result:
        payload = inputs or {}
        seed_id = payload.get("seed_id")
        top_k = int(payload.get("top_k", 5))
        autolink = bool(payload.get("autolink", True))
        prompt = payload.get("prompt") or payload.get("topic") or "Create a single collective idea from these connected ideas."
        require_llm = bool(payload.get("require_llm", False))

        # If caller provides a free-text 'topic' (or 'text') but no seed_id, pick the nearest idea as seed
        if not seed_id:
            topic = payload.get("topic") or payload.get("text")
            if isinstance(topic, str) and topic.strip():
                try:
                    embedder = EmbeddingModel()
                    chroma = ChromaManager()
                    qemb = embedder.embed_text(topic)
                    qr = chroma.query_ideas(query_embedding=qemb, n_results=1)
                    ids = qr.get("ids") or []
                    if ids and isinstance(ids[0], list):
                        ids = ids[0]
                    if ids:
                        seed_id = ids[0]
                except Exception:
                    # Fall back to agent's internal seed picker
                    seed_id = None

        data = self._task.run(
            seed_id=seed_id,
            top_k=top_k,
            autolink_if_needed=autolink,
            prompt=prompt,
            require_llm=require_llm,
        )
        if data.get("ok"):
            title = data.get("title") or "Collective Idea"
            content = data.get("content") or ""
            raw = f"{title}\n\n{content}".strip()
        else:
            raw = f"Error: {data.get('error', 'unknown error')}"

        # Optionally save an output if caller created an output folder
        out_dir = payload.get("output_dir")
        if out_dir and isinstance(out_dir, str):
            try:
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, "collective_idea.md"), "w", encoding="utf-8") as f:
                    f.write(raw)
            except Exception:
                pass

        return _Result(raw=raw)

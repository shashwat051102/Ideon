from __future__ import annotations
from typing import Any, Dict, List, Optional

from .generator import generate_one_idea


def titleize(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "Untitled"
    t = t[0].upper() + t[1:]
    return t if t.endswith(".") else t


def snippet(txt: str, n: int = 220) -> str:
    s = (txt or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[:n-1] + "…"


def generate_ideas(
    seed_prompt: str,
    style_metrics: Optional[Dict] = None,
    context_chunks: Optional[List[str]] = None,
    n_ideas: int = 5,
    tags: Optional[List[str]] = None,
) -> List[Dict]:
    # Always produce a single idea and return as a one-element list
    idea = generate_one_idea(
        seed_prompt=seed_prompt,
        style_metrics=style_metrics,
        context_chunks=(context_chunks or [])[:3],
        tags=tags or [],
    )
    return [idea]
    
    
    
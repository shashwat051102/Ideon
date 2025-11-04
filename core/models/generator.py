from __future__ import annotations
import os
import json
import logging
import re
from typing import Dict, List, Optional, Tuple

# .env support
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# Environment toggles (support new IDEON_* and legacy IDEAWEAVER_* prefixes)
USE_LLM = str(
    os.getenv("IDEON_USE_LLM", os.getenv("IDEAWEAVER_USE_LLM", "0"))
).lower() in {"1", "true", "yes"}
LLM_MODEL = os.getenv("IDEON_LLM_MODEL", os.getenv("IDEAWEAVER_LLM_MODEL", "gpt-4.1-mini"))

# Lazy LLM init
_LLM = None

def get_llm():
    global _LLM
    if _LLM is not None:
        return _LLM
    if not USE_LLM:
        return None
    try:
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("[generator] OPENAI_API_KEY not set; using local generator")
            return None
        _LLM = ChatOpenAI(api_key=api_key, model=LLM_MODEL, temperature=0.3)
        logger.info("[generator] LLM enabled: %s", LLM_MODEL)
        return _LLM
    except Exception as e:
        logger.warning("[generator] Failed to initialize LLM (%s); using local generator", e)
        _LLM = None
        return None

# Utilities

def _titleize(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "Untitled"
    t = t[0].upper() + t[1:]
    return t if t.endswith(".") else t

def _snippet(txt: str, n: int = 220) -> str:
    s = (txt or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def _unwrap_code_fence(text: str) -> str:
    if not text:
        return ""
    # Remove ```json ... ``` or ``` ... ``` fences
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    return m.group(1).strip() if m else text.strip()


def _extract_text(resp) -> str:
    """Extract plain text from a LangChain message response across types."""
    try:
        content = getattr(resp, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # LangChain can return a list of content parts; join text segments
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
                elif isinstance(part, str):
                    parts.append(part)
            if parts:
                return "".join(parts)
        return str(resp)
    except Exception:
        return str(resp)


def _safe_json_from_text(text: str, default_title: str, default_tags: Optional[List[str]] = None) -> Dict:
    """Try to parse a JSON object from model text; fall back to sensible output."""
    default_tags = list(dict.fromkeys((default_tags or []) + ["synthesis", "llm"]))
    t = (text or "").strip()
    if not t:
        return {"title": default_title, "content": "", "tags": default_tags}

    # Unwrap code fences if present
    t1 = _unwrap_code_fence(t)

    # Try to find a JSON object anywhere in the text (prefer last)
    m = re.search(r"\{[\s\S]*\}", t1)
    candidate = m.group(0) if m else t1
    try:
        data = json.loads(candidate)
        title = str(data.get("title") or default_title).strip()
        content = str(data.get("content") or "").strip()
        tags = data.get("tags") or []
        if not isinstance(tags, list):
            tags = [str(tags)]
        tags = list(dict.fromkeys((tags or []) + default_tags))
        return {"title": title, "content": content, "tags": tags}
    except Exception:
        # Fall back: treat the entire text as content
        first_line = t1.splitlines()[0].strip() if t1 else default_title
        title = _titleize(first_line[:80]) or default_title
        return {"title": title, "content": t1, "tags": default_tags}

# Local generator: user input becomes the idea directly

def generate_local_idea(
    seed_prompt: str,
    style_metrics: Optional[Dict] = None,
    context_chunks: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
) -> Dict:
    seed = (seed_prompt or "").strip()
    ctx = (context_chunks or [])[:3]
    tag_list = tags or []

    # Simple: title = first line or truncated; content = full prompt + context references
    lines = seed.split("\n", 1)
    title = _titleize(lines[0][:80]) if lines else "Untitled"
    content = seed

    if ctx:
        content += "\n\n**Related context:**\n"
        for i, c in enumerate(ctx, 1):
            content += f"- {_snippet(c, 120)}\n"

    out_tags = list(dict.fromkeys(tag_list + ["user-created"]))
    return {"title": title, "content": content, "tags": out_tags}

# LLM generator: optional enrichment of user input

def generate_llm_idea(
    seed_prompt: str,
    style_metrics: Optional[Dict] = None,
    context_chunks: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
) -> Dict:
    llm = get_llm()
    if llm is None:
        return generate_local_idea(seed_prompt, style_metrics, context_chunks, tags)

    from langchain_core.messages import SystemMessage, HumanMessage

    seed = (seed_prompt or "").strip()
    ctx = (context_chunks or [])[:3]
    tag_list = tags or []

    sys_prompt = (
        "You are an assistant that takes user input and optionally enriches it with structure. "
        "Return a single JSON object with keys: title (string), content (string), tags (array of strings). "
        "Keep the user's original idea but you may add light formatting or context."
    )
    ctx_lines = [f"- {c.strip()}" for c in ctx if c and c.strip()]
    user_prompt = (
        f"User idea:\n{seed}\n\n"
        + ("Related context:\n" + "\n".join(ctx_lines) + "\n\n" if ctx_lines else "")
        + f"Desired tags: {', '.join(tag_list) if tag_list else '(none)'}\n"
        + "Return ONLY valid JSON."
    )

    resp = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
    text = _extract_text(resp)
    data = _safe_json_from_text(text, default_title="Untitled", default_tags=["llm-enriched"]) 
    # merge desired tags
    out_tags = list(dict.fromkeys([*tag_list, *data.get("tags", [])]))
    return {"title": data.get("title"), "content": data.get("content"), "tags": out_tags}

# Router

def generate_one_idea(
    seed_prompt: str,
    style_metrics: Optional[Dict] = None,
    context_chunks: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
) -> Dict:
    if get_llm() is not None:
        try:
            return generate_llm_idea(seed_prompt, style_metrics, context_chunks, tags)
        except Exception as e:
            logger.warning("[generator] LLM generation failed (%s); using local", e)
    return generate_local_idea(seed_prompt, style_metrics, context_chunks, tags)

# Composition from linked nodes (no DB writes)

def compose_from_nodes(
    prompt: str,
    nodes: List[Dict],  # each: {node_id, title, content, tags}
    extra_tags: Optional[List[str]] = None,
) -> Dict:
    """Synthesize a coherent document from existing linked nodes. No new DB entries."""
    llm = get_llm()
    materials = []
    for n in nodes:
        title = (n.get("title") or "").strip()
        content = (n.get("content") or "").strip()
        if title or content:
            materials.append(f"### {title}\n{content}")

    if llm is None:
        # Local fallback: stitch with a preface
        pre = "Synthesis based on linked ideas:\n\n"
        body = "\n\n".join(materials)
        return {
            "title": f"Synthesis: {prompt[:60]}",
            "content": pre + body,
            "tags": list(dict.fromkeys((extra_tags or []) + ["synthesis", "local"])),
        }

    from langchain_core.messages import SystemMessage, HumanMessage

    sys = (
        "You are an expert summarizer. Synthesize a concise, coherent write-up using the provided materials. "
        "Integrate overlapping points, remove redundancy, keep structure readable (use markdown headings). "
        "Return a single JSON object with keys: title, content, tags (array)."
    )
    user = (
        f"Topic: {prompt}\n\nMaterials (markdown excerpts):\n\n" + "\n\n".join(materials) +
        "\n\nReturn ONLY valid JSON."
    )
    resp = llm.invoke([SystemMessage(content=sys), HumanMessage(content=user)])
    text = _extract_text(resp)
    data = _safe_json_from_text(text, default_title="Synthesis", default_tags=["synthesis"]) 
    out_tags = list(dict.fromkeys((extra_tags or []) + data.get("tags", [])))
    return {"title": data.get("title"), "content": data.get("content"), "tags": out_tags}

from __future__ import annotations
from flask import Blueprint, request, render_template, jsonify
import re
from core.database.sqlite_manager import SQLiteManager
from core.models.generator import compose_from_nodes, generate_one_idea
from core.database.chroma_manager import ChromaManager
from core.models.embeddings import EmbeddingModel
from core.crews.graph_agent import GraphAgent

bp = Blueprint("compose", __name__)
db = SQLiteManager()
chroma = ChromaManager()
embedder = EmbeddingModel()
gagent = GraphAgent()


def _extract_keywords(text: str, top_n: int = 5) -> list[str]:
    """Naive keyword extractor for tags.

    - Lowercase alphanumeric tokens
    - Remove common stopwords and very short tokens
    - Rank by frequency then length; return top_n unique tokens
    """
    text = (text or "").lower()
    tokens = re.findall(r"[a-z][a-z0-9'-]+", text)
    stop = {
        'the','and','for','that','with','this','from','your','you','are','but','not','was','were','have','has','had',
        'into','onto','over','under','about','into','as','on','in','of','to','a','an','it','its','is','be','by','or','if',
        'we','i','me','my','our','us','they','them','their','he','she','his','her','hers','him','at','so','do','does','did',
        'can','could','should','would','will','just','also','than','then','when','while','how','what','which','who','whom',
        'there','here','out','up','down','new'
    }
    freq: dict[str,int] = {}
    for t in tokens:
        if len(t) < 3 or t in stop or t.isdigit():
            continue
        freq[t] = freq.get(t, 0) + 1
    # sort by frequency desc, then length desc, then alphabetically
    ranked = sorted(freq.items(), key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))
    return [w for w, _ in ranked[:top_n]]

@bp.route("/compose")
def compose_home():
    # Scope list view strictly to voice profile via cookie token; no fallback
    from flask import request
    token = request.cookies.get("vp_token")
    vp = db.get_voice_profile_by_token(token) if token else None
    if not vp:
        from flask import redirect, url_for
        return redirect(url_for("auth.login"))
    rows = db.list_idea_nodes(limit=30, user_id=vp.get("user_id"), voice_profile_id=vp.get("profile_id")) if vp else []
    return render_template("compose.html", rows=rows, result=None)

@bp.route("/compose/synthesize", methods=["POST"])
def compose_synthesize():
    node_id = request.form.get("node_id") or (request.get_json(silent=True) or {}).get("node_id")
    prompt = request.form.get("prompt") or (request.get_json(silent=True) or {}).get("prompt") or "Linked idea synthesis"
    if not node_id:
        tkn = request.cookies.get("vp_token")
        vp = db.get_voice_profile_by_token(tkn) if tkn else None
        rows = db.list_idea_nodes(limit=30, user_id=vp.get("user_id"), voice_profile_id=vp.get("profile_id")) if vp else []
        return render_template("compose.html", rows=rows, result={"error": "Select a seed idea."})

    # Collect seed + neighbors
    seed = db.get_idea_node(node_id)
    if not seed:
        return render_template("compose.html", rows=db.list_idea_nodes(limit=30), result={"error": "Seed idea not found."})
    edges = db.list_edges_for_node(node_id, limit=100)
    neighbor_ids = set()
    for e in edges:
        s = e.get("src_id"); d = e.get("dst_id")
        if s and s != node_id: neighbor_ids.add(s)
        if d and d != node_id: neighbor_ids.add(d)
    nodes = [seed]
    for nid in neighbor_ids:
        n = db.get_idea_node(nid)
        if n: nodes.append(n)

    result = compose_from_nodes(prompt=prompt, nodes=nodes, extra_tags=["compose", "autolink"])
    tkn = request.cookies.get("vp_token")
    vp = db.get_voice_profile_by_token(tkn) if tkn else None
    rows = db.list_idea_nodes(limit=30, user_id=vp.get("user_id"), voice_profile_id=vp.get("profile_id")) if vp else []
    return render_template("compose.html", rows=rows, result={"ok": True, "data": result, "sources": [n.get("node_id") for n in nodes]})


@bp.route("/compose/create", methods=["POST"])
def compose_create():
    """Create a new idea node from raw user input.
    - Uses local/enriched generator to produce title/content
    - Collects top-k context from Chroma (documents) to pass as context to the generator
    - Stores the new idea in SQLite and adds its embedding to Chroma
    - Runs autolink for the new node to create similarity edges based on top_k
    """
    payload = request.form or (request.get_json(silent=True) or {})
    seed_text = (payload.get("content") or payload.get("prompt") or "").strip()
    if not seed_text:
        return render_template("compose.html", rows=db.list_idea_nodes(limit=30), result={"error": "Please provide idea text."})

    top_k = int(payload.get("top_k", 5))

    # Require voice token to create under a profile
    tkn = request.cookies.get("vp_token")
    vp = db.get_voice_profile_by_token(tkn) if tkn else None
    if not vp:
        return render_template("compose.html", rows=[], result={"error": "Please sign in with your Voice ID."})

    # Embed and query Chroma for contextual documents (top_k) scoped to this profile
    emb = embedder.embed_text(seed_text)
    ctx_texts = []
    try:
        qr = chroma.query_ideas(query_embedding=emb, n_results=top_k, where={"voice_profile_id": {"$eq": vp.get("profile_id")}})
        docs = qr.get("documents") or []
        if docs and isinstance(docs[0], list):
            docs = docs[0]
        for d in docs:
            if d:
                ctx_texts.append(d)
    except Exception:
        ctx_texts = []

    # Ask the generator to produce an idea (may be local fallback if no LLM)
    idea = generate_one_idea(seed_prompt=seed_text, context_chunks=ctx_texts, tags=["user-created"]) or {}
    title = (idea.get("title") or seed_text.splitlines()[0] or "Untitled")[:200]
    content = idea.get("content") or seed_text
    tags = idea.get("tags") or ["user-created"]
    # Enrich tags with top keywords from the idea text
    kw = _extract_keywords(f"{title} {content}", top_n=5)
    tags = list(dict.fromkeys(tags + kw))

    # Save to DB
    try:
        # Use the authenticated voice profile for ownership
        user_id = vp.get("user_id")
        voice_profile_id = vp.get("profile_id")
        node_id = db.create_idea_node(
            user_id=user_id,
            title=title,
            content=content,
            tags=tags,
            voice_profile_id=voice_profile_id,
            source_chunk_ids=None,
            source_file_ids=None,
            request_key=None,
            metadata={"generator": "compose_create", "ctx_top_k": top_k},
        )

        # Add embedding to chroma
        try:
            md = {"title": title, "tags": ",".join(tags)}
            if vp:
                md.update({"user_id": user_id, "voice_profile_id": voice_profile_id})
            chroma.add_idea(node_id=node_id, embedding=emb, content=content, metadata=md)
        except Exception:
            pass

        # Autolink new node to nearest neighbors (mapping based on context top_k)
        try:
            created_edges = gagent.autolink_for_node(node_id=node_id, top_k=top_k, exclude_self=True)
        except Exception:
            created_edges = []

        return render_template(
            "compose.html",
            rows=db.list_idea_nodes(limit=30, user_id=user_id, voice_profile_id=voice_profile_id),
            result={"ok": True, "created_node": node_id, "created_edges": len(created_edges), "data": {"title": title, "content": content}},
        )

    except Exception as e:
        return render_template(
            "compose.html",
            rows=db.list_idea_nodes(limit=30, user_id=vp.get("user_id") if vp else None, voice_profile_id=vp.get("profile_id") if vp else None),
            result={"error": str(e)},
        )

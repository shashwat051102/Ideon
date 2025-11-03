from __future__ import annotations
from flask import Blueprint, request, render_template, jsonify
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

@bp.route("/compose")
def compose_home():
    rows = db.list_idea_nodes(limit=30)
    return render_template("compose.html", rows=rows, result=None)

@bp.route("/compose/synthesize", methods=["POST"])
def compose_synthesize():
    node_id = request.form.get("node_id") or (request.get_json(silent=True) or {}).get("node_id")
    prompt = request.form.get("prompt") or (request.get_json(silent=True) or {}).get("prompt") or "Linked idea synthesis"
    if not node_id:
        return render_template("compose.html", rows=db.list_idea_nodes(limit=30), result={"error": "Select a seed idea."})

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
    return render_template("compose.html", rows=db.list_idea_nodes(limit=30), result={"ok": True, "data": result, "sources": [n.get("node_id") for n in nodes]})


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

    # Embed and query Chroma for contextual documents (top_k)
    emb = embedder.embed_text(seed_text)
    ctx_texts = []
    try:
        qr = chroma.query_ideas(query_embedding=emb, n_results=top_k)
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

    # Save to DB
    try:
        node_id = db.create_idea_node(
            user_id="anonymous",
            title=title,
            content=content,
            tags=tags,
            voice_profile_id=None,
            source_chunk_ids=None,
            source_file_ids=None,
            request_key=None,
            metadata={"generator": "compose_create", "ctx_top_k": top_k},
        )

        # Add embedding to chroma
        try:
            chroma.add_idea(node_id=node_id, embedding=emb, content=content, metadata={"title": title, "tags": ",".join(tags)})
        except Exception:
            pass

        # Autolink new node to nearest neighbors (mapping based on context top_k)
        try:
            created_edges = gagent.autolink_for_node(node_id=node_id, top_k=top_k, exclude_self=True)
        except Exception:
            created_edges = []

        return render_template("compose.html", rows=db.list_idea_nodes(limit=30), result={"ok": True, "created_node": node_id, "created_edges": len(created_edges), "data": {"title": title, "content": content}})

    except Exception as e:
        return render_template("compose.html", rows=db.list_idea_nodes(limit=30), result={"error": str(e)})

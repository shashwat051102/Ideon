from __future__ import annotations
from flask import Blueprint, render_template, jsonify, request
import os
import json
from core.database.sqlite_manager import SQLiteManager
from core.crews.graph_agent import GraphAgent
from core.database.chroma_manager import ChromaManager
from core.models.embeddings import EmbeddingModel
from core.Collective_ideas.agent import CollectiveIdeaAgent

bp = Blueprint("map", __name__)
db = SQLiteManager()
gagent = GraphAgent()
chroma = ChromaManager()
embedder = EmbeddingModel()


@bp.route("/map")
def map_view():
    # Require login for map UI
    token = request.cookies.get("vp_token")
    if not token or not db.get_voice_profile_by_token(token):
        from flask import redirect, url_for
        return redirect(url_for("auth.login"))
    return render_template("map.html")


@bp.route("/api/graph")
def api_graph():
    limit = int(request.args.get("limit", 200))
    # Scope strictly to voice profile using cookie token; no DB fallback
    token = request.cookies.get("vp_token")
    vp = db.get_voice_profile_by_token(token) if token else None
    if not vp:
        return jsonify({"elements": {"nodes": [], "edges": []}, "login_required": True})
    user = db.get_user(vp.get("user_id")) if vp else None
    nodes = db.list_idea_nodes(limit=limit, user_id=user["user_id"], voice_profile_id=vp.get("profile_id")) if user else []
    edges = db.list_all_edges(limit=1000)

    # Deduplicate edges by (src_id, dst_id, edge_type)
    seen = set()
    e = []
    # Track synthetic cluster nodes that may appear as edge endpoints
    cluster_nodes: dict[str, str] = {}
    for x in edges:
        key = (x.get("src_id"), x.get("dst_id"), x.get("edge_type"))
        if key in seen:
            continue
        seen.add(key)
        # Capture cluster labels if present in edge metadata
        if (x.get("edge_type") == "cluster_member") and isinstance(x.get("src_id"), str) and x["src_id"].startswith("cluster:"):
            md = x.get("metadata")
            try:
                mdj = json.loads(md) if isinstance(md, str) else (md or {})
            except Exception:
                mdj = {}
            label = (mdj.get("label") or x["src_id"]).strip()
            cluster_nodes[x["src_id"]] = label
        # Parse metadata to surface edge details to the client (distance, tag_overlap)
        md_raw = x.get("metadata")
        try:
            md = json.loads(md_raw) if isinstance(md_raw, str) else (md_raw or {})
        except Exception:
            md = {}
        e.append({
            "data": {
                "id": x["edge_id"],
                "source": x["src_id"],
                "target": x["dst_id"],
                "type": x.get("edge_type"),
                "distance": md.get("distance"),
                "tag_overlap": md.get("tag_overlap"),
            }
        })

    def label_for(r):
        t = (r.get("title") or "").strip()
        if t:
            return t
        c = (r.get("content") or "").strip()
        return (c[:60] + "…") if len(c) > 60 else (c or "Untitled")

    n = [{"data": {"id": r["node_id"], "label": label_for(r)}} for r in nodes]
    # Add any missing cluster nodes referenced by edges
    for cid, clabel in cluster_nodes.items():
        n.append({"data": {"id": cid, "label": clabel, "is_cluster": True}})

    # Ensure edges only reference nodes we are returning
    node_ids = {x["data"]["id"] for x in n}
    e = [edge for edge in e if edge["data"].get("source") in node_ids and edge["data"].get("target") in node_ids]

    return jsonify({"elements": {"nodes": n, "edges": e}, "counts": {"nodes": len(n), "edges": len(e)}})


@bp.route("/api/graph/autolink", methods=["POST"])
def api_graph_autolink():
    payload = request.get_json(silent=True) or {}
    top_k = int(payload.get("top_k", 5))
    max_distance = float(payload.get("max_distance", 0.85))
    require_tag_overlap = bool(payload.get("require_tag_overlap", False))  # changed default to False for context-only
    min_tag_overlap = int(payload.get("min_tag_overlap", 1))
    close_override_distance = float(payload.get("close_override_distance", 0.35))
    min_cosine = float(payload.get("min_cosine", 0.55))
    require_mutual = bool(payload.get("require_mutual", False))
    try:
        # Safety clamps: cosine distance typically in [0,2], but we restrict to [0,1] to avoid overlinking;
        # cosine similarity in [-1,1], clamp requested threshold to [0,1].
        if max_distance < 0:
            max_distance = 0.0
        if max_distance > 1.0:
            max_distance = 1.0
        if min_cosine < 0:
            min_cosine = 0.0
        if min_cosine > 1.0:
            min_cosine = 1.0
        # Require at least two ideas to produce any edges and choose a dynamic autolink batch size
        # Scope autolink to active voice profile if present, using cookie token first
        token = request.cookies.get("vp_token")
        vp = db.get_voice_profile_by_token(token) if token else None
        if not vp:
            return jsonify({"created": 0, "detail": [], "login_required": True})
        user = db.get_user(vp.get("user_id")) if vp else None
        if not user:
            return jsonify({"created": 0, "detail": [], "login_required": True})
        current_nodes = db.list_idea_nodes(limit=10000, user_id=user["user_id"], voice_profile_id=vp.get("profile_id"))
        total = len(current_nodes)
        if total < 2:
            return jsonify({"created": 0, "detail": [], "message": "Need at least 2 ideas to autolink."})
        # Link more nodes when we have a larger graph while keeping requests bounded
        batch_limit = min(max(10, total), 200)
        res = gagent.autolink_recent(
            limit=batch_limit,
            top_k=top_k,
            max_distance=max_distance,
            require_tag_overlap=require_tag_overlap,
            min_tag_overlap=min_tag_overlap,
            close_override_distance=close_override_distance,
            min_cosine=min_cosine,
            require_mutual=require_mutual,
            user_id=(user["user_id"] if user else None),
            voice_profile_id=(vp.get("profile_id") if vp else None),
        )
        created = sum(len(x.get("created_edges", [])) for x in res)
        return jsonify({"created": created, "detail": res})
    except Exception as e:
        # Be resilient: return a structured error instead of HTTP 500 so the UI can show details
        return jsonify({"created": 0, "detail": [], "error": str(e)}), 200


@bp.route("/api/graph/collective", methods=["POST"])
def api_graph_collective():
    """
    Synthesize a collective idea from a seed node and its connected neighbors.
    Does NOT create a new node - just returns the synthesis.
    Request: { "seed_id": "..." }
    Returns: { "title": "...", "content": "...", "tags": [...], "source_nodes": [...] }
    """
    try:
        payload = request.get_json(silent=True) or {}
        seed_id = payload.get("seed_id")
        prompt = payload.get("prompt") or "Synthesize these connected ideas into one coherent collective idea."
        top_k = int(payload.get("top_k", 5))
        autolink = bool(payload.get("autolink", True))
        # allow clients to force require_llm; env can set default to require LLM by default
        env_require = str(
            os.getenv("IDEON_COLLECTIVE_REQUIRE_LLM", os.getenv("IDEAWEAVER_COLLECTIVE_REQUIRE_LLM", "0"))
        ).lower() in {"1", "true", "yes"}
        require_llm = bool(payload.get("require_llm", env_require))

        agent = CollectiveIdeaAgent()
        data = agent.run(
            seed_id=seed_id,  # None is allowed (agent will pick a seed with edges)
            top_k=top_k,
            autolink_if_needed=autolink,
            prompt=prompt,
            require_llm=require_llm,
        )

        if not data or data.get("ok") is not True:
            # Pass through explicit error if provided, else a generic message
            return jsonify({"error": data.get("error") if isinstance(data, dict) else "Failed to create collective idea"}), 400

        return jsonify({
            "title": data.get("title"),
            "content": data.get("content"),
            "tags": data.get("tags"),
            "source_nodes": data.get("source_nodes", []),
        })

    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()
        return jsonify({"error": f"Failed to create collective idea: {error_msg}"}), 500


@bp.route("/api/graph/reset", methods=["POST"])
def api_graph_reset():
    # Reset only ideas and graph (edges) – do not touch chunks or voice profiles
    db.delete_all_edges()
    db.delete_all_idea_nodes()
    try:
        chroma.clear_ideas()
    except Exception:
        pass
    return jsonify({"ok": True, "message": "Ideas and graph have been reset."})


@bp.route('/api/graph/context_map', methods=['POST'])
def api_graph_context_map():
    """Return the top-k nearest idea nodes (ids, distances, documents) for a given free-text query."""
    payload = request.get_json(silent=True) or {}
    text = payload.get('text') or ''
    if not text:
        return jsonify({'error': 'Missing text'}), 400
    top_k = int(payload.get('top_k', 5))
    try:
        qemb = embedder.embed_text(text)
        # Require voice token and restrict retrieval to that profile only
        token = request.cookies.get("vp_token")
        vp = db.get_voice_profile_by_token(token) if token else None
        if not vp:
            return jsonify({"results": [], "login_required": True})
        where = {"voice_profile_id": {"$eq": vp.get("profile_id")}}
        qr = chroma.query_ideas(query_embedding=qemb, n_results=top_k, where=where)
        ids = qr.get('ids') or []
        dists = qr.get('distances') or []
        docs = qr.get('documents') or []
        # normalize nested responses (chroma returns lists inside lists)
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        if dists and isinstance(dists[0], list):
            dists = dists[0]
        if docs and isinstance(docs[0], list):
            docs = docs[0]
        out = []
        for i, nid in enumerate(ids):
            out.append({'node_id': nid, 'distance': (dists[i] if i < len(dists) else None), 'document': (docs[i] if i < len(docs) else None)})
        return jsonify({'results': out})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


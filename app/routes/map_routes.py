from __future__ import annotations
from flask import Blueprint, render_template, jsonify, request
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
    return render_template("map.html")


@bp.route("/api/graph")
def api_graph():
    limit = int(request.args.get("limit", 200))
    nodes = db.list_idea_nodes(limit=limit)
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
        # Require at least two ideas to produce any edges
        current_nodes = db.list_idea_nodes(limit=2)
        if len(current_nodes) < 2:
            return jsonify({"created": 0, "detail": [], "message": "Need at least 2 ideas to autolink."})
        res = gagent.autolink_recent(
            limit=10,
            top_k=top_k,
            max_distance=max_distance,
            require_tag_overlap=require_tag_overlap,
            min_tag_overlap=min_tag_overlap,
            close_override_distance=close_override_distance,
            min_cosine=min_cosine,
            require_mutual=require_mutual,
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
        # allow clients to force require_llm, default False so we gracefully fall back to local synthesis
        require_llm = bool(payload.get("require_llm", False))

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
        qr = chroma.query_ideas(query_embedding=qemb, n_results=top_k)
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


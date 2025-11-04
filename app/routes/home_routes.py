from __future__ import annotations
from flask import Blueprint, request, render_template, jsonify, redirect, url_for
from core.database.sqlite_manager import SQLiteManager
from core.crews.idea_agent import IdeaGeneratorAgent
from core.models import generator

bp = Blueprint("home", __name__)
db = SQLiteManager()
agent = IdeaGeneratorAgent()

# Fixed generation settings
DEFAULT_N_IDEAS = 1
DEFAULT_CTX_TOP_K = 3


@bp.route("/")
def index():
    print("[home] GET / -> rendering home.html")
    return render_template("home.html")


@bp.route("/healthz")
def healthz():
    return "ok", 200, {"Content-Type": "text/plain; charset=utf-8"}


@bp.route("/api/llm/status")
def llm_status():
    """Report LLM readiness and environment toggles for quick troubleshooting."""
    import os
    # Probe LLM (will lazy-init if configured)
    llm = generator.get_llm()
    ready = llm is not None
    use_llm = str(os.getenv("IDEON_USE_LLM", os.getenv("IDEAWEAVER_USE_LLM", "0"))).lower() in {"1", "true", "yes"}
    model = os.getenv("IDEON_LLM_MODEL", os.getenv("IDEAWEAVER_LLM_MODEL", "gpt-4.1-mini"))
    api_key = os.getenv("OPENAI_API_KEY") or ""
    base_url = os.getenv("OPENAI_BASE_URL") or None
    # Light masking for display
    key_hint = f"...{api_key[-4:]}" if api_key else None
    try:
        from langchain_openai import ChatOpenAI  # noqa: F401
        import_ok = True
    except Exception:
        import_ok = False
    return jsonify({
        "use_llm": use_llm,
        "llm_ready": ready,
        "model": model,
        "openai_key_present": bool(api_key),
        "openai_key_hint": key_hint,
        "openai_base_url": base_url,
        "langchain_openai_import": import_ok,
    })


@bp.route("/plain")
def plain():
    return "hello", 200, {"Content-Type": "text/plain; charset=utf-8"}


@bp.route("/ideas")
def ideas():
    rows = db.list_idea_nodes(limit=100)
    return render_template("history.html", rows=rows)


@bp.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True) or request.form
    username = data.get("user") or "local_user"
    prompt = data.get("prompt") or ""
    # Enforce fixed values regardless of client input
    n_ideas = DEFAULT_N_IDEAS
    ctx_top_k = DEFAULT_CTX_TOP_K
    tags = [t.strip() for t in (data.get("tags") or "").split(",") if t.strip()]
    try:
        # Log for verification
        print(f"[generate] user={username} n_ideas={n_ideas} ctx_top_k={ctx_top_k} tags={tags}")
        res = agent.generate(username=username, prompt=prompt, n_ideas=n_ideas, ctx_top_k=ctx_top_k, tags=tags)
        print(f"[generate] created count={res.get('count')} ids={res.get('created_idea_ids')}")
        if request.is_json:
            return jsonify(res)
        return redirect(url_for("home.ideas"))
    except Exception as e:
        msg = str(e)
        if request.is_json:
            return jsonify({"error": msg}), 400
        return redirect(url_for("voice.voice_view", setup="1", error=msg))

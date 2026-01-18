from __future__ import annotations
from flask import Blueprint, render_template, request, redirect, url_for, jsonify, make_response
import os
from core.database.sqlite_manager import SQLiteManager
from core.crews.style_agent import StyleLearnerAgent
from core.database.chroma_manager import ChromaManager

bp = Blueprint("voice", __name__)
db = SQLiteManager()
style_agent = StyleLearnerAgent()
chroma = ChromaManager()


@bp.route("/voice")
def voice_view():
    # Ensure we have a user
    user = db.get_user_by_username("local_user")
    if not user:
        try:
            uid = db.create_user("local_user")
            user = db.get_user(uid) or {"user_id": uid, "username": "local_user"}
        except Exception:
            user = None

    # Resolve active profile strictly by cookie token (no DB fallback)
    vp = None
    token = request.cookies.get("vp_token")
    if token:
        vp = db.get_voice_profile_by_token(token)

    # Auto-provision a default voice profile on first visit if none exists (disabled by default)
    auto_flag = str(os.getenv("IDEON_VOICE_AUTOSETUP", "0")).lower() in {"1", "true", "yes"}
    setup_qs = str(request.args.get("setup", "")).lower() in {"1", "true", "yes", "y"}
    if user and not vp and (auto_flag or setup_qs):
        try:
            style_agent.learn_voice_profile(username="local_user", profile_name="Default Voice")
            # Redirect without query params to avoid showing any prior error
            return redirect(url_for("voice.voice_view"))
        except Exception as e:
            # Fall through to template with error message if provisioning fails
            return render_template(
                "voice.html",
                user=user,
                vp=None,
                setup=request.args.get("setup"),
                error=str(e),
            )

    # Only show the current profile; don't list other profiles to avoid cross-access UX
    profiles = [vp] if vp else []

    return render_template(
        "voice.html",
        user=user,
        vp=vp,
        profiles=profiles,
        setup=request.args.get("setup"),
        error=request.args.get("error"),
    )


@bp.route("/voice/create", methods=["POST"])
def voice_create():
    data = request.get_json(silent=True) or request.form
    username = data.get("user") or "local_user"
    profile_name = data.get("profile_name") or "Default Voice"
    try:
        res = style_agent.learn_voice_profile(username=username, profile_name=profile_name)
        # Fetch token for the created profile and set login cookie
        vp = db.get_active_voice_profile(res["user_id"]) if res else None
        if vp:
            token = vp.get("auth_token") or ""
            resp = make_response(redirect(url_for("voice.voice_view")))
            if token:
                resp.set_cookie("vp_token", token, max_age=60*60*24*365, httponly=True, secure=True, samesite="Lax")
            if request.is_json:
                return jsonify({"status": "ok", **res, "token": token})
            return resp
        if request.is_json:
            return jsonify({"status": "ok", **res})
        return redirect(url_for("voice.voice_view"))
    except Exception as e:
        msg = str(e)
        if request.is_json:
            return jsonify({"error": msg}), 400
        return redirect(url_for("voice.voice_view", error=msg))


@bp.route("/voice/login", methods=["POST"]) 
def voice_login():
    data = request.get_json(silent=True) or request.form
    token = (data.get("token") or "").strip()
    vp = db.get_voice_profile_by_token(token)
    if not vp:
        return jsonify({"error": "Invalid token"}), 400 if request.is_json else redirect(url_for("voice.voice_view", error="Invalid token"))
    # Mark this profile active for its user and set cookie
    try:
        db.set_active_voice_profile(vp["profile_id"])  # keep single active per user
    except Exception:
        pass
    resp = make_response(redirect(url_for("voice.voice_view")))
    resp.set_cookie("vp_token", token, max_age=60*60*24*365, httponly=True, secure=True, samesite="Lax")
    if request.is_json:
        return jsonify({"ok": True, "profile_id": vp["profile_id"], "token": token})
    return resp


@bp.route("/voice/logout")
def voice_logout():
    resp = make_response(redirect(url_for("voice.voice_view")))
    resp.delete_cookie("vp_token")
    return resp


@bp.route("/voice/activate/<profile_id>", methods=["POST"])
def voice_activate(profile_id: str):
    """Activate a profile by id and set vp_token cookie accordingly."""
    current_token = request.cookies.get("vp_token")
    current_vp = db.get_voice_profile_by_token(current_token) if current_token else None
    target_vp = db.get_voice_profile(profile_id)
    if not target_vp:
        return jsonify({"error": "Profile not found"}), 404 if request.is_json else redirect(url_for("voice.voice_view", error="Profile not found"))
    # Only allow activation if same owner as current token
    if not current_vp or (current_vp.get("user_id") != target_vp.get("user_id")):
        return jsonify({"error": "Not allowed"}), 403 if request.is_json else redirect(url_for("auth.login", error="Please sign in with the correct Voice ID"))
    try:
        db.set_active_voice_profile(profile_id)
    except Exception:
        pass
    token = target_vp.get("auth_token") or ""
    resp = make_response(redirect(url_for("voice.voice_view")))
    if token:
        resp.set_cookie("vp_token", token, max_age=60*60*24*365, httponly=True, secure=True, samesite="Lax")
    if request.is_json:
        return jsonify({"ok": True, "profile_id": profile_id, "token": token})
    return resp


@bp.route("/voice/reset", methods=["POST"])
def voice_reset():
    """Delete ideas and edges for the active profile (and its vectors)."""
    # Resolve user and active profile like in voice_view
    user = db.get_user_by_username("local_user")
    if not user:
        return jsonify({"error": "No user"}), 400 if request.is_json else redirect(url_for("voice.voice_view", error="No user"))
    vp = None
    token = request.cookies.get("vp_token")
    if token:
        vp = db.get_voice_profile_by_token(token)
    if not vp:
        vp = db.get_active_voice_profile(user["user_id"]) if user else None
    if not vp:
        return jsonify({"error": "No active profile"}), 400 if request.is_json else redirect(url_for("voice.voice_view", error="No active profile"))

    try:
        db.delete_ideas_and_edges_for_profile(user["user_id"], vp["profile_id"])
        chroma.delete_ideas_for_profile(vp["profile_id"])
    except Exception as e:
        if request.is_json:
            return jsonify({"error": str(e)}), 500
        return redirect(url_for("voice.voice_view", error=str(e)))

    if request.is_json:
        return jsonify({"ok": True})
    return redirect(url_for("voice.voice_view"))

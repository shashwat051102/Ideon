from __future__ import annotations
from flask import Blueprint, render_template, request, redirect, url_for, jsonify
from core.database.sqlite_manager import SQLiteManager
from core.crews.style_agent import StyleLearnerAgent

bp = Blueprint("voice", __name__)
db = SQLiteManager()
style_agent = StyleLearnerAgent()


@bp.route("/voice")
def voice_view():
    user = db.get_user_by_username("local_user")
    vp = db.get_active_voice_profile(user["user_id"]) if user else None
    return render_template("voice.html", user=user, vp=vp, setup=request.args.get("setup"), error=request.args.get("error"))


@bp.route("/voice/create", methods=["POST"])
def voice_create():
    data = request.get_json(silent=True) or request.form
    username = data.get("user") or "local_user"
    profile_name = data.get("profile_name") or "Default Voice"
    try:
        res = style_agent.learn_voice_profile(username=username, profile_name=profile_name)
        if request.is_json:
            return jsonify({"status": "ok", **res})
        return redirect(url_for("voice.voice_view"))
    except Exception as e:
        msg = str(e)
        if request.is_json:
            return jsonify({"error": msg}), 400
        return redirect(url_for("voice.voice_view", error=msg))

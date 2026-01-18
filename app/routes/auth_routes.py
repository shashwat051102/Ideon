from __future__ import annotations
from flask import Blueprint, render_template, request, redirect, url_for, jsonify, make_response
from core.database.sqlite_manager import SQLiteManager

bp = Blueprint("auth", __name__)
db = SQLiteManager()


@bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        # If a cookie exists, validate it; if valid, go home; if invalid, clear and show login
        token = request.cookies.get("vp_token")
        if token:
            vp = None
            try:
                vp = db.get_voice_profile_by_token(token)
            except Exception:
                vp = None
            if vp:
                return redirect(url_for("home.index"))
            # Invalid/stale cookie → clear it and continue to login page
            resp = make_response(render_template("login.html", error=request.args.get("error")))
            resp.delete_cookie("vp_token")
            return resp
        return render_template("login.html", error=request.args.get("error"))

    # POST: attempt login with Voice ID token
    data = request.get_json(silent=True) or request.form
    token = (data.get("token") or "").strip()
    if not token:
        err = "Please provide your Voice ID token."
        if request.is_json:
            return jsonify({"error": err}), 400
        return render_template("login.html", error=err), 400

    vp = db.get_voice_profile_by_token(token)
    if not vp:
        err = "Invalid Voice ID token."
        if request.is_json:
            return jsonify({"error": err}), 401
        return render_template("login.html", error=err), 401

    # Successful login → set cookie
    resp = make_response(redirect(url_for("home.index")))
    resp.set_cookie("vp_token", token, max_age=60*60*24*365, httponly=True, secure=True, samesite="Lax")
    if request.is_json:
        return jsonify({"ok": True, "profile_id": vp.get("profile_id")})
    return resp


@bp.route("/logout")
def logout():
    resp = make_response(redirect(url_for("auth.login")))
    resp.delete_cookie("vp_token")
    return resp

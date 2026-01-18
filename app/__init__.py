from __future__ import annotations
from flask import Flask, request, redirect, jsonify
import os
from core.database.sqlite_manager import SQLiteManager


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config.update(
        TEMPLATES_AUTO_RELOAD=True,
        PROPAGATE_EXCEPTIONS=True,
    )

    # Register blueprints
    from .routes.home_routes import bp as home_bp
    from .routes.map_routes import bp as map_bp
    from .routes.voice_routes import bp as voice_bp
    from .routes.auth_routes import bp as auth_bp

    app.register_blueprint(home_bp)
    app.register_blueprint(map_bp)
    app.register_blueprint(voice_bp)
    app.register_blueprint(auth_bp)

    @app.context_processor
    def _inject_active_profile():
        """Expose the active voice profile as `active_profile` using only the vp_token cookie.

        No implicit fallback to a DB 'active' profile. If there's no valid token,
        `active_profile` will be None and pages can prompt for Voice ID.
        """
        vp = None
        db = SQLiteManager()
        try:
            token = request.cookies.get("vp_token")
            if token:
                vp = db.get_voice_profile_by_token(token)
        except Exception:
            vp = None
        finally:
            # Best-effort to release connection promptly (avoid Windows file locks)
            try:
                db.close()
            except Exception:
                pass
        return {"active_profile": vp}

    @app.before_request
    def _log_request():
        # Lightweight request log to help diagnose blank pages
        try:
            print(f"[request] {request.method} {request.path}")
        except Exception:
            pass

    @app.before_request
    def _force_https_redirect():
        """Redirect http→https when running behind a proxy/LB that sets X-Forwarded-Proto.

        Classic ELB does not support redirect rules; enforce at the app layer when the
        client reached us via http. Skip for localhost/dev and when the header is absent.
        """
        try:
            # Allow opt-out via env (defaults to on in production)
            enabled = str(os.getenv("IDEON_FORCE_HTTPS", os.getenv("IDEAWEAVER_FORCE_HTTPS", "1"))).lower() in {"1", "true", "yes"}
            if not enabled:
                return None
            # Skip LB/EB health checks and internal probes
            if request.path in ("/healthz", "/favicon.ico"):
                return None
            ua = request.headers.get("User-Agent", "")
            if "ELB-HealthChecker" in ua:
                return None

            xf_proto = request.headers.get("X-Forwarded-Proto")
            host = request.headers.get("Host") or request.host
            # Only redirect if proxy indicates http and it's not localhost
            if xf_proto and xf_proto != "https" and host and not host.startswith("127.0.0.1") and not host.startswith("localhost"):
                url = request.url.replace("http://", "https://", 1)
                # Flask may include a trailing '?' for empty query; strip for cleanliness
                if url.endswith("?"):
                    url = url[:-1]
                return redirect(url, code=301)
        except Exception:
            # Never fail the request because of redirect logic
            return None

    @app.before_request
    def _require_login():
        """Fail closed: require login for all non-exempt routes.

        Redirect to /login when there's no valid vp_token cookie. Allow essentials like
        /login, /logout, /healthz, /favicon.ico, and static assets to load without auth
        so the login page renders correctly and EB health checks pass.
        """
        try:
            # Exempt paths and assets
            if request.path in {"/login", "/logout", "/healthz", "/favicon.ico", "/voice"}:
                return None
            if request.path.startswith("/static/"):
                return None
            # Allow unauthenticated creation of a new voice (onboarding)
            if request.path == "/voice/create" and request.method == "POST":
                return None

            token = request.cookies.get("vp_token")
            wants_json = request.path.startswith("/api/") or "application/json" in (request.headers.get("Accept", ""))
            if not token:
                return (jsonify({"login_required": True}), 401) if wants_json else redirect("/login")

            # Validate token maps to an existing voice profile
            db = SQLiteManager()
            try:
                vp = db.get_voice_profile_by_token(token)
            finally:
                try:
                    db.close()
                except Exception:
                    pass

            if not vp:
                return (jsonify({"login_required": True}), 401) if wants_json else redirect("/login")
        except Exception:
            # If anything goes wrong, send the user to login rather than allowing access
            wants_json = request.path.startswith("/api/") or "application/json" in (request.headers.get("Accept", ""))
            return (jsonify({"login_required": True}), 401) if wants_json else redirect("/login")

    @app.after_request
    def _set_security_headers(resp):
        """Add basic security headers (HSTS when over HTTPS)."""
        try:
            if request.is_secure or request.headers.get("X-Forwarded-Proto") == "https":
                # 6 months HSTS with subdomains; adjust as needed
                resp.headers.setdefault("Strict-Transport-Security", "max-age=15552000; includeSubDomains")
        except Exception:
            pass
        return resp

    @app.errorhandler(Exception)
    def _handle_exception(e):
        # Development-only: return plaintext traceback to surface errors
        import traceback
        tb = traceback.format_exc()
        try:
            print(tb)
        except Exception:
            pass
        return ("Internal error\n\n" + tb, 500, {"Content-Type": "text/plain; charset=utf-8"})

    return app


# Enable running directly: `python -m app`
if __name__ == "__main__":
    app = create_app()
    # Debug on by default locally; allow override via IDEON_DEBUG/FLASK_DEBUG
    debug_flag = str(os.getenv("IDEON_DEBUG", os.getenv("FLASK_DEBUG", "1"))).lower() in {"1", "true", "yes"}
    # No reloader to avoid double-spawn on Windows
    app.run(host="127.0.0.1", port=5000, debug=debug_flag, use_reloader=False, threaded=True)

from __future__ import annotations
from flask import Flask, request


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

    app.register_blueprint(home_bp)
    app.register_blueprint(map_bp)
    app.register_blueprint(voice_bp)

    @app.before_request
    def _log_request():
        # Lightweight request log to help diagnose blank pages
        try:
            print(f"[request] {request.method} {request.path}")
        except Exception:
            pass

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
    # Debug on, but no reloader to avoid double-spawn on Windows
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False, threaded=True)

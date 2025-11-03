from __future__ import annotations
from app import create_app

app = create_app()

if __name__ == "__main__":
	# Run development server (no reloader to avoid double-spawn on Windows tasks)
	app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False, threaded=True)

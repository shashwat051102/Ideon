import sys
from pathlib import Path

# Ensure repo root on sys.path
repo = Path(__file__).resolve().parents[1]
if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))

import app  # noqa: F401

# Instantiate app to exercise __init__ hooks
application = app.create_app()
print("ok")

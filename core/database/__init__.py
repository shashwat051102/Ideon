try:
    from .chroma_manager import ChromaManager  # optional; may require chromadb
except Exception:
    ChromaManager = None  # type: ignore
from .sqlite_manager import SQLiteManager

__all__ = ["SQLiteManager"]
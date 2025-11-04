from __future__ import annotations
import os
import hashlib
from typing import List, Optional

# Speed up imports and avoid TensorFlow/JAX heavy integrations from transformers
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

# Defer sentence-transformers import unless explicitly enabled
_HAS_ST = False
SentenceTransformer = None  # type: ignore


class EmbeddingModel:
    """
    Local-first embedding model with graceful fallback.
    Preferred: sentence-transformers (normalized embeddings).
    Fallback: deterministic hashing to fixed-dim vector.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        # type: ignore[name-defined] until optional ST import occurs
        self.model: Optional[object] = None
        self.dim = 384  # default for all-MiniLM-L6-v2
        self.available = False

        # Only attempt heavy import when user explicitly opts in
        use_st = os.environ.get(
            "IDEON_USE_ST",
            os.environ.get("IDEAWEAVER_USE_ST", "0"),
        ).lower() in {"1", "true", "yes"}
        if use_st:
            global _HAS_ST, SentenceTransformer
            if SentenceTransformer is None:
                try:
                    from sentence_transformers import SentenceTransformer as _ST  # type: ignore
                    SentenceTransformer = _ST
                    _HAS_ST = True
                except Exception:
                    _HAS_ST = False
            if _HAS_ST and SentenceTransformer is not None:
                try:
                    self.model = SentenceTransformer(model_name)
                    _ = self.model.encode(["ok"], normalize_embeddings=True)
                    self.available = True
                    try:
                        vec = self.model.encode(["detect"], normalize_embeddings=True)[0]
                        self.dim = len(vec)
                    except Exception:
                        pass
                except Exception:
                    self.model = None
                    self.available = False

        if not self.available:
            # Fallback dimension: match all-MiniLM default (and existing Chroma collections)
            self.dim = 384

    def embed_text(self, text: str) -> List[float]:
        """Return a normalized embedding for a single text."""
        text = (text or "").strip()
        if self.available and self.model is not None:
            v = self.model.encode([text], normalize_embeddings=True)[0]
            return v.tolist()

        # Deterministic hashing fallback
        if not text:
            vec = [0.0] * self.dim
            vec[0] = 1.0
            return vec

        vec = [0.0] * self.dim
        for token in text.split():
            h = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            val = ((h >> 8) % 1000) / 1000.0 - 0.5  # [-0.5, 0.5]
            vec[idx] += val

        # L2 normalize
        norm = (sum(x * x for x in vec) ** 0.5) or 1.0
        return [x / norm for x in vec]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Batch embed; preserves order."""
        return [self.embed_text(t) for t in texts]


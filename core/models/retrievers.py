from __future__ import annotations
from typing import Any, Dict, List, Optional

from core.database.chroma_manager import ChromaManager
from core.models.embeddings import EmbeddingModel


class SemanticRetriever:
    def __init__(
        self,
        chroma: Optional[ChromaManager] = None,
        embedder: Optional[EmbeddingModel] = None,
    ):
        self.chroma = chroma or ChromaManager()
        self.embedder = embedder or EmbeddingModel()

    @staticmethod
    def _format_results(qr: Dict) -> List[Dict[str, Any]]:
        ids = qr.get("ids") or []
        docs = qr.get("documents") or []
        metas = qr.get("metadatas") or []
        dists = qr.get("distances") or []
        if ids and isinstance(ids[0], list):
            ids = ids[0]
            docs = docs[0] if docs else []
            metas = metas[0] if metas else []
            dists = dists[0] if dists else []
        out: List[Dict[str, Any]] = []
        for i, _id in enumerate(ids):
            out.append({
                "id": _id,
                "document": docs[i] if i < len(docs) else None,
                "metadata": metas[i] if i < len(metas) else {},
                "distance": dists[i] if i < len(dists) else None,
            })
        return out

    def search_chunks(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        emb = self.embedder.embed_text(query or "")
        qr = self.chroma.query_chunks(query_embedding=emb, n_results=top_k, where=where if where else None)
        return self._format_results(qr)

    def search_ideas(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        emb = self.embedder.embed_text(query or "")
        qr = self.chroma.query_ideas(query_embedding=emb, n_results=top_k, where=where if where else None)
        return self._format_results(qr)
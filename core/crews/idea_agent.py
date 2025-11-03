from __future__ import annotations
import json
from typing import Dict, List, Optional, Tuple
import hashlib

from core.database.sqlite_manager import SQLiteManager
from core.database.chroma_manager import ChromaManager
from core.models.embeddings import EmbeddingModel
from core.models.retrievers import SemanticRetriever
from core.models.generators import generate_ideas



class IdeaGeneratorAgent:
    
    def __init__(
        self,
        db_path: str = "storage/sqlite/metadata.db",
        chroma: Optional[ChromaManager] = None,
        embedder: Optional[EmbeddingModel] = None,
        retriever: Optional[SemanticRetriever] = None,
    ):
        self.db = SQLiteManager(db_path)
        self.chroma = chroma or ChromaManager()
        self.embedder = embedder or EmbeddingModel()
        self.retriever = retriever or SemanticRetriever(chroma=self.chroma, embedder=self.embedder)

    def ensure_user_id(self, username: str) -> str:
        conn = self.db.connect()
        cur = conn.cursor()
        cur.execute("SELECT user_id FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        self.db.close()
        if row:
            return row["user_id"]
        return self.db.create_user(username)   
    
    def active_voice_profile(self, user_id: int) -> Optional[Dict]:
        
        try:
            vp = self.db.get_active_voice_profile(user_id)
            return vp
        except Exception:
            return None
        
    def collect_context(self, prompt: str, top_k: int = 5) -> Tuple[List[str], List[str]]:
        hits = self.retriever.search_chunks(prompt, top_k=top_k, where={"category": {"$eq": "writing_sample"}})
        ctx_texts: List[str] = []
        ctx_ids: List[str] = []
        for h in hits:
            t = (h.get("document") or "").strip()
            if t:
                ctx_texts.append(t)
                cid = h.get("id")
                if cid:
                    ctx_ids.append(cid)
        return ctx_texts, ctx_ids
    
    def generate(
        self,
        username: str,
        prompt: str,
        n_ideas: int = 5,
        ctx_top_k: int = 5,
        tags: Optional[List[str]] = None,
        num_ideas: Optional[int] = None,  # alias for backward-compat
    ) -> Dict:
        if num_ideas is not None and (n_ideas == 5 or n_ideas is None):
            n_ideas = int(num_ideas)

        # Enforce fixed generation settings regardless of caller
        n_ideas = 1
        ctx_top_k = 3

        user_id = self.ensure_user_id(username)
        vp = self.active_voice_profile(user_id)

        if not vp:
            raise RuntimeError("No active voice profile found for user.")
        
        
        try:
            metrics = json.loads(vp.get("analysis_metrics") or "{}") if isinstance(vp.get("analysis_metrics"), str) else (vp.get("analysis_metrics") or {})
        except Exception:
            metrics = {}
        try:
            vp_files = json.loads(vp.get("source_file_ids") or "[]") if isinstance(vp.get("source_file_ids"), str) else (vp.get("source_file_ids") or [])
        except Exception:
            vp_files = []

        # Build a stable request key for idempotency (user + prompt + sorted tags)
        normalized_tags = ",".join(sorted((tags or [])))
        req_key_src = f"{user_id}|{prompt.strip()}|{normalized_tags}"
        request_key = hashlib.sha1(req_key_src.encode("utf-8")).hexdigest()

        # If an idea for this exact request already exists, return it instead of generating anew
        existing = self.db.get_idea_by_request_key(user_id, request_key)
        if existing:
            return {
                "user_id": user_id,
                "username": username,
                "voice_profile_id": vp.get("profile_id") if vp else None,
                "created_idea_ids": [existing["node_id"]],
                "count": 1,
                "deduped": True,
            }

        ctx_texts, ctx_chunk_ids = self.collect_context(prompt, top_k=ctx_top_k)
        ideas = generate_ideas(
            seed_prompt=prompt,
            style_metrics=metrics,
            context_chunks=ctx_texts,
            n_ideas=n_ideas,
            tags=tags or [],
        )
        # Hard cap to 1 idea to guard against any upstream overrides
        if len(ideas) > 1:
            ideas = ideas[:1]
        
        
        created_ids: List[str] = []
        
        for idx, idea in enumerate(ideas):
            title = idea["title"]
            content = idea["content"]
            i_tags = idea.get("tags") or []
            
            embedding = self.embedder.embed_text(content)
            node_id = self.db.create_idea_node(
                user_id = user_id,
                title = title,
                content = content,
                tags = i_tags,
                voice_profile_id = vp.get("profile_id"),
                source_chunk_ids = ctx_chunk_ids,
                source_file_ids = vp_files,
                request_key = request_key,
                metadata={"prompt": prompt, "generator": "local", "rank": idx + 1}
            )
            
            
            
            self.chroma.add_idea(
                node_id=node_id,
                embedding = embedding,
                content=content,
                metadata={
                    "title": title,
                    "user_id": user_id,
                    "username": username,
                    "tags": ",".join(i_tags),
                    "voice_profile_id": vp.get("profile_id"),
                    "prompt": prompt,
                    "rank": idx + 1,
                    "provenance_chunk_ids": ",".join(ctx_chunk_ids),
                },
            )
            
            
            created_ids.append(node_id)
            
        return {
            "user_id": user_id,
            "username": username,
            "voice_profile_id": vp.get("profile_id"),
            "created_idea_ids": created_ids,
            "count": len(created_ids),
            "deduped": False,
        }
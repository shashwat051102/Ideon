from __future__ import annotations
import os
from typing import Dict, List, Tuple
from core.database.sqlite_manager import SQLiteManager
from core.database.chroma_manager import ChromaManager
from core.models.embeddings import EmbeddingModel
from core.models.metrics import compute_text_metrics  # alias to compute_style_metrics


class StyleLearnerAgent:
    def __init__(
        self,
        db_path: str = "storage/sqlite/metadata.db",
        chroma: ChromaManager | None = None,
        embedder: EmbeddingModel | None = None,
    ):
        self.db = SQLiteManager(db_path=db_path)
        self.chroma = chroma or ChromaManager()
        self.embedder = embedder or EmbeddingModel()

    def _ensure_user_id(self, username: str) -> str:
        conn = self.db.connect()
        cur = conn.cursor()
        cur.execute("SELECT user_id FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        self.db.close()
        if row:
            return row["user_id"]
        return self.db.create_user(username)

    def _collect_samples(self, max_files: int = 50, max_chars_per_file: int = 4000) -> Tuple[str, List[str]]:
        texts: List[str] = []
        file_ids: List[str] = []
        # Preferred path: read any previously ingested writing samples from SQLite
        rows = self.db.get_source_files(category="writing_sample")
        if rows:
            for r in rows[:max_files]:
                fid = r.get("file_id") or r.get("filepath") or ""
                combined = ""
                try:
                    chunks = self.db.get_chunks_for_file(r.get("file_id", ""))
                except Exception:
                    chunks = []
                if chunks:
                    combined = "".join(c.get("content", "") for c in chunks)
                if not combined:
                    rel = r.get("filepath", "")
                    fpath = rel if os.path.isabs(rel) else os.path.join(os.getcwd(), rel)
                    if os.path.exists(fpath):
                        try:
                            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                                combined = f.read()
                        except Exception:
                            combined = ""
                if not combined.strip():
                    continue
                texts.append(combined[:max_chars_per_file])
                file_ids.append(fid)
            return ("\n\n".join(texts), file_ids)

        # Fallback: read directly from the repo's data/writing_samples when DB is empty (first-run environments)
        samples_dir = os.path.join(os.getcwd(), "data", "writing_samples")
        if os.path.isdir(samples_dir):
            added = 0
            for name in sorted(os.listdir(samples_dir)):
                if added >= max_files:
                    break
                if not (name.endswith(".txt") or name.endswith(".md")):
                    continue
                fpath = os.path.join(samples_dir, name)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        txt = f.read().strip()
                except Exception:
                    txt = ""
                if not txt:
                    continue
                texts.append(txt[:max_chars_per_file])
                # Use relative file path as an identifier placeholder (since no DB row exists yet)
                rel = os.path.relpath(fpath).replace("\\", "/")
                file_ids.append(rel)
                added += 1
        return ("\n\n".join(texts), file_ids)

    def _build_embeddings(self, text: str) -> List[float]:
        return self.embedder.embed_text(text or "")

    def learn_voice_profile(
        self,
        username: str = "local_user",
        profile_name: str = "Default Voice",
        max_files: int = 50,
        max_chars_per_file: int = 4000
    ) -> Dict:
        user_id = self._ensure_user_id(username)
        text, file_ids = self._collect_samples(max_files=max_files, max_chars_per_file=max_chars_per_file)
        if not text.strip():
            raise ValueError("No valid writing samples found for learning voice profile.")
        metrics = compute_text_metrics(text)
        embedding = self._build_embeddings(text)
        profile_id = self.db.create_voice_profile(
            user_id=user_id,
            profile_name=profile_name,
            source_file_ids=file_ids,
            analysis_metrics=metrics,
        )
        self.chroma.add_voice_profile(
            profile_id=profile_id,
            embedding=embedding,
            metadata={
                "profile_name": profile_name,
                "user_id": user_id,
                "username": username,
                "metrics": metrics,
            },
        )
        return {
            "profile_id": profile_id,
            "user_id": user_id,
            "username": username,
            "profile_name": profile_name,
            "metrics": metrics,
            "source_file_ids": file_ids,
        }
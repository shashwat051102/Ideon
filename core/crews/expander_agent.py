from __future__ import annotations
import json
from typing import Dict, List, Optional

from core.database.sqlite_manager import SQLiteManager
from core.database.chroma_manager import ChromaManager
from core.models.embeddings import EmbeddingModel


def snippet(x: str, n: int = 240) -> str:
    s = (x or "").replace("\r", " ").replace("\n", " ").strip()
    return s if len(s) <= n else s[:n - 1] + "…"


class ExpanderAgent:
    def __init__(self, db_path: str = "storage/sqlite/metadata.db", chroma: Optional[ChromaManager] = None, embedder: Optional[EmbeddingModel] = None):
        self.db = SQLiteManager(db_path=db_path)
        self.chroma = chroma or ChromaManager()
        self.embedder = embedder or EmbeddingModel()
        
    
    def expand(self, parent_node_id: str, types: List[str] = None) -> Dict:
        types = types or ["example", "plan", "risks"]
        parent = self.db.get_idea_node(parent_node_id)
        if not parent:
            raise RuntimeError("parent not found")
        
        user_id = parent.get("user_id")
        voice_profile_id = parent.get("voice_profile_id")
        title = parent.get("title") or "Untitled"
        content = parent.get("content") or ""
        tags = parent.get("tags")
        
        if isinstance(tags,str):
            try:
                chunks = json.loads(chunks)
            except Exception:
                chunks = []
        
        files = parent.get("source_file_ids")
        if isinstance(files, str):
            try:
                files = json.loads(files)
            except Exception:
                files = []
        
        made_ids = List[str] = []
        for t in types:
            if t == "examples":
                new_title = f"Examples: {title}"
                body = [
                    f"Context: {_snippet(content)}",
                    "",
                    "Examples:",
                    "- Example 1: A concrete scenario showing the idea in action.",
                    "- Example 2: A contrasting scenario highlighting constraints.",
                    "- Example 3: A low-resource variation for quick validation."
                ]
            elif t == "plan":
                new_title = f"3-step plan: {title}"
                body = [
                    f"Context: {_snippet(content)}",
                    "",
                    "3-step plan:",
                    "1) Define the smallest measurable outcome.",
                    "2) Assemble resources and assign owners.",
                    "3) Run a 7-day pilot and capture metrics."
                ]
            elif t == "risks":
                new_title = f"Risks: {title}"
                body = [
                    f"Context: {_snippet(content)}",
                    "",
                    "Risks and mitigations:",
                    "- Risk: Adoption stalls in first week",
                    "  Mitigation: Daily check-ins and simple wins",
                    "- Risk: Tooling overhead",
                    "  Mitigation: Offline-first, paper backups",
                    "- Fallback: Roll back to previous workflow with captured learnings"
                ]
            else:
                new_title = f"{t.capitalize()}: {title}"
                body = [f"Context: {snippet(content)}", "", t]
            
            new_content = "\n".join(body).strip()
            new_tags = list(dict.fromkeys((tags or []) + ["expansion", t]))
            embedding = self.embedder.embed_text(new_content)
            node_id = self.db.create_idea_node(
                user_id=user_id,
                title=new_title,
                content=new_content,
                tags=new_tags,
                voice_profile_id=voice_profile_id,
                source_chunk_ids=chunks or [],
                source_file_ids=files or [],
                metadata={"parent_id": parent_node_id, "expansion_type": t},
            )
            self.chroma.add_idea(
                node_id=node_id,
                embedding=embedding,
                content=new_content,
                metadata={
                    "title": new_title,
                    "user_id": user_id,
                    "tags": ",".join(new_tags),
                    "voice_profile_id": voice_profile_id,
                    "parent_id": parent_node_id,
                    "expansion_type": t,
                },
            )
            self.db.create_edge(src_id=parent_node_id, dst_id=node_id, edge_type=f"expand_{t}", weight=1.0, metadata={})
            made_ids.append(node_id)
        return {"parent_node_id": parent_node_id, "created_idea_ids": made_ids, "count": len(made_ids)}
    
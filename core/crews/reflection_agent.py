from __future__ import annotations
import json
from typing import Dict, List, Optional

from core.database.sqlite_manager import SQLiteManager


class ReflectionAgent:
    def __init__(self, db_path: str = "storage/sqlite/metadata.db"):
        self.db = SQLiteManager(db_path=db_path)
        
    
    def suggest(self, limit: int = 5) -> List[Dict]:
        ideas = self.db.list_idea_nodes(Limit=200)
        scored = []
        
        for row in ideas:
            nid = row["node_id"]
            edges = self.db.list_edges_for_node(nid, Limit = 1000)
            deg = len(edges)
            title = row.get("title") or "Untitled"
            tags = row.get("tags")
            
            if isinstance(tags, str):
                try:
                    tags = json.loads(tags)
                except Exception:
                    tags = []
            
            scored.append((deg, nid, title, tags or []))
        
        scored.sort(key = lambda x: x[0])
        picks = scored[:limit]
        out: List[Dict] = []
        
        for _, nid, title, tags in picks:
            prompts = [
                f"Generate examples for: {title}",
                f"Write a 3-step plan for: {title}",
                f"List risks and mitigations for: {title}",
            ]
            if "auto-generated" in tags:
                prompts.append(f"Refine tone and cadence for: {title}")
            out.append({"node_id": nid, "title": title, "suggested_prompts": prompts})
        return out
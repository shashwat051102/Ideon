from __future__ import annotations
import json
from typing import Dict, List, Optional, Tuple

from core.database.sqlite_manager import SQLiteManager
from core.database.chroma_manager import ChromaManager
from core.models.embeddings import EmbeddingModel
from core.utils.graph_utils import cosine_similarity, kmeans, top_tokens


class GraphAgent:
    
    def __init__(
        self,
        db_path: str = "storage/sqlite/metadata.db",
        chroma: Optional[ChromaManager] = None,
        embedder: Optional[EmbeddingModel] = None,
    ):
        self.db = SQLiteManager(db_path=db_path)
        self.chroma = chroma or ChromaManager()
        self.embedder = embedder or EmbeddingModel()
    
    def get_idea_content(self, node_id: str) -> Optional[str]:
        node = self.db.get_idea_node(node_id)
        return (node or {}).get("content")
    
    def get_idea_embedding(self, node_id: str) -> Optional[List[float]]:
        data = self.chroma.get_ideas(ids=[node_id], include=["embeddings"])
        embs = data.get("embeddings")
        if embs is None:
            embs = []
        if len(embs) > 0:
            emb = embs[0]
            if hasattr(emb, "tolist"):
                emb = emb.tolist()
            return list(emb) if not isinstance(emb, list) else emb
        content = self.get_idea_content(node_id)
        if content:
            return self.embedder.embed_text(content)
        return None
    
    def weight_from_distance(self, d: Optional[float]) -> float:
        if d is None:
            return 0.0
        return 1.0/(1.0+float(d))
    
    
    def autolink_for_node(
        self,
        node_id: str,
        top_k: int = 5,
        exclude_self: bool = True,
        max_distance: float = 0.85,
        require_tag_overlap: bool = False,  # changed default to False for context-only linking
        min_tag_overlap: int = 1,
        close_override_distance: float = 0.35,
        min_cosine: float = 0.55,
        require_mutual: bool = False,
    ) -> List[Dict]:
        """
        Create similarity edges from node_id to its nearest neighbors with stricter rules:
        - Only link if distance <= max_distance, unless distance <= close_override_distance.
        - If require_tag_overlap, require at least min_tag_overlap shared tags unless distance <= close_override_distance.
        """
        emb = self.get_idea_embedding(node_id)
        if not emb:
            return []

        # Source tags
        src = self.db.get_idea_node(node_id) or {}
        src_tags_txt = (src.get("tags") or "")
        src_tags = {t.strip().lower() for t in src_tags_txt.split(",") if t and t.strip()}

        # Restrict neighbor search to the same voice_profile if available
        where = None
        try:
            src_vp = (src.get("voice_profile_id") or "").strip()
            if src_vp:
                where = {"voice_profile_id": {"$eq": src_vp}}
        except Exception:
            where = None

        qr = self.chroma.query_ideas(query_embedding=emb, n_results=top_k + (1 if exclude_self else 0), where=where)
        ids = (qr.get("ids") or [[]])
        dists = (qr.get("distances") or [[]])
        ids = ids[0] if ids and isinstance(ids[0], list) else ids
        dists = dists[0] if dists and isinstance(dists[0], list) else dists

        # Fallback: if vector store returns too few neighbors (e.g., not indexed yet),
        # compute similarities on-the-fly from recent ideas in the same profile.
        if not ids or len(ids) <= (1 if exclude_self else 0):
            try:
                # Pull a recent candidate pool from the same profile
                candidates = self.db.list_idea_nodes(limit=200, user_id=src.get("user_id"), voice_profile_id=src.get("voice_profile_id"))
                sims: List[Tuple[str, float]] = []
                for c in candidates:
                    cid = c.get("node_id")
                    if not cid or (exclude_self and cid == node_id):
                        continue
                    cemb = self.get_idea_embedding(cid)
                    if not cemb:
                        # embed from content as last resort
                        content = (c.get("content") or "").strip()
                        cemb = self.embedder.embed_text(content) if content else None
                    if not cemb:
                        continue
                    # cosine on normalized vectors = dot
                    try:
                        cos = sum(a*b for a, b in zip(emb, cemb))
                    except Exception:
                        cos = 0.0
                    sims.append((cid, float(cos)))
                # Rank by cosine desc and take top_k
                sims.sort(key=lambda x: x[1], reverse=True)
                sims = sims[:top_k]
                # Convert to ids/dists using a cosine->distance mapping (1 - cos)
                ids = [s[0] for s in sims]
                dists = [max(0.0, 1.0 - s[1]) for s in sims]
            except Exception:
                # keep empty
                ids, dists = [], []

        results: List[Dict] = []
        for i, rid in enumerate(ids):
            if exclude_self and rid == node_id:
                continue

            dist = dists[i] if i < len(dists) else None

            # Retrieve destination embedding and compute cosine now so we can use it as an alternative allow-path
            dst_emb = self.get_idea_embedding(rid)
            cos_val: Optional[float] = None
            if dst_emb is not None:
                try:
                    cos_val = sum(a*b for a, b in zip(emb, dst_emb))
                except Exception:
                    cos_val = None
            # Tag overlap screen
            dst = self.db.get_idea_node(rid) or {}
            dst_tags_txt = (dst.get("tags") or "")
            dst_tags = {t.strip().lower() for t in dst_tags_txt.split(",") if t and t.strip()}
            overlap = len(src_tags.intersection(dst_tags)) if (src_tags or dst_tags) else 0
            # Composite screen combining distance, cosine, and tag overlap
            # Decide allowance using either distance screens or cosine strength
            allowed_distance = False
            if dist is not None:
                d = float(dist)
                allowed_distance = (
                    d <= float(max_distance) or
                    d <= float(close_override_distance) or
                    (
                        require_tag_overlap and overlap >= int(min_tag_overlap) and
                        d <= (float(max_distance) + 0.10)
                    )
                )
            allowed_cosine = (cos_val is not None and cos_val >= float(min_cosine))
            # If distance isn't strong enough, allow a strong cosine match to pass
            allowed = allowed_distance or allowed_cosine

            # Mutual nearest neighbor (optional) to reduce spurious links
            if allowed and require_mutual and dst_emb is not None:
                try:
                    qr2 = self.chroma.query_ideas(query_embedding=dst_emb, n_results=top_k + (1 if exclude_self else 0))
                    ids2 = qr2.get("ids") or [[]]
                    ids2 = ids2[0] if ids2 and isinstance(ids2[0], list) else ids2
                    if node_id not in ids2:
                        allowed = False
                except Exception:
                    pass
            if not allowed:
                continue

            # Derive weight from distance if available; otherwise from cosine strength
            weight = 1.0 / (1.0 + float(dist)) if dist is not None else (float(cos_val) if cos_val is not None else 0.0)
            # Skip duplicates (directed)
            if self.db.edge_exists(src_id=node_id, dst_id=rid, edge_type="similar"):
                continue
            edge_id = self.db.create_edge(
                src_id=node_id,
                dst_id=rid,
                edge_type="similar",
                weight=weight,
                metadata={
                    "distance": dist,
                    "tag_overlap": overlap,
                    "src_tags": ",".join(sorted(src_tags)) if src_tags else "",
                    "dst_tags": ",".join(sorted(dst_tags)) if dst_tags else "",
                },
            )
            results.append({
                "edge_id": edge_id,
                "dst_id": rid,
                "weight": weight,
                "distance": dist,
                "tag_overlap": overlap,
            })
        return results
    
    
    def autolink_recent(self, limit: int = 10, top_k: int = 5, max_distance: float = 0.6, require_tag_overlap: bool = True, min_tag_overlap: int = 1, close_override_distance: float = 0.3, min_cosine: float = 0.55, require_mutual: bool = False, *, user_id: Optional[str] = None, voice_profile_id: Optional[str] = None) -> List[Dict]:
        """Autolink the most recent ideas, optionally scoped to a user and voice profile."""
        ideas = self.db.list_idea_nodes(limit=limit, user_id=user_id, voice_profile_id=voice_profile_id)
        out: List[Dict] = []
        for row in ideas:
            nid = row["node_id"]
            created = self.autolink_for_node(
                nid,
                top_k=top_k,
                max_distance=max_distance,
                require_tag_overlap=require_tag_overlap,
                min_tag_overlap=min_tag_overlap,
                close_override_distance=close_override_distance,
                min_cosine=min_cosine,
                require_mutual=require_mutual,
            )
            out.append({"node_id": nid, "created_edges": created})
        return out
    
    
    def cluster_ideas(self, k: int = 3) -> Dict:
        data = self.chroma.get_ideas(include=["embeddings", "metadatas", "documents", "ids"])
        ids = data.get("ids") or []
        embs = data.get("embeddings") or []
        metas = data.get("metadatas") or []
        docs = data.get("documents") or []
        if not ids or not embs:
            return {"clusters": []}
        
        labels, centroids = kmeans(embs, k=k, max_iter=25, seed=42)
        clusters: Dict[int, Dict] = {i: {"ids": [], "titles": [], "tags": []} for i in range(max(labels) + 1)}
        
        for i, nid in enumerate(ids):
            label = labels[i]
            clusters[label]["ids"].append(nid)
            md = metas[i] if i < len(metas) else {}
            title = (md.get("title") or "").strip()
            if title:
                clusters[label]["titles"].append(title)
            tags = (md.get("tags") or "")
            if tags:
                clusters[label]["tags"].extend([t.strip() for t in tags.split(",") if t.strip()])
        
        results = []
        
        for cid, info in clusters.items():
            label_tokens = top_tokens(info["titles"], n=3)
            tag_counts = {}
            for t in info["tags"]:
                tag_counts[t] = tag_counts.get(t, 0) + 1
            label = " / ".join(label_tokens) if label_tokens else (info["titles"][0] if info["titles"] else f"cluster-{cid}")

            cluster_node_id = f"cluster:{k}:{cid}"
            for nid in info["ids"]:
                self.db.create_edge(src_id=cluster_node_id, dst_id=nid, edge_type="cluster_member", weight=1.0, metadata={"label": label})
            
            results.append({
                "cluster_id": cluster_node_id,
                "label": label,
                "size": len(info["ids"]),
                "sample_titles": info["titles"][:3],
                "top_tags": sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:3],
            })
            
        return {"k": k, "clusters": results}
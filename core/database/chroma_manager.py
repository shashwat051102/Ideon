from __future__ import annotations
import os
import json
from typing import Dict, List, Optional
import logging
import chromadb


class ChromaManager:
    def __init__(self, persist_directory: str = "storage/chromadb"):
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)
        self._init_collections()

    def _init_collections(self):
        """Initialize or recreate collections. Be resilient to corrupted/missing collections.

        If a collection operation fails (for example Chromadb complains a stored UUID does not exist),
        attempt to delete the named collection and recreate it. If that also fails, recreate the
        client and retry. As a last resort, attach a safe no-op collection wrapper so calls don't
        raise and the app can continue.
        """
        def safe_collection(name: str):
            # Minimal safe stub that mirrors the Chroma collection interface used here
            class _Safe:
                def add(self, *args, **kwargs):
                    self._log(name, 'add called on safe collection; no-op')
                def query(self, *args, **kwargs):
                    self._log(name, 'query called on safe collection; returning empty result')
                    return {"ids": [], "documents": [], "metadatas": [], "distances": []}
                def get(self, *args, **kwargs):
                    self._log(name, 'get called on safe collection; returning empty result')
                    return {"ids": [], "documents": [], "metadatas": [], "embeddings": [], "distances": []}
                def count(self, *args, **kwargs):
                    return 0
                def delete(self, *args, **kwargs):
                    self._log(name, 'delete called on safe collection; no-op')
                def _log(self, n, m):
                    try:
                        self_logger = logging.getLogger(__name__)
                        self_logger.warning("[chroma-safe] %s: %s", n, m)
                    except Exception:
                        pass
            return _Safe()

        names = ["chunks", "voice_profiles", "ideas"]
        created = {}
        for name in names:
            try:
                if name == "chunks":
                    created[name] = self.client.get_or_create_collection("chunks", metadata={"desc": "Doc chunk embeddings"})
                elif name == "voice_profiles":
                    created[name] = self.client.get_or_create_collection("voice_profiles", metadata={"desc": "Voice embeddings"})
                elif name == "ideas":
                    created[name] = self.client.get_or_create_collection("ideas", metadata={"desc": "Idea embeddings"})
            except Exception as e:
                # Attempt to recover: delete the potentially-broken collection and recreate
                self.logger.warning("[chroma] _init_collections failed for %s: %s; attempting delete+recreate", name, e)
                try:
                    # try delete by name (chroma will ignore unknown names but may delete existing)
                    try:
                        self.client.delete_collection(name)
                    except Exception:
                        # best-effort: ignore
                        pass
                    # recreate
                    created[name] = self.client.get_or_create_collection(name, metadata={"desc": f"{name} collection"})
                except Exception as e2:
                    self.logger.error("[chroma] failed to recreate collection %s: %s; attempting client recreate", name, e2)
                    try:
                        # Recreate the client and retry create
                        self.client = chromadb.PersistentClient(path=self.persist_directory)
                        created[name] = self.client.get_or_create_collection(name, metadata={"desc": f"{name} collection"})
                    except Exception as e3:
                        self.logger.error("[chroma] failed after client recreate for %s: %s; falling back to safe collection", name, e3)
                        created[name] = safe_collection(name)

        # Attach to self ensuring keys exist
        self.chunks = created.get("chunks") or safe_collection("chunks")
        self.voice = created.get("voice_profiles") or safe_collection("voice_profiles")
        self.ideas = created.get("ideas") or safe_collection("ideas")

    def _prepare_metadata(self, metadata: Dict) -> Dict:
        out: Dict = {}
        for k, v in (metadata or {}).items():
            # Handle dicts (metrics flattened, others JSON-encoded)
            if isinstance(v, dict):
                if k == "metrics":
                    for mk, mv in v.items():
                        key = f"metrics_{mk}"
                        out[key] = mv if isinstance(mv, (str, int, float, bool)) or mv is None else str(mv)
                else:
                    out[k] = json.dumps(v, ensure_ascii=False)
                continue

            # Convert lists/tuples/sets to a safe string form (chroma expects primitive values)
            if isinstance(v, (list, tuple, set)):
                # If it's a simple list of primitives, join as comma-separated string
                if all(isinstance(x, (str, int, float, bool)) or x is None for x in v):
                    out[k] = ",".join(["" if x is None else str(x) for x in v])
                else:
                    # Fallback: JSON-encode complex lists
                    out[k] = json.dumps(list(v), ensure_ascii=False)
                continue

            # Pass through primitive values; convert others to string
            if isinstance(v, (str, int, float, bool)) or v is None:
                out[k] = v
            else:
                out[k] = str(v)
        return out

    # chunks
    def add_chunk(self, chunk_id: str, embedding: List[float], content: str, metadata: Dict):
        self.chunks.add(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[self._prepare_metadata(metadata)],
        )

    def query_chunks(self, query_embedding: List[float], n_results: int = 5, where: Optional[Dict] = None) -> Dict:
        try:
            return self.chunks.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where if where else None,
                include=["metadatas", "documents", "distances"],
            )
        except Exception as e:
            # Try to recover from missing/deleted collection by re-initializing
            self.logger.warning("[chroma] query_chunks error: %s; attempting to re-init collections", e)
            try:
                self._init_collections()
                return self.chunks.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=where if where else None,
                    include=["metadatas", "documents", "distances"],
                )
            except Exception as e2:
                self.logger.error("[chroma] query_chunks failed after retry: %s", e2)
                return {"metadatas": [], "documents": [], "distances": []}

    def count_chunks(self) -> int:
        try:
            return self.chunks.count()
        except Exception:
            return 0

    # voice profiles
    def add_voice_profile(self, profile_id: str, embedding: List[float], metadata: Dict):
        md = self._prepare_metadata(metadata)
        self.voice.add(
            ids=[profile_id],
            embeddings=[embedding],
            metadatas=[md],
            documents=[md.get("profile_name", "")],
        )

    def query_voice(self, query_embedding: List[float], n_results: int = 3, where: Optional[Dict] = None) -> Dict:
        try:
            return self.voice.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where if where else None,
                include=["metadatas", "documents", "distances"],
            )
        except Exception as e:
            self.logger.warning("[chroma] query_voice error: %s; attempting to re-init collections", e)
            try:
                self._init_collections()
                return self.voice.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=where if where else None,
                    include=["metadatas", "documents", "distances"],
                )
            except Exception as e2:
                self.logger.error("[chroma] query_voice failed after retry: %s", e2)
                return {"metadatas": [], "documents": [], "distances": []}

    # ideas
    def add_idea(self, node_id: str, embedding: List[float], content: str, metadata: Dict):
        try:
            self.ideas.add(
                ids=[node_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[self._prepare_metadata(metadata)],
            )
        except Exception as e:
            # Recover from collection-not-found by re-initializing once
            self.logger.warning("[chroma] add_idea error: %s; attempting to re-init collections", e)
            try:
                self._init_collections()
                self.ideas.add(
                    ids=[node_id],
                    embeddings=[embedding],
                    documents=[content],
                    metadatas=[self._prepare_metadata(metadata)],
                )
            except Exception as e2:
                self.logger.error("[chroma] add_idea failed after retry: %s", e2)

    def query_ideas(self, query_embedding: List[float], n_results: int = 5, where: Optional[Dict] = None) -> Dict:
        try:
            return self.ideas.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where if where else None,
                include=["metadatas", "documents", "distances"],
            )
        except Exception as e:
            self.logger.warning("[chroma] query_ideas error: %s; attempting to re-init collections", e)
            try:
                self._init_collections()
                return self.ideas.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=where if where else None,
                    include=["metadatas", "documents", "distances"],
                )
            except Exception as e2:
                self.logger.error("[chroma] query_ideas failed after retry: %s", e2)
                return {"ids": [], "documents": [], "metadatas": [], "distances": []}

    def get_ideas(self, ids: Optional[List[str]] = None, include: Optional[List[str]] = None) -> Dict:
        allowed = {"documents", "embeddings", "metadatas", "distances", "uris", "data"}
        inc = list(dict.fromkeys([i for i in (include or ["documents", "metadatas"]) if i in allowed]))
        try:
            resp = self.ideas.get(ids=ids, include=inc) if ids else self.ideas.get(include=inc)
        except Exception as e:
            # Attempt to recover once
            self.logger.warning("[chroma] get_ideas error: %s; attempting to re-init collections", e)
            try:
                self._init_collections()
                resp = self.ideas.get(ids=ids, include=inc) if ids else self.ideas.get(include=inc)
            except Exception as e2:
                self.logger.error("[chroma] get_ideas failed after retry: %s", e2)
                resp = {"ids": [], "documents": [], "metadatas": [], "embeddings": [], "distances": []}
        # normalize embeddings to plain Python lists to avoid NumPy truthiness issues
        if "embeddings" in resp and resp["embeddings"] is not None:
            embs = resp["embeddings"]
            if hasattr(embs, "tolist"):
                resp["embeddings"] = embs.tolist()
            elif isinstance(embs, list):
                resp["embeddings"] = [e.tolist() if hasattr(e, "tolist") else list(e) if not isinstance(e, list) else e for e in embs]
        return resp

    def count_ideas(self) -> int:
        try:
            return self.ideas.count()
        except Exception:
            return 0

    def clear_ideas(self) -> None:
        """Remove all idea vectors. Recreate the collection if needed."""
        try:
            # Fast path: drop the collection entirely
            self.client.delete_collection("ideas")
        except Exception:
            try:
                # Fallback: delete all docs via a broad where filter
                self.ideas.delete(where={})
            except Exception:
                pass
        # Ensure we have a live handle again
        try:
            self.ideas = self.client.get_or_create_collection("ideas", metadata={"desc": "Idea embeddings"})
        except Exception:
            pass

    def delete_ideas_for_profile(self, voice_profile_id: str) -> None:
        """Delete idea vectors for a specific voice_profile_id."""
        if not voice_profile_id:
            return
        try:
            self.ideas.delete(where={"voice_profile_id": str(voice_profile_id)})
        except Exception as e:
            self.logger.warning("[chroma] delete_ideas_for_profile error: %s; attempting re-init", e)
            try:
                self._init_collections()
                self.ideas.delete(where={"voice_profile_id": str(voice_profile_id)})
            except Exception as e2:
                self.logger.error("[chroma] delete_ideas_for_profile failed after retry: %s", e2)
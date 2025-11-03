from __future__ import annotations
import os
import json
import sqlite3
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional


class SQLiteManager:
    def __init__(self, db_path: str = "storage/sqlite/metadata.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        # one connection per thread to avoid cross-thread usage errors
        self._local = threading.local()
        self._initialize_database()

    def _new_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=30000;")
        except Exception:
            pass
        return conn

    def connect(self) -> sqlite3.Connection:
        conn: Optional[sqlite3.Connection] = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._new_connection()
            self._local.conn = conn
        return conn

    def close(self) -> None:
        conn: Optional[sqlite3.Connection] = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            finally:
                self._local.conn = None

    def _initialize_database(self):
        # Use a short-lived connection for migrations to avoid pinning a cross-thread handle
        conn = self._new_connection()
        cur = conn.cursor()

        # users
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            preferences TEXT,
            writing_style_profile_id TEXT,
            metadata TEXT
        )""")

        # source_files
        cur.execute("""
        CREATE TABLE IF NOT EXISTS source_files (
            file_id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            file_type TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            uploaded_at TEXT NOT NULL,
            uploaded_by TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            tags TEXT,
            category TEXT NOT NULL,
            metadata TEXT,
            processing_status TEXT NOT NULL,
            processed_at TEXT
        )""")

        # doc_chunks (not required by tests but reserved)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS doc_chunks (
            chunk_id TEXT PRIMARY KEY,
            source_file_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            content_length INTEGER NOT NULL,
            start_char INTEGER NOT NULL,
            end_char INTEGER NOT NULL,
            embedding_id TEXT,
            chunk_type TEXT NOT NULL,
            created_at TEXT NOT NULL,
            metadata TEXT
        )""")

        # idea_nodes (schema will be ensured on first use)
        cur.execute("""CREATE TABLE IF NOT EXISTS idea_nodes (node_id TEXT PRIMARY KEY)""")

        # edges (schema will be ensured on first use)
        cur.execute("""CREATE TABLE IF NOT EXISTS edges (edge_id TEXT PRIMARY KEY)""")

        # voice_profiles (minimal fields to support active profile lookup)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS voice_profiles (
            profile_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            profile_name TEXT,
            analysis_metrics TEXT,
            source_file_ids TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1
        )""")

        # voice_profiles (for style learning)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS voice_profiles (
            profile_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            profile_name TEXT NOT NULL,
            is_active INTEGER NOT NULL,
            analysis_metrics TEXT,
            source_file_ids TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )""")

        # Ensure expected columns exist (lightweight migrations)
        self._ensure_table_columns(cur, "idea_nodes", {
            "user_id": "TEXT",
            "title": "TEXT",
            "content": "TEXT",
            "tags": "TEXT",
            "voice_profile_id": "TEXT",
            "source_chunk_ids": "TEXT",
            "source_file_ids": "TEXT",
            "request_key": "TEXT",
            "metadata": "TEXT",
            "created_at": "TEXT",
            "updated_at": "TEXT",
        })
        self._ensure_table_columns(cur, "edges", {
            "src_id": "TEXT",
            "dst_id": "TEXT",
            "edge_type": "TEXT",
            "weight": "REAL",
            "metadata": "TEXT",
            "created_at": "TEXT",
        })
        # Ensure a partial unique index to prevent duplicate inserts for the same request key
        try:
            cur.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS uq_idea_request_key
                ON idea_nodes(user_id, request_key)
                WHERE request_key IS NOT NULL
            """)
        except Exception:
            pass
        conn.commit()
        conn.close()

    def _ensure_table_columns(self, cur: sqlite3.Cursor, table: str, columns: Dict[str, str]) -> None:
        cur.execute(f"PRAGMA table_info({table})")
        existing = {row[1] for row in cur.fetchall()}  # column names
        for name, decl in columns.items():
            if name not in existing:
                cur.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")

    # Users
    def create_user(self, username: str, preferences: Optional[Dict[str, Any]] = None) -> str:
        user_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        conn = self.connect()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO users (user_id, username, created_at, updated_at, preferences, writing_style_profile_id, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (user_id, username, now, now, json.dumps(preferences or {}), None, json.dumps({})))
        conn.commit()
        return user_id

    def get_user_by_username(self, username: str) -> Optional[Dict]:
        conn = self.connect()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        return dict(row) if row else None

    def get_user(self, user_id: str) -> Optional[Dict]:
        conn = self.connect()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    # Source files CRUD
    def create_source_file(self,
                           filename: str,
                           filepath: str,
                           file_type: str,
                           file_size: int,
                           uploaded_by: str,
                           content_hash: str,
                           category: str,
                           tags: Optional[List[str]] = None,
                           metadata: Optional[Dict[str, Any]] = None,
                           processing_status: str = "new") -> str:
        file_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        conn = self.connect()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO source_files
            (file_id, filename, filepath, file_type, file_size, uploaded_at, uploaded_by,
             content_hash, tags, category, metadata, processing_status, processed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            file_id, filename, filepath, file_type, file_size, now, uploaded_by,
            content_hash, json.dumps(tags or []), category, json.dumps(metadata or {}),
            processing_status, None
        ))
        conn.commit()
        return file_id

    def get_source_file(self, file_id: str) -> Optional[Dict]:
        conn = self.connect()
        cur = conn.cursor()
        cur.execute("SELECT * FROM source_files WHERE file_id = ?", (file_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def update_source_file_status(self, file_id: str, status: str, processed_at: Optional[str] = None) -> None:
        conn = self.connect()
        cur = conn.cursor()
        cur.execute("""
            UPDATE source_files
            SET processing_status = ?, processed_at = ?
            WHERE file_id = ?
        """, (status, processed_at or datetime.now().isoformat(), file_id))
        conn.commit()

    def delete_source_file(self, file_id: str) -> None:
        conn = self.connect()
        cur = conn.cursor()
        cur.execute("DELETE FROM source_files WHERE file_id = ?", (file_id,))
        conn.commit()

    # ---------------------- Ideas ----------------------
    def _ensure_idea_schema(self, cur: sqlite3.Cursor) -> None:
        self._ensure_table_columns(cur, "idea_nodes", {
            "user_id": "TEXT",
            "title": "TEXT",
            "content": "TEXT",
            "tags": "TEXT",
            "voice_profile_id": "TEXT",
            "source_chunk_ids": "TEXT",
            "source_file_ids": "TEXT",
            "metadata": "TEXT",
            "created_at": "TEXT",
            "updated_at": "TEXT",
        })

    def create_idea_node(self,
                          user_id: str,
                          title: str,
                          content: str,
                          tags: List[str],
                          voice_profile_id: str | None,
                          source_chunk_ids: List[str] | None = None,
                          source_file_ids: List[str] | None = None,
                          request_key: str | None = None,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        node_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        conn = self.connect()
        cur = conn.cursor()
        self._ensure_idea_schema(cur)
        cur.execute(
            """
            INSERT INTO idea_nodes (
                node_id, user_id, title, content, tags, voice_profile_id,
                source_chunk_ids, source_file_ids, request_key, metadata, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                node_id, user_id, title, content, 
                ",".join(tags or []), voice_profile_id,
                ",".join(source_chunk_ids or []), ",".join(source_file_ids or []),
                request_key,
                json.dumps(metadata or {}), now, now,
            ),
        )
        conn.commit()
        return node_id

    def list_idea_nodes(self, limit: int = 100) -> List[Dict[str, Any]]:
        conn = self.connect()
        cur = conn.cursor()
        self._ensure_idea_schema(cur)
        cur.execute(
            "SELECT node_id, user_id, title, content, tags, voice_profile_id, created_at, updated_at FROM idea_nodes ORDER BY datetime(created_at) DESC LIMIT ?",
            (limit,),
        )
        rows = cur.fetchall()
        return [dict(r) for r in rows]

    def get_idea_by_request_key(self, user_id: str, request_key: str) -> Optional[Dict[str, Any]]:
        """Return the most recent idea with the same (user_id, request_key) if it exists."""
        conn = self.connect()
        cur = conn.cursor()
        self._ensure_idea_schema(cur)
        cur.execute(
            "SELECT * FROM idea_nodes WHERE user_id = ? AND request_key = ? ORDER BY datetime(created_at) DESC LIMIT 1",
            (user_id, request_key),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def delete_all_idea_nodes(self) -> None:
        conn = self.connect()
        cur = conn.cursor()
        cur.execute("DELETE FROM idea_nodes")
        conn.commit()

    # ---------------------- Edges ----------------------
    def _ensure_edge_schema(self, cur: sqlite3.Cursor) -> None:
        self._ensure_table_columns(cur, "edges", {
            "src_id": "TEXT",
            "dst_id": "TEXT",
            "edge_type": "TEXT",
            "weight": "REAL",
            "metadata": "TEXT",
            "created_at": "TEXT",
        })

    def create_edge(self, src_id: str, dst_id: str, edge_type: str, weight: float = 1.0, metadata: Optional[Dict[str, Any]] = None) -> str:
        edge_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        conn = self.connect()
        cur = conn.cursor()
        self._ensure_edge_schema(cur)
        cur.execute(
            """
            INSERT INTO edges (edge_id, src_id, dst_id, edge_type, weight, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (edge_id, src_id, dst_id, edge_type, weight, json.dumps(metadata or {}), now),
        )
        conn.commit()
        return edge_id

    def delete_all_edges(self) -> None:
        conn = self.connect()
        cur = conn.cursor()
        cur.execute("DELETE FROM edges")
        conn.commit()

    def list_all_edges(self, limit: int = 1000) -> List[Dict[str, Any]]:
        conn = self.connect()
        cur = conn.cursor()
        self._ensure_edge_schema(cur)
        cur.execute("SELECT * FROM edges ORDER BY datetime(created_at) DESC LIMIT ?", (limit,))
        return [dict(r) for r in cur.fetchall()]

    # ---------------------- Voice Profiles ----------------------
    def get_active_voice_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        conn = self.connect()
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM voice_profiles WHERE user_id = ? AND active = 1 ORDER BY datetime(updated_at) DESC LIMIT 1",
            (user_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_source_files(self, category: Optional[str] = None) -> List[Dict]:
        conn = self.connect()
        cur = conn.cursor()
        if category:
            cur.execute("SELECT * FROM source_files WHERE category = ? ORDER BY uploaded_at DESC", (category,))
        else:
            cur.execute("SELECT * FROM source_files ORDER BY uploaded_at DESC")
        rows = [dict(r) for r in cur.fetchall()]
        return rows

    # Chunks
    def get_chunks_for_file(self, source_file_id: str) -> List[Dict]:
        conn = self.connect()
        cur = conn.cursor()
        cur.execute("SELECT * FROM doc_chunks WHERE source_file_id = ? ORDER BY chunk_index ASC", (source_file_id,))
        rows = [dict(r) for r in cur.fetchall()]
        return rows

    # Voice profiles
    def create_voice_profile(self, user_id: str, profile_name: str, source_file_ids: List[str], analysis_metrics: Dict) -> str:
        profile_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        conn = self.connect()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO voice_profiles (profile_id, user_id, profile_name, is_active, analysis_metrics, source_file_ids, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                profile_id,
                user_id,
                profile_name,
                1,
                json.dumps(analysis_metrics or {}),
                json.dumps(source_file_ids or []),
                now,
                now,
            ),
        )
        # ensure single active profile per user
        cur.execute("UPDATE voice_profiles SET is_active = 0 WHERE user_id = ? AND profile_id != ?", (user_id, profile_id))
        conn.commit()
        return profile_id

    def get_active_voice_profile(self, user_id: str) -> Optional[Dict]:
        conn = self.connect()
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM voice_profiles WHERE user_id = ? AND is_active = 1 ORDER BY updated_at DESC LIMIT 1",
            (user_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    # Ideas
    def create_idea_node(
        self,
        user_id: str,
        title: str,
        content: str,
        tags: List[str],
        voice_profile_id: Optional[str] = None,
        source_chunk_ids: Optional[List[str]] = None,
        source_file_ids: Optional[List[str]] = None,
        request_key: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        node_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        conn = self.connect()
        cur = conn.cursor()
        # ensure columns
        self._ensure_table_columns(cur, "idea_nodes", {
            "user_id": "TEXT",
            "title": "TEXT",
            "content": "TEXT",
            "tags": "TEXT",
            "voice_profile_id": "TEXT",
            "source_chunk_ids": "TEXT",
            "source_file_ids": "TEXT",
            "request_key": "TEXT",
            "metadata": "TEXT",
            "created_at": "TEXT",
            "updated_at": "TEXT",
        })
        cur.execute(
            """
            INSERT INTO idea_nodes (node_id, user_id, title, content, tags, voice_profile_id, source_chunk_ids, source_file_ids, request_key, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                node_id,
                user_id,
                title,
                content,
                ",".join(tags or []),
                voice_profile_id,
                ",".join(source_chunk_ids or []),
                ",".join(source_file_ids or []),
                request_key,
                json.dumps(metadata or {}),
                now,
                now,
            ),
        )
        conn.commit()
        return node_id

    def get_idea_node(self, node_id: str) -> Optional[Dict]:
        conn = self.connect()
        cur = conn.cursor()
        cur.execute("SELECT * FROM idea_nodes WHERE node_id = ?", (node_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def list_idea_nodes(self, limit: int = 100) -> List[Dict]:
        conn = self.connect()
        cur = conn.cursor()
        cur.execute("SELECT * FROM idea_nodes ORDER BY created_at DESC LIMIT ?", (limit,))
        return [dict(r) for r in cur.fetchall()]

    def get_idea_by_request_key(self, user_id: str, request_key: str) -> Optional[Dict]:
        conn = self.connect()
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM idea_nodes WHERE user_id = ? AND request_key = ? ORDER BY created_at DESC LIMIT 1",
            (user_id, request_key),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def delete_all_idea_nodes(self) -> None:
        conn = self.connect()
        cur = conn.cursor()
        cur.execute("DELETE FROM idea_nodes")
        conn.commit()

    # Edges
    def create_edge(self, src_id: str, dst_id: str, edge_type: str, weight: float = 1.0, metadata: Optional[Dict[str, Any]] = None) -> str:
        edge_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        conn = self.connect()
        cur = conn.cursor()
        self._ensure_table_columns(cur, "edges", {
            "src_id": "TEXT",
            "dst_id": "TEXT",
            "edge_type": "TEXT",
            "weight": "REAL",
            "metadata": "TEXT",
            "created_at": "TEXT",
        })
        cur.execute(
            """
            INSERT INTO edges (edge_id, src_id, dst_id, edge_type, weight, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (edge_id, src_id, dst_id, edge_type, float(weight), json.dumps(metadata or {}), now),
        )
        conn.commit()
        return edge_id

    def list_all_edges(self, limit: int = 1000) -> List[Dict]:
        conn = self.connect()
        cur = conn.cursor()
        cur.execute("SELECT * FROM edges ORDER BY created_at DESC LIMIT ?", (limit,))
        return [dict(r) for r in cur.fetchall()]

    def list_edges_for_node(self, node_id: str, limit: int = 1000) -> List[Dict]:
        conn = self.connect()
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM edges WHERE src_id = ? OR dst_id = ? ORDER BY created_at DESC LIMIT ?",
            (node_id, node_id, limit),
        )
        return [dict(r) for r in cur.fetchall()]

    def delete_all_edges(self) -> None:
        conn = self.connect()
        cur = conn.cursor()
        cur.execute("DELETE FROM edges")
        conn.commit()



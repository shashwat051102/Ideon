import argparse
import json
import sqlite3
from pathlib import Path

from core.database.chroma_manager import ChromaManager
from core.models.embeddings import EmbeddingModel

def reindex(db_path: Path):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT dc.chunk_id, dc.content, dc.chunk_index, dc.metadata,
               sf.file_id, sf.filepath, sf.category
        FROM doc_chunks dc
        JOIN source_files sf ON sf.file_id = dc.source_file_id
        ORDER BY sf.filepath, dc.chunk_index
    """)
    rows = cur.fetchall()
    conn.close()

    chroma = ChromaManager()
    embedder = EmbeddingModel()

    added = 0
    for r in rows:
        chunk_id = r["chunk_id"]
        content = r["content"] or ""
        try:
            md = json.loads(r["metadata"] or "{}")
        except Exception:
            md = {}
        md.update({
            "source_file_id": r["file_id"],
            "chunk_index": r["chunk_index"],
            "filepath": r["filepath"],
            "category": r["category"],
        })
        emb = embedder.embed_text(content)
        chroma.add_chunk(chunk_id=chunk_id, embedding=emb, content=content, metadata=md)
        added += 1
    print(f"Reindexed {added} chunks into Chroma.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="storage/sqlite/metadata.db", help="Path to SQLite DB")
    args = ap.parse_args()
    reindex(Path(args.db))
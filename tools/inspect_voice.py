import argparse
import json
import sqlite3
from pathlib import Path
from pprint import pprint
import chromadb

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DB = BASE_DIR / "storage" / "sqlite" / "metadata.db"
DEFAULT_CHROMA = BASE_DIR / "storage" / "chromadb"

def list_sqlite_voice_profiles(db_path: Path):
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found at: {db_path}")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT vp.profile_id, vp.profile_name, vp.created_at, vp.updated_at,
               vp.user_id, u.username, vp.is_active, vp.source_file_ids, vp.analysis_metrics
        FROM voice_profiles vp
        LEFT JOIN users u ON u.user_id = vp.user_id
        ORDER BY vp.created_at DESC
    """)
    rows = cur.fetchall()
    conn.close()
    out = []
    for r in rows:
        item = dict(r)
        try:
            item["source_file_ids"] = json.loads(item.get("source_file_ids") or "[]")
        except Exception:
            pass
        try:
            item["analysis_metrics"] = json.loads(item.get("analysis_metrics") or "{}")
        except Exception:
            pass
        out.append(item)
    return out

def list_chroma_voice_profiles(chroma_dir: Path, ids=None):
    client = chromadb.PersistentClient(path=str(chroma_dir))
    coll = client.get_or_create_collection("voice_profiles")
    if ids:
        return coll.get(ids=ids, include=["metadatas", "documents"])
    return coll.get(include=["metadatas", "documents"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DEFAULT_DB), help="Path to SQLite DB")
    ap.add_argument("--chroma", default=str(DEFAULT_CHROMA), help="Path to Chroma persist dir")
    args = ap.parse_args()

    db_path = Path(args.db).resolve()
    chroma_dir = Path(args.chroma).resolve()

    print(f"SQLite voice_profiles (DB: {db_path})")
    profiles = list_sqlite_voice_profiles(db_path)
    for p in profiles:
        print(f"- {p['profile_id']} | name={p['profile_name']} | user={p.get('username')} | active={p['is_active']}")
    if not profiles:
        print("(none)")
        return

    print(f"\nChroma voice_profiles (dir: {chroma_dir})")
    ids = [p["profile_id"] for p in profiles]
    data = list_chroma_voice_profiles(chroma_dir, ids=ids)
    docs = data.get("documents") or []
    metas = data.get("metadatas") or []
    for i, pid in enumerate(data.get("ids", [])):
        md = metas[i] if i < len(metas) else {}
        doc = docs[i] if i < len(docs) else ""
        print(f"- {pid} | doc={repr(doc)} | profile_name={md.get('profile_name')}")

    print("\nFirst profile full record (SQLite):")
    pprint(profiles[0])

if __name__ == "__main__":
    main()
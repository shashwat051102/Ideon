import os
import argparse
from typing import List, Tuple

from core.database.sqlite_manager import SQLiteManager
from core.database.chroma_manager import ChromaManager
from core.models.embeddings import EmbeddingModel
from core.utils.text_chunking import chunk_by_sentences, chunk_by_chars
from core.utils.file_utils import iter_files_by_extensions, read_text_file, sha256_text


def _chunk_text(text: str) -> List[Tuple[int, int, str]]:
    # Prefer sentence chunks; fallback to char windows for edge cases
    chunks = chunk_by_sentences(text, max_tokens=180, overlap_sentences=1)
    if not chunks:
        chunks = chunk_by_chars(text, max_chars=1200, overlap=120)
    return chunks


def _already_ingested(db: SQLiteManager, content_hash: str, relpath: str) -> bool:
    try:
        rows = db.get_source_files()
        for r in rows:
            if r.get("content_hash") == content_hash or r.get("filepath") == relpath:
                return True
    except Exception:
        pass
    return False


def ingest_directory(
    root_dir: str = "data",
    subdirs: List[str] = None,
    db_path: str = "storage/sqlite/metadata.db",
    default_username: str = "local_user"
):
    subdirs = subdirs or ["writing_samples", "notes"]

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db = SQLiteManager(db_path=db_path)
    chroma = ChromaManager()
    embedder = EmbeddingModel()

    # Ensure a user exists (simple approach: create if not found)
    user_id = db.create_user(default_username)

    for sub in subdirs:
        folder = os.path.join(root_dir, sub)
        if not os.path.isdir(folder):
            continue

        for fpath in iter_files_by_extensions(folder, extensions={".txt", ".md"}):
            relpath = os.path.relpath(fpath).replace("\\", "/")
            try:
                text = read_text_file(fpath)
            except Exception as e:
                print(f"Skip unreadable file {relpath}: {e}")
                continue

            content_hash = sha256_text(text)
            if _already_ingested(db, content_hash, relpath):
                print(f"Skip duplicate {relpath}")
                continue

            try:
                file_size = os.path.getsize(fpath)
            except Exception:
                file_size = len(text.encode("utf-8", errors="ignore"))

            category = "writing_sample" if sub == "writing_samples" else "note"

            # FIX: pass uploaded_by=user_id
            file_id = db.create_source_file(
                filename=os.path.basename(fpath),
                filepath=relpath,
                file_type="text/plain",
                file_size=file_size,
                uploaded_by=user_id,
                content_hash=content_hash,
                category=category,
                tags=[sub],
            )

            chunks = _chunk_text(text)
            for idx, (start, end, chunk) in enumerate(chunks):
                embedding = embedder.embed_text(chunk)

                # Create chunk record (SQLiteManager should compute content_length)
                chunk_id = db.create_doc_chunk(
                    source_file_id=file_id,
                    content=chunk,
                    chunk_index=idx,
                    start_char=start,
                    end_char=end,
                    chunk_type="paragraph",
                    embedding_id=None,
                    metadata={"filepath": relpath, "category": category},
                )

                # Store embedding in Chroma keyed by chunk_id
                chroma.add_chunk(
                    chunk_id=chunk_id,
                    embedding=embedding,
                    content=chunk,
                    metadata={
                        "source_file_id": file_id,
                        "chunk_index": idx,
                        "filepath": relpath,
                        "category": category,
                    },
                )

            db.update_file_status(file_id, "completed")
            print(f"Ingested {relpath}: {len(chunks)} chunks")


def main():
    parser = argparse.ArgumentParser(description="IdeaWeaver Ingest Pipeline")
    parser.add_argument("--root", default="data", help="Root data directory")
    parser.add_argument("--db", default="storage/sqlite/metadata.db", help="SQLite DB path")
    parser.add_argument(
        "--subdirs",
        default="writing_samples,notes",
        help="Comma-separated list of subfolders under root to ingest",
    )
    args = parser.parse_args()

    subdirs = [s.strip() for s in args.subdirs.split(",") if s.strip()]
    ingest_directory(root_dir=args.root, subdirs=subdirs, db_path=args.db)


if __name__ == "__main__":
    main()
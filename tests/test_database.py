import os
import uuid
from core.database import SQLiteManager

DB_PATH = "storage/sqlite/test_metadata.db"

def setup_module(module):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def teardown_module(module):
    try:
        os.remove(DB_PATH)
    except FileNotFoundError:
        pass

def test_sqlite_init_and_user_crud():
    db = SQLiteManager(db_path=DB_PATH)
    user_id = db.create_user("tester", {"theme": "dark"})
    fetched = db.get_user(user_id)
    assert fetched is not None
    assert fetched["user_id"] == user_id
    assert fetched["username"] == "tester"

def test_source_file_insert_and_query():
    db = SQLiteManager(db_path=DB_PATH)
    user_id = db.create_user("u")
    file_id = db.create_source_file(
        filename="test.txt",
        filepath="data/notes/test.txt",
        file_type="text/plain",
        file_size=12,
        uploaded_by=user_id,
        content_hash=str(uuid.uuid4()),
        category="note",
        tags=["tag1", "tag2"]
    )
    files = db.get_source_files()
    assert any(f["file_id"] == file_id for f in files)
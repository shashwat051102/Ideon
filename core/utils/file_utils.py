import os
import hashlib
from typing import Iterable, Optional


TEXT_EXTENSIONS = {".txt", ".md"}


def iter_files_by_extensions(root: str, extensions: Optional[Iterable[str]] = None):
    exts = set(e.lower() for e in (extensions or TEXT_EXTENSIONS))
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if os.path.splitext(name)[1].lower() in exts:
                yield os.path.join(dirpath, name)


def read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


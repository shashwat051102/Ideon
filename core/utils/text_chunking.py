import re
from typing import List, Tuple


def count_tokens_approx(text: str) -> int:
    """Approximate token count by whitespace split."""
    return len((text or "").split())


def split_sentences(text: str) -> List[str]:
    """Naive sentence splitter using punctuation."""
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_by_chars(text: str, max_chars: int = 1200, overlap: int = 120) -> List[Tuple[int, int, str]]:
    """
    Sliding window by characters with overlap.
    Returns list of (start, end, chunk_text).
    """
    text = text or ""
    n = len(text)
    if n == 0:
        return []

    chunks: List[Tuple[int, int, str]] = []
    start = 0
    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end]
        chunks.append((start, end, chunk))
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks


def chunk_by_sentences(
    text: str,
    max_tokens: int = 180,
    overlap_sentences: int = 1
) -> List[Tuple[int, int, str]]:
    """
    Group sentences until approx token budget is met, with sentence overlap.
    Returns list of (start_char, end_char, chunk_text).
    """
    text = text or ""
    sents = split_sentences(text)
    if not sents:
        return []

    # Map sentence spans in original text
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for s in sents:
        idx = text.find(s, cursor)
        if idx == -1:
            idx = text.find(s)
        start = max(idx, 0)
        end = start + len(s)
        spans.append((start, end))
        cursor = end

    chunks: List[Tuple[int, int, str]] = []
    i = 0
    n = len(sents)
    while i < n:
        token_count = 0
        start_idx = i
        j = i
        while j < n:
            tc = count_tokens_approx(sents[j])
            if token_count + tc > max_tokens and j > i:
                break
            token_count += tc
            j += 1

        start_char = spans[start_idx][0]
        end_char = spans[j - 1][1]
        chunk_text = text[start_char:end_char]
        chunks.append((start_char, end_char, chunk_text))

        i = max(j - overlap_sentences, start_idx + 1)

    return chunks
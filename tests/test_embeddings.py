from core.models.embeddings import EmbeddingModel
from core.utils.text_chunking import (
    count_tokens_approx,
    split_sentences,
    chunk_by_chars,
    chunk_by_sentences,
)


def test_embedding_determinism_and_shape():
    model = EmbeddingModel()
    v1 = model.embed_text("Hello world, this is a test.")
    v2 = model.embed_text("Hello world, this is a test.")
    assert isinstance(v1, list)
    assert isinstance(v2, list)
    assert len(v1) == len(v2) == model.dim
    assert v1 == v2
    norm = sum(x * x for x in v1) ** 0.5
    assert abs(norm - 1.0) < 1e-6


def test_batch_embeddings():
    model = EmbeddingModel()
    vecs = model.embed_texts(["a", "b", "c"])
    assert len(vecs) == 3
    assert all(len(v) == model.dim for v in vecs)


def test_text_chunking_by_chars():
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 10
    chunks = chunk_by_chars(text, max_chars=50, overlap=10)
    assert len(chunks) > 1
    for k in range(1, len(chunks)):
        prev = chunks[k - 1]
        curr = chunks[k]
        assert curr[0] <= prev[1]


def test_text_chunking_by_sentences():
    text = "One short sentence. Another slightly longer sentence here! And a third?"
    sents = split_sentences(text)
    assert len(sents) >= 3

    chunks = chunk_by_sentences(text, max_tokens=6, overlap_sentences=1)
    assert len(chunks) >= 2
    for (start, end, chunk) in chunks:
        assert 0 <= start < end <= len(text)
        assert chunk.strip() != ""


def test_token_count_approx():
    assert count_tokens_approx("a b  c\nd") == 4
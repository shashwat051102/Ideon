import re
from typing import Dict, List

_POS = {"good","great","excellent","positive","optimistic","hopeful","clear","insightful","creative","smart","simple","effective","powerful","useful","friendly","fast","reliable"}
_NEG = {"bad","poor","negative","pessimistic","confusing","hard","slow","buggy","broken","complex","difficult","risky"}

def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

def _tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z']+", (text or "").lower())

def _avg_sentence_len_words(text: str) -> float:
    sents = _split_sentences(text)
    if not sents:
        return 0.0
    lengths = [max(1, len(_tokens(s))) for s in sents]
    return sum(lengths) / len(lengths)

def _avg_word_len(text: str) -> float:
    words = _tokens(text)
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)

def _vocab_div(text: str) -> float:
    words = _tokens(text)
    if not words:
        return 0.0
    return len(set(words)) / len(words)

def _punct_rate(text: str) -> float:
    if not text:
        return 0.0
    punct = re.findall(r"[,.!?;:]", text)
    return len(punct) / max(1, len(text))

def _sentiment_proxy(text: str) -> float:
    words = _tokens(text)
    if not words:
        return 0.0
    pos = sum(1 for w in words if w in _POS)
    neg = sum(1 for w in words if w in _NEG)
    return (pos - neg) / len(words)

def compute_style_metrics(text: str) -> Dict:
    return {
        "avg_sentence_len_words": round(_avg_sentence_len_words(text), 3),
        "avg_word_len_chars": round(_avg_word_len(text), 3),
        "vocab_diversity": round(_vocab_div(text), 3),
        "punctuation_rate": round(_punct_rate(text), 3),
        "sentiment_proxy": round(_sentiment_proxy(text), 3),
        "sample_chars": len(text or ""),
        "sample_sentences": len(_split_sentences(text)),
        "sample_words": len(_tokens(text)),
    }

# Back-compat alias for existing imports
compute_text_metrics = compute_style_metrics
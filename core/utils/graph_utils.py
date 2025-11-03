from __future__ import annotations
import math
import random
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple, Sequence


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0

    dot  = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def kmeans(vectors: List[List[float]], k: int, max_iter: int = 30, seed: int = 42) -> Tuple[List[int], List[List[float]]]:
    if not vectors:
        return [], []
    
    n = len(vectors)
    k = max(1,min(k,n))
    rnd = random.Random(seed)
    centroids = [vectors[i] for i in rnd.sample(range(n),k)]
    labels = [0]*n
    
    
    def assign() -> int:
        changes = 0
        for i,v in enumerate(vectors):
            best_idx, best_d = 0, float("inf")
            for j, c in enumerate(centroids):
                sim = cosine_similarity(v,c)
                d = 1.0 - sim
                if d<best_d:
                    best_d, best_idx = d,j
            if labels[i] != best_idx:
                labels[i] = best_idx
                changes += 1
        return changes
    
    def recompute():
        dim = len(vectors[0])
        sums = [[0.0]*dim for _ in range(k)]
        counts = [0]*k
        for v,i in zip(vectors, labels):
            counts[i] += 1
            for d in range(dim):
                sums[i][d] += v[d]
        for j in range(k):
            if counts[j] == 0:
                sums[j] = vectors[rnd.randint(0,n)]
                counts[j] = 1
            centroids[j] = [x / counts[j] for x in sums[j]]
    for _ in range(max_iter):
        changes = assign()
        if changes == 0:
            break
        recompute()
    return labels, centroids

def top_tokens(texts: List[str], n: int = 5) -> List[str]:
    import re
    toks = []
    for t in texts:
        toks.extend(re.findall(r"[A-Za-z][A-Za-z\-']+", (t or "").lower()))
    stop = {
        "the","and","for","with","from","that","this","into","over","under","about","your","you",
        "to","of","in","on","a","an","it","is","are","as","by","at","be","or","we","our","how"
    }
    toks = [w for w in toks if w not in stop and len(w) >= 3]
    counts = Counter(toks)
    return [w for w, _ in counts.most_common(n)]
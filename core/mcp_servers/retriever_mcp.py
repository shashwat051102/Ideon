from __future__ import annotations
import argparse
import json
from typing import Any, Dict, Optional

from core.models.retrievers import SemanticRetriever
from core.database.chroma_manager import ChromaManager
from core.models.embeddings import EmbeddingModel


def parse_where(where_str: Optional[str]) -> Optional[Dict[str, Any]]:
    if not where_str or not where_str.strip():
        return None
    clauses = []
    for part in [p.strip() for p in where_str.split(",") if p.strip()]:
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()
        if v.lower() in ("true", "false"):
            val: Any = (v.lower() == "true")
        else:
            try:
                val = int(v)
            except ValueError:
                try:
                    val = float(v)
                except ValueError:
                    val = v
        clauses.append({k: {"$eq": val}})
    if not clauses:
        return None
    if len(clauses) == 1:
        k = next(iter(clauses[0].keys()))
        return {k: clauses[0][k]}
    return {"$and": clauses}


def main():
    ap = argparse.ArgumentParser(description="Retriever MCP — semantic search over chunks/ideas")
    ap.add_argument("--type", choices=["chunks", "ideas"], default="chunks", help="Collection to search")
    ap.add_argument("--query", "-q", required=True, help="Query text")
    ap.add_argument("--k", type=int, default=5, help="Top-k results")
    ap.add_argument("--where", default="", help="Filters as key=value pairs, comma-separated")
    ap.add_argument("--json", action="store_true", help="Output JSON")
    args = ap.parse_args()

    retriever = SemanticRetriever(chroma=ChromaManager(), embedder=EmbeddingModel())
    where = parse_where(args.where)
    results = retriever.search_ideas(args.query, top_k=args.k, where=where) if args.type == "ideas" else retriever.search_chunks(args.query, top_k=args.k, where=where)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    if not results:
        print("No results.")
        return

    for i, r in enumerate(results, 1):
        md = r.get("metadata") or {}
        doc = (r.get("document") or "")
        snippet = doc[:160].replace("\n", " ")
        tail = "..." if len(doc) > 160 else ""
        print(f"{i}. id={r.get('id')} dist={r.get('distance')}")
        print(f"   category={md.get('category')} filepath={md.get('filepath')}")
        print(f"   text: {snippet}{tail}")


if __name__ == "__main__":
    main()
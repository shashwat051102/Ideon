from __future__ import annotations
import json
import argparse

from core.crews.idea_agent import IdeaGeneratorAgent

def main():
    ap = argparse.ArgumentParser(description="Model MCP — Idea Generator")
    ap.add_argument("--db", default="storage/sqlite/metadata.db", help="SQLite DB path")
    ap.add_argument("--user", default="local_user", help="Username")
    ap.add_argument("--prompt", "-p", required=True, help="Seed prompt")
    ap.add_argument("--ideas", type=int, default=5, help="Number of ideas to generate")
    ap.add_argument("--ctx", type=int, default=5, help="Top-k context chunks to retrieve")
    ap.add_argument("--tags", default="", help="Comma-separated tags to attach")
    ap.add_argument("--json", action="store_true", help="Print JSON result")
    args = ap.parse_args()
    
    
    agent = IdeaGeneratorAgent(db_path=args.db)
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    result = agent.generate(
        username=args.user,
        prompt=args.prompt,
        n_ideas=args.ideas,
        ctx_top_k=args.ctx,
        tags=tags,
    )
    
    
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("Idea generation complete:")
        print(f"- user:   {result['username']}")
        print(f"- count:  {result['count']}")
        for nid in result["created_idea_ids"]:
            print(f"  • {nid}")


if __name__ == "__main__":
    main()
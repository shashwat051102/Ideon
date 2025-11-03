from __future__ import annotations
import argparse
import json

from core.crews.graph_agent import GraphAgent


def main():
    
    ap = argparse.ArgumentParser(description="Graph MCP — auto-link and cluster ideas")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("autolink-one", help="Auto-link a single idea to similar ideas")
    s1.add_argument("--id", required=True, help="Idea node_id")
    s1.add_argument("--k", type=int, default=5, help="Top-k similar links to create")

    s2 = sub.add_parser("autolink-recent", help="Auto-link most recent ideas")
    s2.add_argument("--limit", type=int, default=10, help="How many recent ideas")
    s2.add_argument("--k", type=int, default=5, help="Top-k per idea")

    s3 = sub.add_parser("cluster", help="Cluster ideas and create cluster edges")
    s3.add_argument("--k", type=int, default=3, help="Number of clusters")
    s3.add_argument("--json", action="store_true", help="Output JSON summary")

    s4 = sub.add_parser("list-edges", help="List edges for a given node")
    s4.add_argument("--id", required=True, help="Node id (idea or cluster)")

    args = ap.parse_args()
    agent = GraphAgent()
    
    
    if args.cmd == "autolink-one":
        created = agent.autolink_for_node(args.id, top_k=args.k)
        
        print(json.dumps(created, ensure_ascii=False, indent=2))
        return
    
    if args.cmd == "autolink-recent":
        res = agent.autolink_recent(Limit=args.limit, top_k=args.k)
        
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return
    
    if args.cmd == "cluster":
        res = agent.cluster_ideas(k=args.k)
        
        if args.json:
            print(json.dumps(res, ensure_ascii=False, indent=2))
        else:
            for c in res.get("clusters", []):
                print(f"{c['cluster_id']}: {c['label']} (n={c['size']}) -> {', '.join(c['sample_titles'])}")
        return
    
    if args.cmd == "list-edges":
        edges = agent.db.list_edges_for_node(args.id)
        print(json.dumps(edges, ensure_ascii=False, indent=2))
        return
    
if __name__ == "__main__":
    main()
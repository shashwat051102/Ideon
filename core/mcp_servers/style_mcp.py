import argparse
from core.crews.style_agent import StyleLearnerAgent

def main():
    parser = argparse.ArgumentParser(description="Style MCP — Build Voice Profile from writing samples")
    parser.add_argument("--db", default="storage/sqlite/metadata.db", help="SQLite DB path")
    parser.add_argument("--user", default="local_user", help="Username to attach the profile to")
    parser.add_argument("--name", default="Default Voice", help="Voice profile name")
    parser.add_argument("--max_files", type=int, default=50, help="Max writing sample files to use")
    parser.add_argument("--max_chars", type=int, default=4000, help="Max chars per file")
    args = parser.parse_args()
    
    
    agent = StyleLearnerAgent(db_path=args.db)
    result = agent.learn_voice_profile(
        username=args.user,
        profile_name=args.name,
        max_files=args.max_files,
        max_chars_per_file=args.max_chars,
    )
    
    print("Voice profile created:")
    print(f"- profile_id: {result['profile_id']}")
    print(f"- username:   {result['username']}")
    print(f"- name:       {result['profile_name']}")
    print(f"- files used: {len(result['source_file_ids'])}")
    print(f"- metrics:    {result['metrics']}")
    

if __name__ == "__main__":
    main()
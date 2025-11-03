from __future__ import annotations
from typing import Dict, List, Optional, Set

from core.database.sqlite_manager import SQLiteManager
from core.crews.graph_agent import GraphAgent
from core.models import generator

# Optional crewai-style Agent template (no external params file)
try:
    from crewai import Agent as CrewAgent
except Exception:  # crewai may be optional in this project
    CrewAgent = None  # type: ignore

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None  # type: ignore

# Local in-code agent template (role/goal/backstory)
LOCAL_AGENT = {
    "role": "Collective Idea Synthesizer",
    "goal": (
        "Combine a seed idea and its related neighbors from the graph into a clear, cohesive collective idea."
    ),
    "backstory": (
        "You analyze semantically linked ideas in a knowledge map and synthesize them into a unified, well-structured write-up."
    ),
}


class CollectiveIdeaAgent:
    """
    Creates a collective idea by synthesizing a seed node and its connected neighbors.
    If no neighbors exist and autolink_if_needed is True, it will attempt to autolink
    the seed to its nearest neighbors first.
    """

    def __init__(self):
        self.db = SQLiteManager()
        self.g = GraphAgent()

    def _pick_seed(self) -> Optional[str]:
        # Pick the most recent node that already has edges, otherwise the most recent node.
        nodes = self.db.list_idea_nodes(limit=50)
        for r in nodes:
            nid = r.get("node_id")
            if not nid:
                continue
            edges = self.db.list_edges_for_node(nid, limit=1)
            if edges:
                return nid
        return nodes[0]["node_id"] if nodes else None

    def _neighbor_ids(self, seed_id: str) -> Set[str]:
        edges = self.db.list_edges_for_node(seed_id, limit=100)
        nids: Set[str] = set()
        for e in edges:
            s = e.get("src_id"); d = e.get("dst_id")
            if s and s != seed_id:
                nids.add(s)
            if d and d != seed_id:
                nids.add(d)
        return nids

    def create_collective_idea(
        self,
        seed_id: Optional[str] = None,
        autolink_if_needed: bool = True,
        top_k: int = 5,
        prompt: str = "Create a single collective idea from these connected ideas.",
        require_llm: bool = False,
    ) -> Dict:
        # Resolve seed
        sid = seed_id or self._pick_seed()
        if not sid:
            return {"error": "No ideas available to synthesize."}

        # Ensure we have neighbors
        neighbors = self._neighbor_ids(sid)
        if autolink_if_needed and not neighbors:
            try:
                self.g.autolink_for_node(
                    sid,
                    top_k=top_k,
                    max_distance=1.2,
                    require_tag_overlap=False,
                    min_tag_overlap=0,
                    close_override_distance=0.45,
                )
                neighbors = self._neighbor_ids(sid)
            except Exception as e:
                return {"error": f"Autolink failed: {e}"}

        if not neighbors:
            return {"error": "Seed has no connected neighbors; add related ideas or run autolink."}

        # Load seed + neighbor nodes
        seed = self.db.get_idea_node(sid)
        if not seed:
            return {"error": "Seed node not found."}
        nodes: List[Dict] = [seed]
        for nid in neighbors:
            r = self.db.get_idea_node(nid)
            if r:
                nodes.append(r)

        # Optionally enforce LLM requirement
        if require_llm and generator.get_llm() is None:
            return {"error": "LLM is not configured. Set IDEAWEAVER_USE_LLM=true and OPENAI_API_KEY to use LLM synthesis."}

        # Synthesize using generator (LLM if configured, local otherwise)
        try:
            result = generator.compose_from_nodes(
                prompt=prompt,
                nodes=nodes,
                extra_tags=["collective"],
            )
        except Exception as e:
            return {"error": f"Synthesis failed: {e}"}

        return {
            "ok": True,
            "seed_id": sid,
            "source_nodes": [n.get("node_id") for n in nodes if n.get("node_id")],
            "title": result.get("title"),
            "content": result.get("content"),
            "tags": result.get("tags"),
        }

    # Keep a simple run() wrapper to match prior agent templates
    def run(
        self,
        seed_id: Optional[str] = None,
        top_k: int = 5,
        autolink_if_needed: bool = True,
        prompt: str = "Create a single collective idea from these connected ideas.",
        require_llm: bool = False,
    ) -> Dict:
        return self.create_collective_idea(
            seed_id=seed_id,
            autolink_if_needed=autolink_if_needed,
            top_k=top_k,
            prompt=prompt,
            require_llm=require_llm,
        )

    # Provide a crewai-style Agent builder to match original template usage (no YAML)
    def agent(self):
        """
        Build and return a crewai.Agent using local in-code config.
        Safe if crewai/langchain_openai aren't installed: will return None.
        """
        if CrewAgent is None:
            return None
        llm = None
        if ChatOpenAI is not None:
            # Let ChatOpenAI pick up OPENAI_API_KEY from environment; match generator default model
            try:
                llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3)
            except Exception:
                llm = None
        return CrewAgent(
            role=LOCAL_AGENT["role"],
            goal=LOCAL_AGENT["goal"],
            backstory=LOCAL_AGENT["backstory"],
            verbose=True,
            llm=llm,
        )
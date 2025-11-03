from __future__ import annotations
from typing import Optional, Dict
from crewai import Task
import logging
logger = logging.getLogger(__name__)
from .agent import CollectiveIdeaAgent

LOCAL_TASK = {
    "description": (
        "Synthesize a collective idea from a seed idea and its connected neighbors in the graph. "
        "If the seed has no neighbors, attempt to autolink similar ideas (top-k) and then synthesize."
    ),
    "expected_output": (
        "Return a JSON object with keys: ok (bool), seed_id (str), source_nodes (list[str]), "
        "title (str), content (str), tags (list[str]) or an error key with a message."
    ),
}


class CollectiveIdeaTask:
    def __init__(self):
        self.agent = CollectiveIdeaAgent()
        self.config = LOCAL_TASK

    def run(
        self,
        seed_id: Optional[str] = None,
        top_k: int = 5,
        autolink_if_needed: bool = True,
        prompt: Optional[str] = None,
        require_llm: bool = False,
    ) -> Dict:
        logger.info("[CollectiveIdeaTask] run seed_id=%s top_k=%s autolink=%s", seed_id, top_k, autolink_if_needed)
        return self.agent.create_collective_idea(
            seed_id=seed_id,
            autolink_if_needed=autolink_if_needed,
            top_k=top_k,
            prompt=prompt or "Create a single collective idea from these connected ideas.",
            require_llm=require_llm,
        )

    # Optional: if an external workflow expects a crewai.Task object, expose a builder
    def as_crewai_task(self) -> Task:
        # Provide a thin wrapper that, when executed by your orchestrator, calls self.run().
        # Note: this requires the surrounding framework to actually invoke .run(); here we only
        # provide metadata consistent with the previous pattern.
        try:
            crew_agent = self.agent.agent()  # Build a crewai.Agent using in-code template
        except Exception:
            crew_agent = None
        return Task(
            description=self.config["description"],
            expected_output=self.config["expected_output"],
            agent=crew_agent,
        )
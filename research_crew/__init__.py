"""
Deprecated package kept only for backward compatibility with the example.
Prefer importing from core.Collective_ideas.crew:

	from core.Collective_ideas.crew import CollectiveCrew as ResearchCrew
"""
from .crew import ResearchCrew  # noqa: F401

__all__ = ["ResearchCrew"]

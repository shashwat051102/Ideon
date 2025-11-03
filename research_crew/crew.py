"""
Deprecated alias. Import from core.Collective_ideas.crew instead:

	from core.Collective_ideas.crew import CollectiveCrew as ResearchCrew

This module remains temporarily to avoid breaking paths and will be removed.
"""
from __future__ import annotations
import warnings
from core.Collective_ideas.crew import CollectiveCrew as ResearchCrew  # type: ignore

warnings.warn(
	"research_crew.crew is deprecated; import from core.Collective_ideas.crew instead",
	DeprecationWarning,
	stacklevel=2,
)

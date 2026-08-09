"""TrajJudge — score agent trajectories for loops, tool hygiene, and leaks."""

from .scoring import ScoreReport, score_trajectory

__version__ = "0.1.0"
__all__ = ["ScoreReport", "score_trajectory", "__version__"]

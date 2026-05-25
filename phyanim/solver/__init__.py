"""Solver backends for physics segments."""

from phyanim.solver.scipy_solver import ScipySegmentSolver
from phyanim.solver.heyoka_solver import HeyokaSegmentSolver

__all__ = ["ScipySegmentSolver", "HeyokaSegmentSolver"]

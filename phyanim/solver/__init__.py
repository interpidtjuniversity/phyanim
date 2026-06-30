"""Solver backends for physics segments."""

from phyanim.solver.scipy_solver import ScipySegmentSolver
from phyanim.solver.heyoka_solver import HeyokaSegmentSolver
from phyanim.solver.kinematic_solver import KinematicSegmentSolver

__all__ = ["ScipySegmentSolver", "HeyokaSegmentSolver", "KinematicSegmentSolver"]

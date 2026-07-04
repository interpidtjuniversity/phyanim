"""Solver backends for physics segments."""

from phyanim.solver.scipy_solver import ScipySegmentSolver

# heyoka is an optional dependency — import lazily so the package works
# without it installed.  The solver code is preserved; it just can't be
# used unless heyoka.py is available.
try:
    from phyanim.solver.heyoka_solver import HeyokaSegmentSolver
except ImportError:
    HeyokaSegmentSolver = None  # type: ignore[assignment,misc]

__all__ = ["ScipySegmentSolver", "HeyokaSegmentSolver"]

"""Physics-first animation modeling toolkit."""

from phyanim.core.animation import PhysicsAnimation
from phyanim.core.events import EventCondition, PhysicsEvent, StateTransition, time_end_event, time_countdown_event
from phyanim.core.objects import PhysicObject2D, PointParticle, object2d, TraceConfig
from phyanim.core.segment import PhysicsSegment
from phyanim.core.state import Parameter, StateVariable

# High-expression API for code/hybrid render modes.
from phyanim.api import TrajectoryData, solve_animation

__all__ = [
    "EventCondition",
    "Parameter",
    "PhysicObject2D",
    "PhysicsAnimation",
    "PhysicsEvent",
    "PhysicsSegment",
    "PointParticle",
    "StateTransition",
    "StateVariable",
    "TraceConfig",
    "TrajectoryData",
    "object2d",
    "solve_animation",
    "time_end_event",
    "time_countdown_event",
]

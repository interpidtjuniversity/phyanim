"""Physics-first animation modeling toolkit."""

from phyanim.core.animation import PhysicsAnimation
from phyanim.core.events import EventCondition, PhysicsEvent, StateTransition, time_end_event
from phyanim.core.keyframe import PhysicsKeyFrame
from phyanim.core.objects import PhysicObject, PointParticle
from phyanim.core.segment import PhysicsSegment
from phyanim.core.solution import SegmentResult, SegmentSolution, StateFunction
from phyanim.core.state import Parameter, StateVariable
from phyanim.core.trajectory import InterpolatedStateFunction

__all__ = [
    "EventCondition",
    "InterpolatedStateFunction",
    "Parameter",
    "PhysicObject",
    "PhysicsAnimation",
    "PhysicsEvent",
    "PhysicsKeyFrame",
    "PhysicsSegment",
    "PointParticle",
    "SegmentResult",
    "SegmentSolution",
    "StateFunction",
    "StateTransition",
    "StateVariable",
    "time_end_event",
]

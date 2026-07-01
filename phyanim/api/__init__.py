"""High-expression API for PhyAnim code/hybrid render modes.

This package re-exports everything that LLM-generated render code needs:
- TrajectoryData and solve_animation for physics trajectory access
- Helper functions for common manim patterns
- All 2D geometry entities
- Common manim imports
"""

from phyanim.api.trajectory import (
    TrajectoryData,
    SegmentInfo,
    ObjectInfo,
    solve_animation,
)
from phyanim.api.helpers import (
    create_tracker,
    attach_position_updater,
    attach_expr_updater,
    interpolate_trajectory,
    create_bound_mobject,
)

# Geometry entities.
from phyanim.core.entity.TwoD import (
    Block,
    CircularArcTrack,
    ConcaveTrack,
    ConvexTrack,
    InclinedPlane,
    LeftSemicircleTrack,
    Pulley,
    RightSemicircleTrack,
    Spring,
    StraightTrack,
    StraightTrackGroup,
    VectorArrow,
)

# Core classes for building physics animations.
from phyanim.core.animation import PhysicsAnimation
from phyanim.core.objects import PhysicObject2D, PointParticle, object2d, TraceConfig
from phyanim.core.state import Parameter, StateVariable
from phyanim.core.segment import PhysicsSegment
from phyanim.core.events import (
    EventCondition,
    PhysicsEvent,
    StateTransition,
    time_end_event,
    time_countdown_event,
)

# Enhance layer (annotations, triggers, transitions, time wrappers).
from phyanim.core.enhance.annotation import (
    Annotation,
    AnnotationActivation,
    ArrowContent,
    MathTexContent,
    TextContent,
)
from phyanim.core.enhance.trigger import (
    ActiveBetween,
    ActiveRule,
    ActiveWhile,
    ActivateEventTimeRange,
    CrossingTrigger,
    Trigger,
)
from phyanim.core.enhance.transition import Transition
from phyanim.core.enhance.timewrapper import TimeWrapper
from phyanim.core.enhance.visual_binding import VisualBinding

__all__ = [
    # Trajectory
    "TrajectoryData",
    "SegmentInfo",
    "ObjectInfo",
    "solve_animation",
    # Helpers
    "create_tracker",
    "attach_position_updater",
    "attach_expr_updater",
    "create_bound_mobject",
    "interpolate_trajectory",
    # Geometry
    "Block",
    "CircularArcTrack",
    "ConcaveTrack",
    "ConvexTrack",
    "InclinedPlane",
    "LeftSemicircleTrack",
    "Pulley",
    "RightSemicircleTrack",
    "Spring",
    "StraightTrack",
    "StraightTrackGroup",
    "VectorArrow",
    # Core
    "PhysicsAnimation",
    "PhysicObject2D",
    "TraceConfig",
    "PointParticle",
    "object2d",
    "Parameter",
    "StateVariable",
    "PhysicsSegment",
    "EventCondition",
    "PhysicsEvent",
    "StateTransition",
    "time_end_event",
    "time_countdown_event",
    # Enhance
    "Annotation",
    "AnnotationActivation",
    "ArrowContent",
    "MathTexContent",
    "TextContent",
    "ActiveBetween",
    "ActiveRule",
    "ActiveWhile",
    "ActivateEventTimeRange",
    "CrossingTrigger",
    "Trigger",
    "Transition",
    "TimeWrapper",
    "VisualBinding",
]

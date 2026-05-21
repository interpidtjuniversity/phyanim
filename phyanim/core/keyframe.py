from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PhysicsKeyFrame:
    """A physical boundary between segments, not just a visual keyframe."""

    time: float
    object_states: dict[str, dict[str, float]]
    parameters: dict[str, float] = field(default_factory=dict)
    segment_id: str | None = None
    event_name: str | None = None
    
    # 这个关键帧里某个物体的状态变量
    def state_for(self, object_id: str) -> dict[str, float]:
        try:
            return dict(self.object_states[object_id])
        except KeyError as exc:
            raise KeyError(f"Keyframe has no state for object '{object_id}'.") from exc

    def states_for(self, object_ids: list[str]) -> dict[str, dict[str, float]]:
        return {object_id: self.state_for(object_id) for object_id in object_ids}

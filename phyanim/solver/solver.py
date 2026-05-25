from typing import Any

class Solver:
    def __init__(self):
        pass

    def _split_state_history_by_object(self,segment: Any,state_history: dict[str, list[float]]) -> dict[str, dict[str, list[float]]]:
        object_states = {object_id: {} for object_id in segment.object_ids or []}

        if segment.state_owners is None:
            raise ValueError(f"Segment '{segment.segment_id}' has no state_owners.")

        for name, values in state_history.items():
            object_id = segment.state_owners[name]
            object_states[object_id][name] = values

        return object_states
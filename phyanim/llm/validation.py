from __future__ import annotations

from typing import Any


class IRValidationError(ValueError):
    pass


def validate_physics_ir(ir: dict[str, Any]) -> None:
    if not isinstance(ir, dict):
        raise IRValidationError("IR must be a JSON object.")

    for key in ("scene", "objects", "initial_keyframe", "segments", "render"):
        if key not in ir:
            raise IRValidationError(f"IR missing required key: {key}")

    _validate_scene(ir["scene"])
    object_ids = _validate_objects(ir["objects"])
    _validate_initial_keyframe(ir["initial_keyframe"], object_ids)
    _validate_segments(ir["segments"], object_ids)
    _validate_render(ir["render"], object_ids)


def _validate_scene(scene: Any) -> None:
    if not isinstance(scene, dict):
        raise IRValidationError("scene must be an object.")
    if scene.get("dimension") != 2:
        raise IRValidationError("Only 2D scenes are supported in the initial design.")


def _validate_objects(objects: Any) -> set[str]:
    if not isinstance(objects, list) or not objects:
        raise IRValidationError("objects must be a non-empty list.")
    object_ids: set[str] = set()
    for obj in objects:
        if not isinstance(obj, dict):
            raise IRValidationError("Each object must be an object.")
        object_id = obj.get("id")
        if not isinstance(object_id, str) or not object_id:
            raise IRValidationError("Each object requires a non-empty string id.")
        if object_id in object_ids:
            raise IRValidationError(f"Duplicate object id: {object_id}")
        object_ids.add(object_id)
        state_variables = obj.get("state_variables")
        if not isinstance(state_variables, list) or not state_variables:
            raise IRValidationError(f"Object '{object_id}' requires state_variables.")
        _validate_symbol_names(state_variables, f"object '{object_id}'")
    return object_ids


def _validate_initial_keyframe(initial_keyframe: Any, object_ids: set[str]) -> None:
    if not isinstance(initial_keyframe, dict):
        raise IRValidationError("initial_keyframe must be an object.")
    object_states = initial_keyframe.get("object_states")
    if not isinstance(object_states, dict):
        raise IRValidationError("initial_keyframe.object_states must be an object.")
    missing = object_ids - set(object_states)
    if missing:
        raise IRValidationError(f"Initial keyframe missing object states: {sorted(missing)}")
    for object_id, state in object_states.items():
        if object_id not in object_ids:
            raise IRValidationError(f"Initial keyframe references unknown object: {object_id}")
        if not isinstance(state, dict):
            raise IRValidationError(f"Initial keyframe state for '{object_id}' must be an object.")
        _validate_symbol_names(state.keys(), f"initial_keyframe.{object_id}")


def _validate_segments(segments: Any, object_ids: set[str]) -> None:
    if not isinstance(segments, list) or not segments:
        raise IRValidationError("segments must be a non-empty list.")
    for segment in segments:
        if not isinstance(segment, dict):
            raise IRValidationError("Each segment must be an object.")
        segment_id = segment.get("id")
        if not isinstance(segment_id, str) or not segment_id:
            raise IRValidationError("Each segment requires an id.")

        segment_type = segment.get("type", "ode")
        if segment_type in {"transition", "instant_transition", "event_transition"} or "transition" in segment:
            transition = segment.get("transition")
            if not isinstance(transition, dict):
                raise IRValidationError(f"Transition segment '{segment_id}' requires transition.")
            segment_object_ids = segment.get("object_ids", [])
            if segment_object_ids:
                unknown = set(segment_object_ids) - object_ids
                if unknown:
                    raise IRValidationError(
                        f"Transition segment '{segment_id}' references unknown objects: {sorted(unknown)}"
                    )
            continue

        segment_object_ids = segment.get("object_ids")
        if not isinstance(segment_object_ids, list) or not segment_object_ids:
            raise IRValidationError(f"Segment '{segment_id}' requires object_ids.")
        unknown = set(segment_object_ids) - object_ids
        if unknown:
            raise IRValidationError(f"Segment '{segment_id}' references unknown objects: {sorted(unknown)}")

        state_vector = segment.get("state_vector")
        equations = segment.get("equations")
        if not isinstance(state_vector, list) or not state_vector:
            raise IRValidationError(f"Segment '{segment_id}' requires state_vector.")
        _validate_symbol_names(state_vector, f"segment '{segment_id}'")
        if not isinstance(equations, dict):
            raise IRValidationError(f"Segment '{segment_id}' requires equations.")
        missing_equations = [name for name in state_vector if name not in equations]
        if missing_equations:
            raise IRValidationError(
                f"Segment '{segment_id}' missing derivative equations for: {missing_equations}"
            )
        end_condition = segment.get("end_condition")
        if not isinstance(end_condition, dict):
            raise IRValidationError(f"Segment '{segment_id}' requires end_condition.")


def _validate_render(render: Any, object_ids: set[str]) -> None:
    if not isinstance(render, dict):
        raise IRValidationError("render must be an object.")
    render_objects = render.get("objects")
    if not isinstance(render_objects, list):
        raise IRValidationError("render.objects must be a list.")
    for item in render_objects:
        if not isinstance(item, dict):
            raise IRValidationError("Each render object must be an object.")
        object_id = item.get("object_id")
        if object_id not in object_ids:
            raise IRValidationError(f"Render references unknown object: {object_id}")


def _validate_symbol_names(names: Any, context: str) -> None:
    seen: set[str] = set()
    for name in names:
        if not isinstance(name, str):
            raise IRValidationError(f"{context} has a non-string state name.")
        if not name.isidentifier():
            raise IRValidationError(f"{context} state '{name}' must be a valid Python/SymPy identifier.")
        if name in seen:
            raise IRValidationError(f"{context} contains duplicate state name: {name}")
        seen.add(name)

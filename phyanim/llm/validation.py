"""Validation for the PhyAnim JSON DSL.

``validate_dsl`` is the single entry point used by the planner (before returning
a DSL to the caller) and by tests. It raises :class:`IRValidationError` on any
structural problem so that the planner can feed the error back to the LLM for a
self-repair round-trip.

The validator is intentionally strict: every state referenced anywhere must be
declared by some object, every multi-object segment must declare owners, and
equation keys must be bare state names (not ``x_dot``).
"""

from __future__ import annotations

from typing import Any

from phyanim.llm.dsl_schema import (
    ANNOTATION_TYPES,
    CONTENT_KINDS,
    ENGINES,
    GEOMETRY_KINDS,
    TIME_WRAPPER_TYPES,
    TRANSITION_DIRS,
    TRANSITION_STYLES,
)


class IRValidationError(ValueError):
    """Raised when a DSL document violates the schema."""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def validate_dsl(ir: dict[str, Any]) -> None:
    """Validate a parsed DSL document in place.

    Raises :class:`IRValidationError` on the first structural problem found.
    """
    if not isinstance(ir, dict):
        raise IRValidationError("DSL must be a JSON object.")

    for key in ("scene", "objects", "initial_states", "segments"):
        if key not in ir:
            raise IRValidationError(f"DSL missing required key: {key}")

    _validate_scene(ir["scene"])
    object_ids, object_state_names = _validate_objects(ir["objects"])
    _validate_initial_states(ir["initial_states"], object_ids, object_state_names)
    _validate_global_parameters(ir.get("global_parameters"))
    _validate_segments(ir["segments"], object_ids, object_state_names)
    if "physics_layer" in ir and ir["physics_layer"] is not None:
        _validate_physics_layer(ir["physics_layer"], object_ids, object_state_names)
    if "render_layer" in ir and ir["render_layer"] is not None:
        _validate_render_layer(ir["render_layer"], object_ids, object_state_names)


# Backward-compatible alias; old IR used validate_physics_ir.
validate_physics_ir = validate_dsl


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

def _validate_scene(scene: Any) -> None:
    if not isinstance(scene, dict):
        raise IRValidationError("scene must be an object.")
    if scene.get("dimension") != 2:
        raise IRValidationError("Only 2D scenes are supported.")
    engine = scene.get("engine", "scipy")
    if engine not in ENGINES:
        raise IRValidationError(
            f"scene.engine must be one of {ENGINES}, got {engine!r}."
        )
    sample_dt = scene.get("sample_dt", 1 / 60)
    if not isinstance(sample_dt, (int, float)) or sample_dt <= 0:
        raise IRValidationError("scene.sample_dt must be a positive number.")


# ---------------------------------------------------------------------------
# Objects
# ---------------------------------------------------------------------------

def _validate_objects(objects: Any) -> tuple[set[str], dict[str, set[str]]]:
    """Return (object_ids, {object_id: {state_names}})."""
    if not isinstance(objects, list) or not objects:
        raise IRValidationError("objects must be a non-empty list.")

    object_ids: set[str] = set()
    object_state_names: dict[str, set[str]] = {}

    for obj in objects:
        if not isinstance(obj, dict):
            raise IRValidationError("Each object must be an object.")
        object_id = obj.get("id")
        _require_identifier(object_id, "object id")
        if object_id in object_ids:
            raise IRValidationError(f"Duplicate object id: {object_id}")
        object_ids.add(object_id)

        obj_type = obj.get("type", "object2d")
        if obj_type not in ("point_particle", "object2d"):
            raise IRValidationError(
                f"Object '{object_id}' type must be 'point_particle' or 'object2d', got {obj_type!r}."
            )

        # Collect declared state names.
        states: set[str] = set()
        if obj_type == "point_particle":
            # PointParticle creates one state per entry in state_names. If
            # state_names is absent, it defaults to x/y/vx/vy. Partial
            # state_names fully replaces the defaults (matching the runtime).
            state_names = obj.get("state_names") or {}
            if not isinstance(state_names, dict):
                raise IRValidationError(
                    f"Object '{object_id}' state_names must be an object."
                )
            if state_names:
                effective = state_names
            else:
                effective = {"x": "x", "y": "y", "vx": "vx", "vy": "vy"}
            for std, name in effective.items():
                _require_identifier(name, f"object '{object_id}' state name for '{std}'")
                if name in states:
                    raise IRValidationError(
                        f"Object '{object_id}' has duplicate state name: {name}"
                    )
                states.add(name)
        else:
            raw_states = obj.get("states", [])
            if not isinstance(raw_states, list):
                raise IRValidationError(f"Object '{object_id}' states must be a list.")
            for name in raw_states:
                _require_identifier(name, f"object '{object_id}' state name")
                if name in states:
                    raise IRValidationError(
                        f"Object '{object_id}' has duplicate state name: {name}"
                    )
                states.add(name)

        object_state_names[object_id] = states

        # Validate parameters.
        parameters = obj.get("parameters", {})
        if parameters is None:
            parameters = {}
        if not isinstance(parameters, dict):
            raise IRValidationError(f"Object '{object_id}' parameters must be an object.")
        for pname in parameters:
            _require_identifier(pname, f"object '{object_id}' parameter")
        # Parameters must not collide with state names of the same object.
        collision = states & set(parameters)
        if collision:
            raise IRValidationError(
                f"Object '{object_id}' parameter names collide with state names: {sorted(collision)}"
            )

        # Validate geometry.
        geometry = obj.get("geometry", "circle")
        if geometry not in GEOMETRY_KINDS:
            raise IRValidationError(
                f"Object '{object_id}' geometry must be one of {GEOMETRY_KINDS}, got {geometry!r}."
            )
        geometry_params = obj.get("geometry_params", {})
        if not isinstance(geometry_params, dict):
            raise IRValidationError(f"Object '{object_id}' geometry_params must be an object.")

        # Validate cartesian_position.
        cart = obj.get("cartesian_position", [])
        if not isinstance(cart, list):
            raise IRValidationError(f"Object '{object_id}' cartesian_position must be a list.")
        for pair in cart:
            if not isinstance(pair, list) or len(pair) != 2:
                raise IRValidationError(
                    f"Object '{object_id}' cartesian_position entries must be [x, y] pairs."
                )

    return object_ids, object_state_names


# ---------------------------------------------------------------------------
# Initial states
# ---------------------------------------------------------------------------

def _validate_initial_states(
    initial_states: Any,
    object_ids: set[str],
    object_state_names: dict[str, set[str]],
) -> None:
    if not isinstance(initial_states, dict):
        raise IRValidationError("initial_states must be an object.")
    missing = object_ids - set(initial_states)
    if missing:
        raise IRValidationError(f"initial_states missing objects: {sorted(missing)}")
    unknown = set(initial_states) - object_ids
    if unknown:
        raise IRValidationError(f"initial_states references unknown objects: {sorted(unknown)}")

    for object_id, state_map in initial_states.items():
        if not isinstance(state_map, dict):
            raise IRValidationError(f"initial_states for '{object_id}' must be an object.")
        declared = object_state_names[object_id]
        # Stateless visual objects (states=[]) are allowed to have empty or absent maps.
        if not declared:
            continue
        missing_states = declared - set(state_map)
        if missing_states:
            raise IRValidationError(
                f"initial_states for '{object_id}' missing states: {sorted(missing_states)}"
            )
        for sname in state_map:
            _require_identifier(sname, f"initial_states '{object_id}' state")


# ---------------------------------------------------------------------------
# Global parameters
# ---------------------------------------------------------------------------

def _validate_global_parameters(params: Any) -> None:
    if params is None:
        return
    if not isinstance(params, dict):
        raise IRValidationError("global_parameters must be an object.")
    for name in params:
        _require_identifier(name, "global parameter")
    for name, value in params.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise IRValidationError(
                f"global_parameters '{name}' must be a number, got {value!r}."
            )


# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------

def _validate_segments(
    segments: Any,
    object_ids: set[str],
    object_state_names: dict[str, set[str]],
) -> None:
    if not isinstance(segments, list) or not segments:
        raise IRValidationError("segments must be a non-empty list.")
    seen_ids: set[str] = set()
    for segment in segments:
        if not isinstance(segment, dict):
            raise IRValidationError("Each segment must be an object.")
        segment_id = segment.get("id")
        _require_identifier(segment_id, "segment id")
        if segment_id in seen_ids:
            raise IRValidationError(f"Duplicate segment id: {segment_id}")
        seen_ids.add(segment_id)

        object_id_list = segment.get("object_ids")
        if not isinstance(object_id_list, list) or not object_id_list:
            raise IRValidationError(f"Segment '{segment_id}' requires non-empty object_ids.")
        unknown = set(object_id_list) - object_ids
        if unknown:
            raise IRValidationError(
                f"Segment '{segment_id}' references unknown objects: {sorted(unknown)}"
            )

        equations = segment.get("equations")
        if not isinstance(equations, dict) or not equations:
            raise IRValidationError(f"Segment '{segment_id}' requires non-empty equations.")
        for sname in equations:
            _require_identifier(sname, f"segment '{segment_id}' equation key")

        owners = segment.get("owners") or {}
        if not isinstance(owners, dict):
            raise IRValidationError(f"Segment '{segment_id}' owners must be an object.")
        # Infer / validate owners.
        if len(object_id_list) == 1 and not owners:
            owners = {sname: object_id_list[0] for sname in equations}
        for sname, owner in owners.items():
            if sname not in equations:
                raise IRValidationError(
                    f"Segment '{segment_id}' owners references unknown state: {sname}"
                )
            if owner not in object_ids:
                raise IRValidationError(
                    f"Segment '{segment_id}' owner '{owner}' is not a declared object."
                )
        # Every equation key must have an owner.
        missing_owners = set(equations) - set(owners)
        if missing_owners:
            raise IRValidationError(
                f"Segment '{segment_id}' missing owners for states: {sorted(missing_owners)}"
            )

        # Owners' states must be declared by their objects (unless the object is
        # stateless, which is invalid for a state it owns).
        for sname, owner in owners.items():
            declared = object_state_names.get(owner, set())
            if not declared:
                raise IRValidationError(
                    f"Segment '{segment_id}' state '{sname}' owned by stateless object '{owner}'."
                )
            if sname not in declared:
                raise IRValidationError(
                    f"Segment '{segment_id}' state '{sname}' is not declared by owner '{owner}'."
                )

        # Segment parameters.
        parameters = segment.get("parameters", {})
        if not isinstance(parameters, dict):
            raise IRValidationError(f"Segment '{segment_id}' parameters must be an object.")
        for pname in parameters:
            _require_identifier(pname, f"segment '{segment_id}' parameter")

        # Derived quantities.
        derived = segment.get("derived", {})
        if derived is None:
            derived = {}
        if not isinstance(derived, dict):
            raise IRValidationError(f"Segment '{segment_id}' derived must be an object.")
        for dname in derived:
            _require_identifier(dname, f"segment '{segment_id}' derived quantity")

        duration = segment.get("duration")
        if not isinstance(duration, (int, float)) or duration <= 0:
            raise IRValidationError(f"Segment '{segment_id}' duration must be a positive number.")

        end_event = segment.get("end_event")
        if not isinstance(end_event, dict):
            raise IRValidationError(f"Segment '{segment_id}' requires an end_event object.")
        _validate_end_event(end_event, segment_id)


def _validate_end_event(end_event: dict[str, Any], segment_id: str) -> None:
    if end_event.get("type") == "countdown":
        value = end_event.get("value")
        if not isinstance(value, (int, float)) or value <= 0:
            raise IRValidationError(
                f"Segment '{segment_id}' countdown end_event requires a positive value."
            )
        return

    expression = end_event.get("expression")
    if not isinstance(expression, str) or not expression.strip():
        raise IRValidationError(
            f"Segment '{segment_id}' end_event requires a non-empty expression."
        )
    direction = end_event.get("direction", 0)
    if direction not in (-1, 0, 1):
        raise IRValidationError(
            f"Segment '{segment_id}' end_event.direction must be -1, 0, or 1."
        )
    transition = end_event.get("transition", {})
    if transition is None:
        transition = {}
    if not isinstance(transition, dict):
        raise IRValidationError(
            f"Segment '{segment_id}' end_event.transition must be an object."
        )
    for sname in transition:
        _require_identifier(sname, f"segment '{segment_id}' transition state")


# ---------------------------------------------------------------------------
# Physics layer
# ---------------------------------------------------------------------------

def _validate_physics_layer(
    layer: Any,
    object_ids: set[str],
    object_state_names: dict[str, set[str]],
) -> None:
    if not isinstance(layer, dict):
        raise IRValidationError("physics_layer must be an object.")
    annotations = layer.get("annotations", [])
    if annotations is None:
        annotations = []
    if not isinstance(annotations, list):
        raise IRValidationError("physics_layer.annotations must be a list.")
    for ann in annotations:
        _validate_annotation(ann, object_ids, object_state_names)
    transitions = layer.get("transitions", [])
    if transitions is None:
        transitions = []
    if not isinstance(transitions, list):
        raise IRValidationError("physics_layer.transitions must be a list.")
    for tr in transitions:
        _validate_transition_spec(tr)


def _validate_annotation(
    ann: Any,
    object_ids: set[str],
    object_state_names: dict[str, set[str]],
) -> None:
    if not isinstance(ann, dict):
        raise IRValidationError("Each annotation must be an object.")
    ann_id = ann.get("id")
    _require_identifier(ann_id, "annotation id")
    ann_type = ann.get("type")
    if ann_type not in ANNOTATION_TYPES:
        raise IRValidationError(
            f"Annotation '{ann_id}' type must be one of {ANNOTATION_TYPES}, got {ann_type!r}."
        )
    content = ann.get("content")
    if not isinstance(content, dict):
        raise IRValidationError(f"Annotation '{ann_id}' content must be an object.")
    kind = content.get("kind")
    if kind not in CONTENT_KINDS:
        raise IRValidationError(
            f"Annotation '{ann_id}' content.kind must be one of {CONTENT_KINDS}, got {kind!r}."
        )
    pos = content.get("pos_variables")
    if not isinstance(pos, list) or len(pos) != 2:
        raise IRValidationError(
            f"Annotation '{ann_id}' content.pos_variables must be a [x, y] pair."
        )
    if kind == "arrow":
        shift = content.get("shift_variables")
        if not isinstance(shift, list) or len(shift) != 2:
            raise IRValidationError(
                f"Annotation '{ann_id}' arrow content requires shift_variables [dx, dy]."
            )
    if kind in ("text", "mathtex"):
        text = content.get("text")
        if not isinstance(text, str) or not text:
            raise IRValidationError(
                f"Annotation '{ann_id}' {kind} content requires non-empty text."
            )

    # Triggers by type.
    if ann_type == "while":
        trigger = ann.get("trigger")
        _validate_trigger(trigger, f"annotation '{ann_id}' trigger")
    elif ann_type == "time_range":
        trigger = ann.get("trigger")
        _validate_trigger(trigger, f"annotation '{ann_id}' trigger")
        advance = ann.get("advance", 0.0)
        delay = ann.get("delay", 0.0)
        if not isinstance(advance, (int, float)):
            raise IRValidationError(f"Annotation '{ann_id}' advance must be a number.")
        if not isinstance(delay, (int, float)):
            raise IRValidationError(f"Annotation '{ann_id}' delay must be a number.")
    elif ann_type == "between":
        _validate_trigger(ann.get("start_trigger"), f"annotation '{ann_id}' start_trigger")
        _validate_trigger(ann.get("end_trigger"), f"annotation '{ann_id}' end_trigger")


def _validate_trigger(trigger: Any, context: str) -> None:
    if not isinstance(trigger, dict):
        raise IRValidationError(f"{context} must be an object.")
    expression = trigger.get("expression")
    if not isinstance(expression, str) or not expression.strip():
        raise IRValidationError(f"{context} requires a non-empty expression.")
    direction = trigger.get("direction", 0)
    if direction not in (-1, 0, 1):
        raise IRValidationError(f"{context}.direction must be -1, 0, or 1.")


def _validate_transition_spec(tr: Any) -> None:
    if not isinstance(tr, dict):
        raise IRValidationError("Each transition spec must be an object.")
    tr_id = tr.get("id")
    _require_identifier(tr_id, "transition id")
    groups = tr.get("groups")
    if not isinstance(groups, list) or not groups:
        raise IRValidationError(f"Transition '{tr_id}' groups must be a non-empty list.")
    for group in groups:
        if not isinstance(group, list):
            raise IRValidationError(f"Transition '{tr_id}' each group must be a list.")
    triggers = tr.get("triggers")
    if not isinstance(triggers, list) or not triggers:
        raise IRValidationError(f"Transition '{tr_id}' triggers must be a non-empty list.")
    for trigger in triggers:
        _validate_trigger(trigger, f"transition '{tr_id}' trigger")
    pos = tr.get("pos_variables")
    if not isinstance(pos, list) or len(pos) != 2:
        raise IRValidationError(f"Transition '{tr_id}' pos_variables must be a [x, y] pair.")
    style = tr.get("style", "scale")
    if style not in TRANSITION_STYLES:
        raise IRValidationError(
            f"Transition '{tr_id}' style must be one of {TRANSITION_STYLES}, got {style!r}."
        )
    group_dir = tr.get("group_dir", {})
    if not isinstance(group_dir, dict):
        raise IRValidationError(f"Transition '{tr_id}' group_dir must be an object.")
    for idx, direction in group_dir.items():
        if direction not in TRANSITION_DIRS:
            raise IRValidationError(
                f"Transition '{tr_id}' group_dir[{idx}] must be one of {TRANSITION_DIRS}."
            )


# ---------------------------------------------------------------------------
# Render layer
# ---------------------------------------------------------------------------

def _validate_render_layer(
    layer: Any,
    object_ids: set[str],
    object_state_names: dict[str, set[str]],
) -> None:
    if not isinstance(layer, dict):
        raise IRValidationError("render_layer must be an object.")
    sub_animations = layer.get("sub_animations", [])
    if sub_animations is None:
        sub_animations = []
    if not isinstance(sub_animations, list):
        raise IRValidationError("render_layer.sub_animations must be a list.")
    for sub in sub_animations:
        _validate_sub_animation(sub, object_ids, object_state_names)


def _validate_sub_animation(
    sub: Any,
    object_ids: set[str],
    object_state_names: dict[str, set[str]],
) -> None:
    if not isinstance(sub, dict):
        raise IRValidationError("Each sub_animation must be an object.")
    wrapper = sub.get("time_wrapper")
    if not isinstance(wrapper, dict):
        raise IRValidationError("sub_animation.time_wrapper must be an object.")
    wrapper_id = wrapper.get("id")
    _require_identifier(wrapper_id, "time_wrapper id")
    wrapper_type = wrapper.get("type")
    if wrapper_type not in TIME_WRAPPER_TYPES:
        raise IRValidationError(
            f"time_wrapper '{wrapper_id}' type must be one of {TIME_WRAPPER_TYPES}, got {wrapper_type!r}."
        )
    _validate_trigger(wrapper.get("trigger"), f"time_wrapper '{wrapper_id}' trigger")
    if wrapper_type == "freeze":
        extend_to = wrapper.get("extend_to")
        if not isinstance(extend_to, (int, float)) or extend_to <= 0:
            raise IRValidationError(
                f"time_wrapper '{wrapper_id}' freeze requires positive extend_to."
            )
    elif wrapper_type == "slow":
        speed = wrapper.get("speed")
        if not isinstance(speed, (int, float)) or speed <= 0:
            raise IRValidationError(
                f"time_wrapper '{wrapper_id}' slow requires positive speed."
            )

    annotations = sub.get("annotations", [])
    if annotations is None:
        annotations = []
    if not isinstance(annotations, list):
        raise IRValidationError(
            f"sub_animation '{wrapper_id}' annotations must be a list."
        )
    for ann in annotations:
        _validate_annotation(ann, object_ids, object_state_names)
    transitions = sub.get("transitions", [])
    if transitions is None:
        transitions = []
    if not isinstance(transitions, list):
        raise IRValidationError(
            f"sub_animation '{wrapper_id}' transitions must be a list."
        )
    for tr in transitions:
        _validate_transition_spec(tr)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_identifier(name: Any, context: str) -> None:
    if not isinstance(name, str) or not name.isidentifier():
        raise IRValidationError(f"{context} must be a valid Python identifier, got {name!r}.")


__all__ = ["validate_dsl", "validate_physics_ir", "IRValidationError"]

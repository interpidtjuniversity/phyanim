"""Tests for the DSL validator and code generator.

These tests do not call any LLM. They exercise:
1. ``validate_dsl`` on a known-good golden DSL and on crafted bad DSLs.
2. ``DSLParser.parse`` producing syntactically valid Python (ast.parse).
3. The generated code referencing the expected phyanim API surface.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

# Ensure the project root is importable when running tests directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phyanim.llm.parser import DSLParser
from phyanim.llm.validation import IRValidationError, validate_dsl

GOLDEN_DSL_PATH = Path(__file__).resolve().parents[1] / "phyanim" / "llm" / "examples" / "dsl_schema.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def golden_dsl() -> dict:
    return json.loads(GOLDEN_DSL_PATH.read_text(encoding="utf-8"))


def _minimal_dsl() -> dict:
    """A minimal but complete valid DSL."""
    return {
        "scene": {"dimension": 2, "engine": "scipy", "sample_dt": 0.01667},
        "global_parameters": {"g": 9.8},
        "objects": [
            {
                "id": "ball",
                "type": "point_particle",
                "mass": 1.0,
                "radius": 0.1,
                "color": "red",
                "cartesian_position": [["x", "y"]],
                "geometry": "circle",
                "geometry_params": {"radius": 0.1, "color": "red"},
            }
        ],
        "initial_states": {"ball": {"x": 0.0, "y": 0.0, "vx": 1.0, "vy": 0.0}},
        "segments": [
            {
                "id": "free_fall",
                "object_ids": ["ball"],
                "equations": {"x": "vx", "y": "vy", "vx": "0", "vy": "-g"},
                "duration": 5,
                "end_event": {"type": "countdown", "value": 3},
            }
        ],
    }


# ---------------------------------------------------------------------------
# Validator tests
# ---------------------------------------------------------------------------

class TestValidateDSL:
    def test_golden_dsl_is_valid(self, golden_dsl: dict) -> None:
        validate_dsl(golden_dsl)

    def test_minimal_dsl_is_valid(self) -> None:
        validate_dsl(_minimal_dsl())

    def test_missing_required_key(self) -> None:
        dsl = _minimal_dsl()
        del dsl["segments"]
        with pytest.raises(IRValidationError, match="missing required key"):
            validate_dsl(dsl)

    def test_dimension_must_be_2(self) -> None:
        dsl = _minimal_dsl()
        dsl["scene"]["dimension"] = 3
        with pytest.raises(IRValidationError, match="2D"):
            validate_dsl(dsl)

    def test_dot_suffix_equation_key_rejected(self) -> None:
        """Equation keys must be bare state names, not x_dot.

        ``x_dot`` happens to be a valid Python identifier, so the real guard is
        the owner/declared-state check: x_dot is never declared by any object.
        """
        dsl = _minimal_dsl()
        dsl["segments"][0]["equations"] = {"x_dot": "vx", "y_dot": "vy"}
        with pytest.raises(IRValidationError, match="not declared by owner"):
            validate_dsl(dsl)

    def test_duplicate_object_id(self) -> None:
        dsl = _minimal_dsl()
        dsl["objects"].append(dict(dsl["objects"][0]))
        with pytest.raises(IRValidationError, match="Duplicate object id"):
            validate_dsl(dsl)

    def test_missing_initial_state(self) -> None:
        dsl = _minimal_dsl()
        dsl["initial_states"]["ball"].pop("vy")
        with pytest.raises(IRValidationError, match="missing states"):
            validate_dsl(dsl)

    def test_segment_missing_owners_multi_object(self) -> None:
        dsl = _minimal_dsl()
        dsl["objects"].append(
            {
                "id": "wall",
                "type": "object2d",
                "states": ["wx"],
                "cartesian_position": [["wx", "wx"]],
                "geometry": "none",
                "geometry_params": {},
            }
        )
        dsl["initial_states"]["wall"] = {"wx": 5.0}
        dsl["segments"][0] = {
            "id": "seg",
            "object_ids": ["ball", "wall"],
            "equations": {"x": "vx", "wx": "0"},
            "duration": 5,
            "end_event": {"type": "countdown", "value": 3},
        }
        with pytest.raises(IRValidationError, match="missing owners"):
            validate_dsl(dsl)

    def test_invalid_geometry_kind(self) -> None:
        dsl = _minimal_dsl()
        dsl["objects"][0]["geometry"] = "flux_capacitor"
        with pytest.raises(IRValidationError, match="geometry"):
            validate_dsl(dsl)

    def test_invalid_annotation_type(self) -> None:
        dsl = _minimal_dsl()
        dsl["physics_layer"] = {
            "annotations": [
                {
                    "id": "bad",
                    "type": "sometimes",
                    "content": {
                        "kind": "text",
                        "text": "hi",
                        "pos_variables": ["0", "0"],
                    },
                    "trigger": {"expression": "t > 0", "direction": 0},
                }
            ]
        }
        with pytest.raises(IRValidationError, match="type"):
            validate_dsl(dsl)

    def test_freeze_requires_extend_to(self) -> None:
        dsl = _minimal_dsl()
        dsl["render_layer"] = {
            "sub_animations": [
                {
                    "time_wrapper": {
                        "id": "w",
                        "type": "freeze",
                        "trigger": {"expression": "t > 1", "direction": 0},
                    },
                    "annotations": [],
                    "transitions": [],
                }
            ]
        }
        with pytest.raises(IRValidationError, match="extend_to"):
            validate_dsl(dsl)


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

class TestDSLParser:
    def test_parse_golden_produces_valid_python(self, golden_dsl: dict) -> None:
        code = DSLParser().parse(golden_dsl)
        # Must parse as valid Python.
        ast.parse(code)

    def test_parse_minimal_produces_valid_python(self) -> None:
        code = DSLParser().parse(_minimal_dsl())
        ast.parse(code)

    def test_generated_code_contains_key_apis(self, golden_dsl: dict) -> None:
        code = DSLParser().parse(golden_dsl)
        for snippet in (
            "PhysicsAnimation(",
            "add_object(",
            "add_segment(",
            "PhysicsSegment(",
            "get_physics_layer()",
            "add_annotation(",
            "add_transition(",
            "get_render_layer()",
            "add_sub_animation(",
            "TimeWrapper(",
            "PhyAnimationMultiLayerScene2D()",
            "scene.render()",
        ):
            assert snippet in code, f"Generated code missing: {snippet}"

    def test_generated_code_has_build_animation(self, golden_dsl: dict) -> None:
        code = DSLParser().parse(golden_dsl)
        assert "def build_animation()" in code
        assert 'if __name__ == "__main__":' in code

    def test_parse_to_file_writes_file(self, golden_dsl: dict, tmp_path: Path) -> None:
        out = DSLParser().parse_to_file(golden_dsl, tmp_path / "gen.py")
        assert out.exists()
        ast.parse(out.read_text(encoding="utf-8"))

    def test_countdown_event_generates_time_countdown(self) -> None:
        dsl = _minimal_dsl()
        code = DSLParser().parse(dsl)
        assert "time_countdown_event(3" in code

    def test_zero_crossing_event_generates_physics_event_terminal(self) -> None:
        dsl = _minimal_dsl()
        dsl["segments"][0]["end_event"] = {
            "expression": "y - (-5)",
            "direction": -1,
            "transition": {"vy": "-vy"},
        }
        code = DSLParser().parse(dsl)
        assert "PhysicsEvent.terminal(" in code
        assert "StateTransition.from_equations(" in code
        assert "direction=-1" in code

    def test_identity_transition_omits_transition_arg(self) -> None:
        dsl = _minimal_dsl()
        dsl["segments"][0]["end_event"] = {
            "expression": "x - 5",
            "direction": 1,
            "transition": {},
        }
        code = DSLParser().parse(dsl)
        assert "transition=" not in code.split("PhysicsEvent.terminal(")[1].split(")")[0]

    def test_point_particle_uses_PointParticle(self) -> None:
        dsl = _minimal_dsl()
        code = DSLParser().parse(dsl)
        assert "PointParticle(" in code

    def test_object2d_uses_object2d_factory(self) -> None:
        dsl = _minimal_dsl()
        dsl["objects"].append(
            {
                "id": "track",
                "type": "object2d",
                "states": ["ax", "ay", "bx", "by"],
                "cartesian_position": [["ax", "ay"], ["bx", "by"]],
                "geometry": "straight_track",
                "geometry_params": {"start": [0, 0, 0], "end": [2, 0, 0]},
            }
        )
        dsl["initial_states"]["track"] = {"ax": 0.0, "ay": 0.0, "bx": 2.0, "by": 0.0}
        code = DSLParser().parse(dsl)
        assert "object2d(" in code
        assert "StraightTrack(" in code

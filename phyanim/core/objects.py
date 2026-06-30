from __future__ import annotations

from dataclasses import dataclass, field

from phyanim.core.state import Parameter, StateVariable
from phyanim.core.validation import normalize_numeric_mapping, require_identifier, require_identifiers
from phyanim.core.enhance.visual_binding import VisualBinding
from manim import Mobject, Circle


@dataclass
class PhysicObject2D:
    """Physical body with parameters, state schema, and render metadata."""

    object_id: str
    parameters: dict[str, Parameter] = field(default_factory=dict)
    state_variables: dict[str, StateVariable] = field(default_factory=dict)

    cartesian_position: list[tuple[str, str]] = field(default_factory=list)

    # 持有一个manim的mobject，用于渲染
    mobject: Mobject | None = None

    # Visual bindings: 状态变量 → 视觉属性（颜色/透明度/缩放/旋转等）
    visual_bindings: list[VisualBinding] = field(default_factory=list)

    # Visual bindings: state variables → visual attributes (color, opacity, etc.)
    visual_bindings: list["VisualBinding"] = field(default_factory=list)

    def __post_init__(self) -> None:
        require_identifier(self.object_id, kind="object_id")
        require_identifiers(self.parameters, kind=f"Object '{self.object_id}' parameter")
        require_identifiers(self.state_variables, kind=f"Object '{self.object_id}' state")
        duplicate_symbols = set(self.parameters) & set(self.state_variables)
        if duplicate_symbols:
            raise ValueError(
                f"Object '{self.object_id}' has duplicate parameter/state symbols: "
                f"{sorted(duplicate_symbols)}"
            )
        mismatched_parameters = [
            name for name, parameter in self.parameters.items()
            if parameter.name != name
        ]
        if mismatched_parameters:
            raise ValueError(
                f"Object '{self.object_id}' parameter keys must match Parameter.name: "
                f"{mismatched_parameters}"
            )
        mismatched_states = [
            name for name, variable in self.state_variables.items()
            if variable.name != name
        ]
        if mismatched_states:
            raise ValueError(
                f"Object '{self.object_id}' state keys must match StateVariable.name: "
                f"{mismatched_states}"
            )
        normalized_positions: list[tuple[str, str]] = []
        for position in self.cartesian_position:
            if position is None:
                continue
            if len(position) != 2:
                raise ValueError(
                    f"Object '{self.object_id}' cartesian position must be (x, y), got {position!r}."
                )
            normalized_positions.append((position[0], position[1]))
        self.cartesian_position = normalized_positions

    def parameter_values(self) -> dict[str, float]:
        return {name: parameter.require_value() for name, parameter in self.parameters.items()}

    def initial_state(self, **values: float | int | str) -> dict[str, float]:
        missing = set(self.state_variables) - set(values)
        unknown = set(values) - set(self.state_variables)
        if missing:
            raise ValueError(f"Object '{self.object_id}' missing initial states: {sorted(missing)}")
        if unknown:
            raise ValueError(f"Object '{self.object_id}' has unknown initial states: {sorted(unknown)}")
        return normalize_numeric_mapping(values, kind=f"Object '{self.object_id}' initial state")

    # 该物理对象要参与绘制，所以必须要返回笛卡尔坐标（这个坐标变量可能是它本身的state变量，也可能是全局的derived变量）但总之都得返回x和y到底是什么
    def cartesian_position_variables(self) -> list[tuple[str, str]]:
        return self.cartesian_position

@dataclass
class PointParticle(PhysicObject2D):
    """A two-dimensional point particle with x, y, vx, vy states."""
    """质点+点电荷模型"""

    def __init__(
        self,
        object_id: str,
        mass: float,
        charge: float | None = None,
        radius: float = 0.01,
        color: str = "red",
        parameter_names: dict[str, str] | None = None,
        state_names: dict[str, str] | None = None,
        cartesian_position: tuple[str, str] | None = None,
        mobject: Mobject | None = None,
    ) -> None:
        parameter_names = parameter_names or {}
        self.state_names = state_names or {
            "x": "x",
            "y": "y",
            "vx": "vx",
            "vy": "vy",
        }

        mass_name = parameter_names.get("mass", "m")
        charge_name = parameter_names.get("charge", "q")
        parameters = {
            mass_name: Parameter(mass_name, mass, "kg", "mass"),
        }
        if charge is not None:
            parameters[charge_name] = Parameter(charge_name, charge, "C", "charge")

        units = {
            "x": "m",
            "y": "m",
            "vx": "m/s",
            "vy": "m/s",
        }
        states = {
            name: StateVariable(name, units.get(kind, None), f"{kind} state")
            for kind, name in self.state_names.items()
        }
        if cartesian_position is None:
            cartesian_position = (
                self.state_names.get("x", "x"),
                self.state_names.get("y", "y"),
            )

        super().__init__(
            object_id=object_id,
            parameters=parameters,
            state_variables=states,
            cartesian_position=[cartesian_position],
            mobject=mobject or Circle(color=color, radius=radius),
        )


def object2d(
    object_id: str,
    *,
    states: dict[str, str | StateVariable],
    parameters: dict[str, float | int | Parameter] | None = None,
    cartesian_position: list[tuple[str, str]] | tuple[str, str] | None = None,
    mobject: Mobject | None = None,
) -> PhysicObject2D:
    """Convenience factory for custom 2D objects."""

    state_variables = {
        name: variable if isinstance(variable, StateVariable) else StateVariable(name, description=variable)
        for name, variable in states.items()
    }
    object_parameters: dict[str, Parameter] = {}
    for name, value in (parameters or {}).items():
        object_parameters[name] = value if isinstance(value, Parameter) else Parameter(name, float(value))
    if cartesian_position is None:
        positions: list[tuple[str, str]] = []
    elif isinstance(cartesian_position, tuple):
        positions = [cartesian_position]
    else:
        positions = list(cartesian_position)
    return PhysicObject2D(
        object_id=object_id,
        parameters=object_parameters,
        state_variables=state_variables,
        cartesian_position=positions,
        mobject=mobject,
    )

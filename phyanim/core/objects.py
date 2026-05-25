from __future__ import annotations

from ast import Tuple
from dataclasses import dataclass, field

from phyanim.core.state import Parameter, StateVariable
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

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ValueError("object_id cannot be empty.")
        invalid_parameters = [name for name in self.parameters if not name.isidentifier()]
        if invalid_parameters:
            raise ValueError(f"Object '{self.object_id}' has invalid parameter names: {invalid_parameters}")
        invalid_states = [name for name in self.state_variables if not name.isidentifier()]
        if invalid_states:
            raise ValueError(f"Object '{self.object_id}' has invalid state names: {invalid_states}")
        duplicate_symbols = set(self.parameters) & set(self.state_variables)
        if duplicate_symbols:
            raise ValueError(
                f"Object '{self.object_id}' has duplicate parameter/state symbols: "
                f"{sorted(duplicate_symbols)}"
            )

    def parameter_values(self) -> dict[str, float]:
        return {name: parameter.require_value() for name, parameter in self.parameters.items()}

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
        cartesian_position: Tuple[str, str] | None = None,
    ) -> None:
        parameter_names = parameter_names or {}
        self.state_names = state_names or {}

        mass_name = parameter_names.get("mass", "m")
        charge_name = parameter_names.get("charge", "q")
        parameters = {
            mass_name: Parameter(mass_name, mass, "kg", "mass"),
        }
        if charge is not None:
            parameters[charge_name] = Parameter(charge_name, charge, "C", "charge")

        self.states = {}
        for key, value in self.state_names.items():
            self.states[value] = StateVariable(value, "暂无单位", f"{key} state")

        super().__init__(
            object_id=object_id,
            parameters=parameters,
            state_variables=self.states,
            cartesian_position=[(cartesian_position)],
            mobject=Circle(color=color, radius=radius),
        )

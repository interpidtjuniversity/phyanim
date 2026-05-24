from __future__ import annotations

from dataclasses import dataclass, field

from phyanim.core.state import Parameter, StateVariable
from manim import Mobject, Circle

@dataclass
class PhysicObject2D:
    """Physical body with parameters, state schema, and render metadata."""

    object_id: str
    parameters: dict[str, Parameter] = field(default_factory=dict)
    state_variables: dict[str, StateVariable] = field(default_factory=dict)

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
    def cartesian_position_variables(self) -> dict[str, str]:
        raise NotImplementedError("Subclasses must implement this method.")

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
        if "x" in state_names:
            self.states[self.state_names["x"]] = StateVariable(self.state_names["x"], "m", "horizontal position")
        if "y" in state_names:
            self.states[self.state_names["y"]] = StateVariable(self.state_names["y"], "m", "vertical position")
        if "vx" in state_names:
            self.states[self.state_names["vx"]] = StateVariable(self.state_names["vx"], "m/s", "horizontal velocity")
        if "vy" in state_names:
            self.states[self.state_names["vy"]] = StateVariable(self.state_names["vy"], "m/s", "vertical velocity")
        if "ax" in state_names:
            self.states[self.state_names["ax"]] = StateVariable(self.state_names["ax"], "m/s^2", "horizontal acceleration")
        if "ay" in state_names:
            self.states[self.state_names["ay"]] = StateVariable(self.state_names["ay"], "m/s^2", "vertical acceleration")

        circle = Circle(color=color, radius=radius)
        self.mobject = circle
        super().__init__(object_id, parameters, self.states, circle)
    
    def cartesian_position_variables(self) -> tuple[str, str]:
        x_name = self.state_names.get("x")
        y_name = self.state_names.get("y")

        if x_name is None or y_name is None:
            raise ValueError(
                f"PointParticle '{self.object_id}' must define x and y state names."
            )

        return x_name, y_name




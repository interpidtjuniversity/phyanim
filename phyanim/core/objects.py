from __future__ import annotations

from dataclasses import dataclass, field

from phyanim.core.state import Parameter, StateVariable


@dataclass
class PhysicObject:
    """Physical body with parameters, state schema, and render metadata."""

    object_id: str
    parameters: dict[str, Parameter] = field(default_factory=dict)
    state_variables: dict[str, StateVariable] = field(default_factory=dict)
    render_model: dict[str, object] = field(default_factory=dict)

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


@dataclass
class PointParticle(PhysicObject):
    """A two-dimensional point particle with x, y, vx, vy states."""
    """质点+点电荷模型"""

    def __init__(
        self,
        object_id: str,
        mass: float,
        charge: float | None = None,
        radius: float = 0.06,
        color: str = "yellow",
        parameter_names: dict[str, str] | None = None,
        state_names: dict[str, str] | None = None,
    ) -> None:
        parameter_names = parameter_names or {}
        state_names = state_names or {}

        mass_name = parameter_names.get("mass", "m")
        charge_name = parameter_names.get("charge", "q")
        x_name = state_names.get("x", "x")
        y_name = state_names.get("y", "y")
        vx_name = state_names.get("vx", "vx")
        vy_name = state_names.get("vy", "vy")

        parameters = {
            mass_name: Parameter(mass_name, mass, "kg", "mass"),
        }
        if charge is not None:
            parameters[charge_name] = Parameter(charge_name, charge, "C", "charge")

        states = {
            x_name: StateVariable(x_name, "m", "horizontal position"),
            y_name: StateVariable(y_name, "m", "vertical position"),
            vx_name: StateVariable(vx_name, "m/s", "horizontal velocity"),
            vy_name: StateVariable(vy_name, "m/s", "vertical velocity"),
        }
        render_model = {"geometry": "circle", "radius": radius, "color": color}
        super().__init__(object_id, parameters, states, render_model)

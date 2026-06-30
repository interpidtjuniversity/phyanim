"""Visual bindings: map physics state variables to visual properties.

A :class:`VisualBinding` declares how a state variable (or a derived
expression) controls a mobject's visual attribute — color, opacity,
stroke width, scale, or rotation — in addition to the positional mapping
provided by :attr:`PhysicObject2D.cartesian_position`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Attributes that can be controlled via visual bindings.
VISUAL_ATTRIBUTES = ("color", "opacity", "stroke_width", "scale", "rotation")

# manim color names accepted in DSL expressions.  When the expression
# evaluates to one of these strings (as a sympy Symbol), the binding
# layer converts it to the corresponding manim color.
_COLOR_NAMES = (
    "RED", "GREEN", "BLUE", "YELLOW", "WHITE", "BLACK", "ORANGE",
    "PURPLE", "PINK", "TEAL", "GOLD", "MAROON", "GRAY", "GREY",
    "BLUE_A", "BLUE_B", "BLUE_C", "BLUE_D", "BLUE_E",
    "RED_A", "RED_B", "RED_C", "RED_D", "RED_E",
    "GREEN_A", "GREEN_B", "GREEN_C", "GREEN_D", "GREEN_E",
    "TEAL_A", "TEAL_B", "TEAL_C", "TEAL_D", "TEAL_E",
    "PURPLE_A", "PURPLE_B", "PURPLE_C", "PURPLE_D", "PURPLE_E",
    "GOLD_A", "GOLD_B", "GOLD_C", "GOLD_D", "GOLD_E",
)


@dataclass
class VisualBinding:
    """Declaration of a state→visual-attribute mapping.

    Parameters
    ----------
    attribute:
        One of :data:`VISUAL_ATTRIBUTES`.
    variables:
        State / derived variable names referenced by *expression*.
    expression:
        A SymPy expression string.  For ``color``, the expression should
        evaluate to a color name string (e.g. via ``Piecewise``); for
        numeric attributes, a float.
    center:
        ``(x, y)`` rotation center for ``rotation`` bindings.
        If ``None``, the mobject's center is used.
    """

    attribute: str
    variables: list[str]
    expression: str
    center: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.attribute not in VISUAL_ATTRIBUTES:
            raise ValueError(
                f"VisualBinding attribute must be one of {VISUAL_ATTRIBUTES}, "
                f"got '{self.attribute}'."
            )
        if not self.variables:
            raise ValueError("VisualBinding requires at least one variable.")
        if not self.expression:
            raise ValueError("VisualBinding requires an expression.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VisualBinding":
        """Construct from a DSL dict."""
        return cls(
            attribute=data["attribute"],
            variables=list(data["variables"]),
            expression=data["expression"],
            center=tuple(data["center"]) if data.get("center") else None,
        )

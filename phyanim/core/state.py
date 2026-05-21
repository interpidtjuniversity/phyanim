from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Parameter:
    """A value that is constant within a physics segment."""

    name: str
    value: float | None = None
    unit: str | None = None
    description: str | None = None

    def require_value(self) -> float:
        if self.value is None:
            raise ValueError(f"Parameter '{self.name}' has no numeric value.")
        return self.value


@dataclass(frozen=True)
class StateVariable:
    """A minimal dynamic degree of freedom solved as a function of time."""

    name: str
    unit: str | None = None
    description: str | None = None

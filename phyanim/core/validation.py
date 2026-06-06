from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterable, Mapping


def require_identifier(name: str, *, kind: str = "symbol") -> None:
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError(f"{kind} '{name}' must be a valid Python identifier.")


def require_identifiers(names: Iterable[str], *, kind: str = "symbol") -> None:
    invalid = [name for name in names if not isinstance(name, str) or not name.isidentifier()]
    if invalid:
        raise ValueError(f"{kind} names must be valid Python identifiers: {invalid}")


def require_unique(names: Iterable[str], *, kind: str = "symbol") -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for name in names:
        if name in seen:
            duplicates.add(name)
        seen.add(name)
    if duplicates:
        raise ValueError(f"{kind} names must be unique: {sorted(duplicates)}")


def normalize_numeric_mapping(values: Mapping[str, float | int | str], *, kind: str) -> dict[str, float]:
    require_identifiers(values, kind=kind)
    normalized: dict[str, float] = {}
    for name, value in values.items():
        try:
            normalized[name] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{kind} '{name}' must have a numeric value, got {value!r}.") from exc
    return normalized


@dataclass
class SymbolRegistry:
    """Tracks model-level symbols so states and parameters stay explicit."""

    symbols: dict[str, str] = field(default_factory=dict)

    def reserve_many(self, names: Iterable[str], *, owner: str, kind: str) -> None:
        names = list(names)
        require_identifiers(names, kind=kind)
        require_unique(names, kind=kind)
        conflicts = {
            name: self.symbols[name]
            for name in names
            if name in self.symbols
        }
        if conflicts:
            details = ", ".join(
                f"{name} already used by {existing_owner}"
                for name, existing_owner in sorted(conflicts.items())
            )
            raise ValueError(f"{owner} cannot reuse {kind} symbols: {details}")
        for name in names:
            self.symbols[name] = owner

    def reserve_mapping(self, values: Mapping[str, object], *, owner: str, kind: str) -> None:
        self.reserve_many(values.keys(), owner=owner, kind=kind)

    @classmethod
    def from_global_parameters(cls, parameters: Mapping[str, object]) -> "SymbolRegistry":
        registry = cls()
        registry.reserve_mapping(parameters, owner="global parameters", kind="parameter")
        return registry

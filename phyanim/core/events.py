from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from phyanim.core.expressions import CompiledExpression, SympyExpressionCompiler
from phyanim.core.validation import require_identifier, require_identifiers


State = dict[str, float]
Parameters = dict[str, float]


class EventFunction(Protocol):
    def __call__(self, time: float, state: State, parameters: Parameters) -> float:
        ...


class TransitionFunction(Protocol):
    def __call__(self, state: State, parameters: Parameters) -> State:
        ...


@dataclass
class EventCondition:
    """Scalar root function used to terminate or mark a segment."""

    function: EventFunction | None = None
    expression: str | None = None
    terminal: bool = True
    direction: int = 0

    @classmethod
    def from_expression(
        cls,
        expression: str,
        *,
        terminal: bool = True,
        direction: int = 0,
    ) -> "EventCondition":
        return cls(expression=expression, terminal=terminal, direction=direction)

    def __post_init__(self) -> None:
        if self.function is None and self.expression is None:
            raise ValueError("EventCondition requires either function or SymPy expression.")
        if self.function is not None and self.expression is not None:
            raise ValueError("EventCondition cannot define both function and expression.")
        if self.direction not in {-1, 0, 1}:
            raise ValueError("EventCondition.direction must be -1, 0, or 1.")

    def build_function(self, symbol_names: set[str] | None = None) -> EventFunction:
        if self.function is not None:
            return self.function
        assert self.expression is not None
        compiled = SympyExpressionCompiler().compile(self.expression, symbol_names=symbol_names)

        def evaluate(time: float, state: State, parameters: Parameters) -> float:
            return compiled.evaluate(state, parameters, time)

        self.function = evaluate
        return evaluate

@dataclass
class StateTransition:
    """State update applied after an event, such as collision impulse."""

    name: str
    function: TransitionFunction | None = None
    # 状态跃迁的方程应该是个方程组
    equations: dict[str, str] | None = None

    @classmethod
    def identity(cls, name: str = "identity_transition") -> "StateTransition":
        return cls(name=name, equations={})

    @classmethod
    def from_equations(cls, name: str, equations: dict[str, str]) -> "StateTransition":
        return cls(name=name, equations=dict(equations))

    def __post_init__(self) -> None:
        require_identifier(self.name, kind="transition name")
        if self.function is None and self.equations is None:
            raise ValueError("StateTransition requires either function or SymPy equations.")
        if self.function is not None and self.equations is not None:
            raise ValueError("StateTransition cannot define both function and equations.")
        if self.equations is not None:
            require_identifiers(self.equations, kind=f"Transition '{self.name}' equation")

    def apply(self, state: State, parameters: Parameters, time: float = 0.0) -> State:
        if self.function is not None:
            return self.function(dict(state), dict(parameters))
        assert self.equations is not None
        compiler = SympyExpressionCompiler()
        symbol_names = set(state) | set(parameters) | {"t"}
        compiled: dict[str, CompiledExpression] = compiler.compile_mapping(
            self.equations,
            symbol_names=symbol_names,
        )
        updated = dict(state)
        # 状态跃迁必须基于事件发生前同一个快照计算，不可以原地更新状态
        source_state = dict(state)
        for name, expression in compiled.items():
            updated[name] = expression.evaluate(source_state, parameters, time)
        return updated


@dataclass
class PhysicsEvent:
    name: str
    # 零点检测，这里应该是一个表达式，如果符号跳变则说明事件发生
    condition: EventCondition
    transition: StateTransition | None = None

    @classmethod
    def terminal(
        cls,
        name: str,
        expression: str,
        *,
        direction: int = 0,
        transition: StateTransition | dict[str, str] | None = None,
    ) -> "PhysicsEvent":
        if transition is None:
            state_transition = StateTransition.identity(f"{name}_transition")
        elif isinstance(transition, dict):
            state_transition = StateTransition.from_equations(f"{name}_transition", transition)
        else:
            state_transition = transition
        return cls(
            name=name,
            condition=EventCondition.from_expression(expression, terminal=True, direction=direction),
            transition=state_transition,
        )

    def __post_init__(self) -> None:
        require_identifier(self.name, kind="event name")
        if self.transition is not None and not self.condition.terminal:
            raise ValueError(
                f"Event '{self.name}' has a transition, so its condition must be terminal."
            )


def time_end_event(name: str = "segment_time_end") -> PhysicsEvent:
    return PhysicsEvent.terminal(
        name,
        "t - t_end",
        direction=1,
        transition=StateTransition.identity(f"{name}_transition"),
    )

def time_countdown_event(countdown: int, name: str = "segment_time_countdown") -> PhysicsEvent:
    return PhysicsEvent.terminal(
        name,
        f"t - t_start - {countdown}",
        direction=1,
        transition=StateTransition.identity(f"{name}_transition"),
    )

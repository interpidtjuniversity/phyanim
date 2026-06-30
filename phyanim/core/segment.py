from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from phyanim.core.events import Parameters, State
from phyanim.core.expressions import CompiledExpression, SympyExpressionCompiler
from phyanim.core.validation import require_identifier, require_identifiers, require_unique

DerivativeFunction = Callable[[float, State, Parameters], State]
DerivedFunction = Callable[[float, State, Parameters], float]


@dataclass
class PhysicsSegment:
    """一个连续物理阶段；时长必须大于0，在该阶段内部，所有物体的所有状态变量按照微分方程演化。"""

    segment_id: str
    # 物体ID列表
    object_ids: list[str]
    # 状态变量列表
    state_vector: list[str]
    # 每个状态变量都有一个在所有物体的唯一名称
    state_owners: dict[str, str] | None = None
    # 导数方程组
    derivative: DerivativeFunction | None = None
    # 每个状态变量都有一个导数函数，key是状态变量名称（代表该变量的导数），value是导数函数
    equations: dict[str, str] | None = None

    # 这个段内部的参数
    parameters: dict[str, float] = field(default_factory=dict)

    # 派生量
    derived_quantities: dict[str, DerivedFunction] = field(default_factory=dict)
    derived_equations: dict[str, str] = field(default_factory=dict)

    # kinematic 模式：闭式解表达式（不走 ODE 积分）
    kinematic_expressions: dict[str, str] | None = None
    # kinematic 模式：采样点数组
    kinematic_samples: dict[str, dict] | None = None

    # 阶段持续时长，这个时长不具有一般意义，因为最终阶段的结束由该阶段后接的事件决定
    duration: float = 1.0

    @classmethod
    def from_equations(
        cls,
        segment_id: str,
        *,
        objects: list[str],
        equations: dict[str, str],
        duration: float,
        owners: dict[str, str] | None = None,
        parameters: dict[str, float] | None = None,
        derived: dict[str, str] | None = None,
    ) -> "PhysicsSegment":
        """Build a segment from symbolic ODE equations.

        The order of the equation mapping is used as the solver state order.
        """

        return cls(
            segment_id=segment_id,
            object_ids=list(objects),
            state_vector=list(equations),
            state_owners=owners,
            equations=dict(equations),
            parameters=dict(parameters or {}),
            derived_equations=dict(derived or {}),
            duration=duration,
        )

    @classmethod
    def from_derivative(
        cls,
        segment_id: str,
        *,
        objects: list[str],
        state_vector: list[str],
        derivative: DerivativeFunction,
        duration: float,
        owners: dict[str, str] | None = None,
        parameters: dict[str, float] | None = None,
        derived_quantities: dict[str, DerivedFunction] | None = None,
    ) -> "PhysicsSegment":
        return cls(
            segment_id=segment_id,
            object_ids=list(objects),
            state_vector=list(state_vector),
            state_owners=owners,
            derivative=derivative,
            parameters=dict(parameters or {}),
            derived_quantities=dict(derived_quantities or {}),
            duration=duration,
        )

    @classmethod
    def from_kinematic(
        cls,
        segment_id: str,
        *,
        objects: list[str],
        state_vector: list[str],
        expressions: dict[str, str],
        duration: float,
        owners: dict[str, str] | None = None,
        parameters: dict[str, float] | None = None,
        derived: dict[str, str] | None = None,
    ) -> "PhysicsSegment":
        """Build a segment from closed-form kinematic expressions.

        Each expression maps a state variable name to a SymPy expression
        of ``t`` (and optionally parameters).  The solver evaluates these
        expressions directly at sample times instead of integrating an ODE.
        """
        return cls(
            segment_id=segment_id,
            object_ids=list(objects),
            state_vector=list(state_vector),
            state_owners=owners,
            kinematic_expressions=dict(expressions),
            parameters=dict(parameters or {}),
            derived_equations=dict(derived or {}),
            duration=duration,
        )

    @classmethod
    def from_samples(
        cls,
        segment_id: str,
        *,
        objects: list[str],
        state_vector: list[str],
        samples: dict[str, dict],
        duration: float,
        owners: dict[str, str] | None = None,
        parameters: dict[str, float] | None = None,
        derived: dict[str, str] | None = None,
    ) -> "PhysicsSegment":
        """Build a segment from pre-computed sample arrays.

        ``samples`` is ``{state_name: {"times": [...], "values": [...]}}``.
        The solver linearly interpolates between sample points.
        """
        return cls(
            segment_id=segment_id,
            object_ids=list(objects),
            state_vector=list(state_vector),
            state_owners=owners,
            kinematic_samples=dict(samples),
            parameters=dict(parameters or {}),
            derived_equations=dict(derived or {}),
            duration=duration,
        )

    def __post_init__(self) -> None:
        require_identifier(self.segment_id, kind="segment_id")
        if not self.object_ids:
            raise ValueError(f"Segment '{self.segment_id}' object_ids cannot be empty.")
        require_identifiers(self.object_ids, kind=f"Segment '{self.segment_id}' object")
        require_identifiers(self.state_vector, kind=f"Segment '{self.segment_id}' state")
        require_unique(self.state_vector, kind=f"Segment '{self.segment_id}' state")
        require_identifiers(self.parameters, kind=f"Segment '{self.segment_id}' parameter")
        require_identifiers(self.derived_quantities, kind=f"Segment '{self.segment_id}' derived quantity")
        require_identifiers(self.derived_equations, kind=f"Segment '{self.segment_id}' derived quantity")
        if self.state_owners is None:
            if len(self.object_ids) != 1:
                raise ValueError(
                    f"Segment '{self.segment_id}' with multiple objects requires explicit state_owners."
                )
            else:
                self.state_owners = {name: self.object_ids[0] for name in self.state_vector}

        # 部分状态变量没有指定所有者，报错
        extra_owners = set(self.state_owners) - set(self.state_vector)
        if extra_owners:
            raise ValueError(f"Segment '{self.segment_id}' has state owners for unknown states: {sorted(extra_owners)}")
        missing_owners = [name for name in self.state_vector if name not in self.state_owners]
        if missing_owners:
            raise ValueError(f"Segment '{self.segment_id}' missing state owners: {missing_owners}")
        unknown_owners = set(self.state_owners.values()) - set(self.object_ids)
        if unknown_owners:
            raise ValueError(f"Segment '{self.segment_id}' has unknown state owners: {sorted(unknown_owners)}")
        if self.derivative is None and self.equations is None and self.kinematic_expressions is None and self.kinematic_samples is None:
            raise ValueError(
                f"Segment '{self.segment_id}' requires either callable derivative, SymPy equations, kinematic expressions, or kinematic samples."
            )
        # kinematic 模式的校验
        if self.kinematic_expressions is not None:
            require_identifiers(self.kinematic_expressions, kind=f"Segment '{self.segment_id}' kinematic expression")
            missing = [name for name in self.state_vector if name not in self.kinematic_expressions]
            if missing:
                raise ValueError(f"Segment '{self.segment_id}' missing kinematic expressions for states: {missing}")
            extra = sorted(set(self.kinematic_expressions) - set(self.state_vector))
            if extra:
                raise ValueError(f"Segment '{self.segment_id}' has kinematic expressions for unknown states: {extra}")
        if self.kinematic_samples is not None:
            missing = [name for name in self.state_vector if name not in self.kinematic_samples]
            if missing:
                raise ValueError(f"Segment '{self.segment_id}' missing kinematic samples for states: {missing}")
        # kinematic 模式与 ODE 模式互斥
        mode_count = sum(1 for x in [self.derivative, self.equations, self.kinematic_expressions, self.kinematic_samples] if x is not None)
        if mode_count > 1:
            raise ValueError(
                f"Segment '{self.segment_id}' cannot define more than one of: derivative, equations, kinematic_expressions, kinematic_samples."
            )
        if self.equations is not None:
            require_identifiers(self.equations, kind=f"Segment '{self.segment_id}' equation")
            missing = [name for name in self.state_vector if name not in self.equations]
            if missing:
                raise ValueError(f"Segment '{self.segment_id}' missing equations for states: {missing}")
            extra = sorted(set(self.equations) - set(self.state_vector))
            if extra:
                raise ValueError(f"Segment '{self.segment_id}' has equations for unknown states: {extra}")
        if self.duration <= 0:
            raise ValueError(f"Segment '{self.segment_id}' duration must be positive.")

    @property
    def object_set(self) -> set[str]:
        return set(self.object_ids)

    @property
    def state_set(self) -> set[str]:
        return set(self.state_vector)

    def object_state_variables(self, object_id: str) -> list[str]:
        assert self.state_owners is not None
        return [name for name in self.state_vector if self.state_owners[name] == object_id]

    # 该段的起始关键帧，需要给该阶段赋值
    def flatten_keyframe_state(self, keyframe_states: dict[str, dict[str, float]]) -> State:
        state: State = {}
        for object_id in self.object_ids or []:
            try:
                object_state = keyframe_states[object_id]
            except KeyError as exc:
                raise KeyError(f"Keyframe has no state for object '{object_id}'.") from exc
            for name in self.object_state_variables(object_id):
                if name not in object_state:
                    # 关键帧赋值应该要完全覆盖这个阶段的这个物体的所有state变量
                    raise ValueError(
                        f"Segment '{self.segment_id}' missing state '{name}' for object '{object_id}'."
                    )
                state[name] = object_state[name]
        return state

    def split_state_by_object(self, flat_state: State) -> dict[str, dict[str, float]]:
        object_states = {object_id: {} for object_id in self.object_ids or []}
        assert self.state_owners is not None
        for name in self.state_vector:
            object_states[self.state_owners[name]][name] = flat_state[name]
        return object_states

    # 合并全局参数和阶段参数
    def merged_parameters(self, global_parameters: dict[str, float]) -> dict[str, float]:
        merged = dict(global_parameters)
        merged.update(self.parameters)
        return merged

    def build_derivative(self, parameters: Parameters) -> DerivativeFunction:
        if self.derivative is not None:
            return self.derivative
        assert self.equations is not None
        # 每个变量都必须有导数函数
        missing = [name for name in self.state_vector if name not in self.equations]
        if missing:
            raise ValueError(f"Segment '{self.segment_id}' missing equations for states: {missing}")

        compiler = SympyExpressionCompiler()
        symbol_names = set(self.state_vector) | set(parameters) | {"t"}
        compiled = compiler.compile_mapping(self.equations, symbol_names=symbol_names)

        def evaluate(time: float, state: State, parameters: Parameters) -> State:
            # 这里保持状态向量顺序稳定，SciPy 的 y 向量才能和物理状态一一对应。
            return {
                name: compiled[name].evaluate(state, parameters, time)
                for name in self.state_vector
            }

        return evaluate

    def build_derived_quantities(self, parameters: Parameters) -> dict[str, DerivedFunction]:
        derived = dict(self.derived_quantities)
        if not self.derived_equations:
            return derived

        compiler = SympyExpressionCompiler()
        symbol_names = set(self.state_vector) | set(parameters) | {"t"}
        compiled: dict[str, CompiledExpression] = compiler.compile_mapping(
            self.derived_equations,
            symbol_names=symbol_names,
        )
        for name, expression in compiled.items():
            def make_function(current_expression: CompiledExpression) -> DerivedFunction:
                
                def evaluate(time: float, state: State, parameters: Parameters) -> float:
                    return current_expression.evaluate(state, parameters, time)

                return evaluate

            derived[name] = make_function(expression)
        return derived

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

State = dict[str, float]
Parameters = dict[str, float]


@dataclass(frozen=True)
class CompiledExpression:
    """用 SymPy 编译出的纯数值表达式。"""

    expression: str
    names: tuple[str, ...]
    function: Callable[..., float]

    def evaluate(self, state: State, parameters: Parameters, time: float) -> float:
        values = []
        for name in self.names:
            if name == "t":
                values.append(time)
            elif name in state:
                values.append(state[name])
            elif name in parameters:
                values.append(parameters[name])
            else:
                raise KeyError(f"Expression '{self.expression}' references unknown symbol '{name}'.")
        return float(self.function(*values))


class SympyExpressionCompiler:
    """把字符串公式编译成 SciPy 可调用的数值函数。"""

    def compile(
        self,
        expression: str,
        symbol_names: set[str] | None = None,
    ) -> CompiledExpression:
        import sympy as sp

        locals_map = None
        if symbol_names is not None:
            invalid_names = sorted(name for name in symbol_names if not name.isidentifier())
            if invalid_names:
                raise ValueError(
                    "SymPy expressions only support plain identifier symbols: "
                    f"{invalid_names}"
                )
            locals_map = {
                name: sp.Symbol(name)
                for name in symbol_names
            }
        
        # 显式声明物理符号，避免 SymPy 把 E/I 等变量解释成内置常数。
        # 符号命名冲突由建模层保证，这里不再做任何临时别名转换。
        parsed = sp.sympify(expression, locals=locals_map)
        names = tuple(sorted(str(symbol) for symbol in parsed.free_symbols))
        symbols = [sp.Symbol(name) for name in names]
        function = sp.lambdify(symbols, parsed, modules="numpy")
        return CompiledExpression(expression=expression, names=names, function=function)

    def compile_mapping(
        self,
        expressions: dict[str, str],
        symbol_names: set[str] | None = None,
    ) -> dict[str, CompiledExpression]:
        return {
            name: self.compile(expression, symbol_names=symbol_names)
            for name, expression in expressions.items()
        }

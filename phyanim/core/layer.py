
from dataclasses import dataclass
from typing import Callable

from phyanim.core.context import Context

@dataclass
class Layer:
    id: str
    contexts: list[Context]
    time_mapping_func: Callable[[float], float]
    total_time: float

    def __post_init__(self):
        for ctx in self.contexts:
            ctx.set_time_mapping_to_physics(self.time_mapping_func)

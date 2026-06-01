
# 物理层Layer就是main layer，所有层的本征时间就是物理层的时间
from phyanim.core.enhance.trigger import Trigger

# 时长拉伸器，暂时先只支持线性映射
class TimeWrapper:
    def __init__(self, id: str, type: str, trigger: Trigger, advance: float = None,  delay: float = None, extend_to: float = None, speed: float = 0.5):
        self.id = id

        # slow or freeze
        self.type = type
        self.trigger = trigger
        self.advance = advance
        self.delay = delay
        self.speed = speed
        self.extend_to = extend_to


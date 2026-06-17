import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phyanim.llm import LLMConfig, make_client, PhysicsLLMPlanner, DSLParser, generate_and_render

config = LLMConfig(
    api_key="sk-77e1188b130f4187b2c66ec583a75616",
    base_url="https://api.deepseek.com",
    model="deepseek-v4-pro",
)
planner = PhysicsLLMPlanner(make_client(config))

dsl = planner.plan("两个小球通过弹簧碰撞，参数你自行设定")
code = DSLParser().parse(dsl)      # → 可执行 .py 字符串
video = generate_and_render(dsl)   # 可选：直接出视频
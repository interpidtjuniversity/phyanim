import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phyanim.llm import LLMConfig, make_client, PhysicsLLMPlanner, render_code

config = LLMConfig(
    api_key="sk-77e1188b130f4187b2c66ec583a75616",
    base_url="https://api.deepseek.com",
    model="deepseek-v4-pro",
)
planner = PhysicsLLMPlanner(make_client(config))

code = planner.plan("一个小球从10米高处自由落体，另一个小球从地面向上以10m/s速度竖直上抛")
print(code)                              # → 可执行 .py 字符串
# render_code(code, output_dir="outputs")  # 可选：直接出视频

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phyanim.llm import DeepSeekClient, LLMConfig, PhysicsLLMPlanner


PROBLEMS = [
    """一个带正电粒子从 x=-2 处以速度 v0 水平向右进入 x=0 到 x=3 的匀强磁场区域，
磁场垂直纸面向里。粒子离开磁场后继续匀速运动。请拆分阶段，给出状态变量、方程、
事件条件，并建议显示速度箭头、洛伦兹力箭头和轨迹。""",
    """质量为 m 的小球从高度 h 处水平抛出，落到地面后与地面发生一次恢复系数 e 的碰撞，
随后继续抛体运动直到第二次落地。请把碰撞前后作为不同阶段，用关键帧衔接。""",
    """两辆一维小车在光滑水平轨道上相向运动，质量分别为 m1、m2，发生完全弹性碰撞。
要求先描述碰撞前的匀速阶段，再描述碰撞瞬间的状态跃迁，最后描述碰撞后的匀速阶段。""",
    """带电粒子先经过一段匀强电场加速区，再进入匀强磁场做偏转，最后离开磁场。
请在 IR 中明确电场阶段、磁场阶段和离开后的自由运动阶段，并给出每段的 ODE。""",
]


def main() -> None:
    config = LLMConfig.from_readme_tail("readme.txt")
    output_dir = Path("outputs/llm_tests")
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, problem in enumerate(PROBLEMS, start=1):
        output_path = output_dir / f"problem_{index}.json"
        if output_path.exists():
            print(f"Skipping problem {index}; {output_path} already exists.")
            continue
        print(f"Testing problem {index}...")
        ir = plan_with_readme_config_or_supported_fallback(config, problem)
        output_path.write_text(json.dumps(ir, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  wrote {output_path}")


def plan_with_readme_config_or_supported_fallback(config: LLMConfig, problem: str) -> dict[str, object]:
    planner = PhysicsLLMPlanner(DeepSeekClient(config))
    try:
        return planner.plan(problem)
    except RuntimeError as exc:
        message = str(exc)
        if "deepseek-v4-lite" not in message or "deepseek-v4-flash" not in message:
            raise
        fallback_config = LLMConfig(
            api_key=config.api_key,
            base_url=config.base_url,
            model="deepseek-v4-flash",
            timeout_seconds=config.timeout_seconds,
        )
        print("  README model deepseek-v4-lite is rejected by API; retrying with deepseek-v4-flash.")
        return PhysicsLLMPlanner(DeepSeekClient(fallback_config)).plan(problem)


if __name__ == "__main__":
    main()

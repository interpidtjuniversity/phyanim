"""Optional rendering execution for generated Python code.

Provides convenience helpers to write generated code to a file and execute
it to render a video.  All manim artifacts (videos, tex, text, images,
audio, partial movies) are confined to a single ``media_dir``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def prepare_render_script(
    code: str,
    root_dir: str | Path | None = None,
    script_name: str = "generated_animation.py",
    code_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write generated code and return its path and the manim media path."""
    root = Path(root_dir).resolve() if root_dir else Path.cwd()
    root.mkdir(parents=True, exist_ok=True)

    script_dir = Path(code_dir).resolve() if code_dir else root / "code"
    script_dir.mkdir(parents=True, exist_ok=True)
    script_path = script_dir / script_name
    script_path.write_text(code, encoding="utf-8")

    manim_media = root / "media"
    manim_media.mkdir(parents=True, exist_ok=True)
    return script_path, manim_media

# 保存生成的源文件
def prepare_source_script(
    code: str,
    root_dir: str | Path | None = None,
    script_name: str = "generated_animation.py",
    source_code_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write generated code and return its path and the manim media path."""
    root = Path(root_dir).resolve() if root_dir else Path.cwd()
    root.mkdir(parents=True, exist_ok=True)

    script_dir = Path(source_code_dir).resolve() if source_code_dir else root / "source_code"
    script_dir.mkdir(parents=True, exist_ok=True)
    script_path = script_dir / f"{script_name}"
    script_path.write_text(code, encoding="utf-8")
    
    return script_path

def prepare_analysis_info(
    analysis_info: str,
    root_dir: str | Path | None = None,
    script_name: str = "generated_animation.txt",
    analysis_info_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write generated code and return its path and the manim media path."""
    root = Path(root_dir).resolve() if root_dir else Path.cwd()
    root.mkdir(parents=True, exist_ok=True)

    analysis_info_dir_ = Path(analysis_info_dir).resolve() if analysis_info_dir else root / "analysis_info"
    analysis_info_dir_.mkdir(parents=True, exist_ok=True)
    analysis_info_path = analysis_info_dir_ / f"{script_name}"
    analysis_info_path.write_text(analysis_info, encoding="utf-8")
    
    return analysis_info_path


def build_render_environment(manim_media: str | Path) -> dict[str, str]:
    """Build the environment used by generated render scripts."""
    env = os.environ.copy()
    project_root = Path(__file__).resolve().parents[2]
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{project_root}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(project_root)
    )
    env["MANIM_MEDIA_DIR"] = str(manim_media)
    return env


def start_render_process(
    code: str,
    root_dir: str | Path | None = None,
    script_name: str = "generated_animation.py",
    code_dir: str | Path | None = None,
) -> tuple[Path, subprocess.Popen[str]]:
    """Write generated code and start its renderer without waiting."""
    script_path, manim_media = prepare_render_script(
        code,
        root_dir=root_dir,
        script_name=script_name,
        code_dir=code_dir,
    )
    process = subprocess.Popen(
        [sys.executable, str(script_path)],
        cwd=str(manim_media),
        env=build_render_environment(manim_media),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return script_path, process


def render_code(
    code: str,
    root_dir: str | Path | None = None,
    script_name: str = "generated_animation.py",
    code_dir: str | Path | None = None,
) -> Path:
    """Write *code* to a script and execute it to render a video.

    All manim output goes under ``<root_dir>/media/``. The generated script
    goes under ``<root_dir>/code/`` (or ``code_dir`` if supplied).
    """
    script_path, process = start_render_process(
        code,
        root_dir=root_dir,
        script_name=script_name,
        code_dir=code_dir,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(
            f"Rendering failed (exit code {process.returncode}).\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        )
    return script_path


__all__ = [
    "build_render_environment",
    "prepare_render_script",
    "prepare_analysis_info",
    "render_code",
    "start_render_process",
]

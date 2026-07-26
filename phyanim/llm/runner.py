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
    media_dir: str | Path | None = None,
    script_name: str = "generated_animation.py",
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write generated code and return its path and the manim media path."""
    root = Path(media_dir).resolve() if media_dir else Path.cwd() / "media"
    root.mkdir(parents=True, exist_ok=True)

    script_dir = Path(output_dir).resolve() if output_dir else root / "code"
    script_dir.mkdir(parents=True, exist_ok=True)
    script_path = script_dir / script_name
    script_path.write_text(code, encoding="utf-8")

    manim_media = root / "media"
    manim_media.mkdir(parents=True, exist_ok=True)
    return script_path, manim_media

# 保存生成的源文件
def prepare_source_script(
    code: str,
    media_dir: str | Path | None = None,
    script_name: str = "generated_animation.py",
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write generated code and return its path and the manim media path."""
    root = Path(media_dir).resolve() if media_dir else Path.cwd() / "media"
    root.mkdir(parents=True, exist_ok=True)

    script_dir = Path(output_dir).resolve() if output_dir else root / "source_code"
    script_dir.mkdir(parents=True, exist_ok=True)
    script_path = script_dir / f"{script_name}"
    script_path.write_text(code, encoding="utf-8")
    
    return script_path


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
    media_dir: str | Path | None = None,
    script_name: str = "generated_animation.py",
    output_dir: str | Path | None = None,
) -> tuple[Path, subprocess.Popen[str]]:
    """Write generated code and start its renderer without waiting."""
    script_path, manim_media = prepare_render_script(
        code,
        media_dir=media_dir,
        script_name=script_name,
        output_dir=output_dir,
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
    media_dir: str | Path | None = None,
    script_name: str = "generated_animation.py",
    output_dir: str | Path | None = None,
) -> Path:
    """Write *code* to a script and execute it to render a video.

    All manim output goes under ``<media_dir>/media/``. The generated script
    goes under ``<media_dir>/code/`` (or ``output_dir`` if supplied).
    """
    script_path, process = start_render_process(
        code,
        media_dir=media_dir,
        script_name=script_name,
        output_dir=output_dir,
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
    "render_code",
    "start_render_process",
]

"""Optional rendering execution for generated Python code.

Provides convenience helpers to write generated code to a file and execute
it to render a video.

Usage::

    from phyanim.llm import render_code
    render_code(code, output_dir="outputs")
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def render_code(
    code: str,
    output_dir: str | Path | None = None,
    script_name: str = "generated_animation.py",
    media_dir: str | Path | None = None,
) -> Path:
    """Write *code* to a temporary script and execute it to render a video.

    Parameters
    ----------
    code:
        Python source code produced by :class:`PhysicsLLMPlanner`.
    output_dir:
        Directory to write the script into. Defaults to cwd.
        **Deprecated alias**: if ``media_dir`` is not given, this is also
        used as the working directory for manim (i.e. media/ goes here).
    script_name:
        File name for the generated script.
    media_dir:
        Working directory for the generated script (manim writes
        ``media/`` relative to this).  If ``None``, falls back to
        ``output_dir`` (or cwd).

    Returns
    -------
    Path
        Path to the generated script.
    """
    script_dir = Path(output_dir).resolve() if output_dir else Path.cwd()
    script_dir.mkdir(parents=True, exist_ok=True)
    script_path = script_dir / script_name
    script_path.write_text(code, encoding="utf-8")

    # Determine the working directory for manim (where media/ is created).
    if media_dir is not None:
        work_dir = Path(media_dir).resolve()
    else:
        work_dir = script_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    # Ensure the phyanim package is importable when running the script.
    project_root = Path(__file__).resolve().parents[2]
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{project_root}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(project_root)
    )

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(work_dir),
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Rendering failed (exit code {result.returncode}).\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return script_path


__all__ = ["render_code"]

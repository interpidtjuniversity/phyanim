"""Optional rendering execution for generated DSL code.

Provides convenience helpers to turn a DSL document into a rendered video file
in one call. This layer is intentionally optional: the planner only produces a
DSL dict, the parser only produces source code, and rendering is a separate
concern invoked only when the caller actually wants a video.

Usage::

    from phyanim.llm import generate_and_render
    video_path = generate_and_render(dsl, output_dir="outputs")
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from phyanim.llm.parser import DSLParser


def render_code(
    code: str,
    output_dir: str | Path | None = None,
    script_name: str = "generated_animation.py",
) -> Path:
    """Write ``code`` to a temporary script and execute it to render a video.

    The script is run with the current Python interpreter. manim writes its
    output under ``media/`` relative to the script's working directory.

    Parameters
    ----------
    code:
        Python source code produced by :class:`DSLParser`.
    output_dir:
        Directory to write the script into and run from. Defaults to the
        current working directory.
    script_name:
        File name for the generated script.

    Returns
    -------
    Path
        Path to the generated script (the video is under ``<output_dir>/media/``).
    """
    work_dir = Path(output_dir) if output_dir else Path.cwd()
    work_dir.mkdir(parents=True, exist_ok=True)
    script_path = work_dir / script_name
    script_path.write_text(code, encoding="utf-8")

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


def generate_and_render(
    dsl: dict[str, Any],
    output_dir: str | Path | None = None,
    script_name: str = "generated_animation.py",
) -> Path:
    """DSL → generated code → executed video render, in one call.

    Returns the path to the generated script; the rendered video lands under
    ``<output_dir>/media/videos/``.
    """
    code = DSLParser().parse(dsl)
    return render_code(code, output_dir=output_dir, script_name=script_name)


__all__ = ["render_code", "generate_and_render"]

"""Optional rendering execution for generated Python code.

Provides convenience helpers to write generated code to a file and execute
it to render a video.  All manim artifacts (videos, tex, text, images,
audio, partial movies) are confined to a single ``media_dir``.

Usage::

    from phyanim.llm import render_code
    render_code(code, media_dir="/home/phyanim/media")
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def render_code(
    code: str,
    media_dir: str | Path | None = None,
    script_name: str = "generated_animation.py",
    output_dir: str | Path | None = None,
) -> Path:
    """Write *code* to a script and execute it to render a video.

    All manim output (videos, tex, text, images, audio, partial movies)
    goes under ``<media_dir>/media/``.  The generated .py script goes
    under ``<media_dir>/code/`` (or ``output_dir`` if given).

    Parameters
    ----------
    code:
        Python source code produced by :class:`PhysicsLLMPlanner`.
    media_dir:
        Root directory for all generated files.  manim's ``config.media_dir``
        is set to ``<media_dir>/media/`` so ALL artifacts stay here.
        Defaults to ``<cwd>/media``.
    script_name:
        File name for the generated script.
    output_dir:
        Override directory for the .py script (defaults to
        ``<media_dir>/code/``).

    Returns
    -------
    Path
        Path to the generated script.
    """
    # Resolve directories.
    root = Path(media_dir).resolve() if media_dir else Path.cwd() / "media"
    root.mkdir(parents=True, exist_ok=True)

    # Script goes to <root>/code/ (or output_dir if given).
    script_dir = Path(output_dir).resolve() if output_dir else root / "code"
    script_dir.mkdir(parents=True, exist_ok=True)
    script_path = script_dir / script_name
    script_path.write_text(code, encoding="utf-8")

    # manim media goes to <root>/media/.
    manim_media = root / "media"
    manim_media.mkdir(parents=True, exist_ok=True)

    # Build environment.
    env = os.environ.copy()
    project_root = Path(__file__).resolve().parents[2]
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{project_root}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(project_root)
    )

    # Set manim's media_dir via environment variable so the generated
    # script picks it up.  manim reads MANIM_MEDIA_DIR if set.
    env["MANIM_MEDIA_DIR"] = str(manim_media)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(manim_media),
        env=env,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Rendering failed (exit code {result.returncode}).\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return script_path


__all__ = ["render_code"]

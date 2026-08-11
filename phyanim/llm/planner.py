from __future__ import annotations

import re
from typing import Any

from phyanim.llm.prompt_builder import build_analysis_system_prompt

ANALYSIS_SYSTEM_PROMPT = build_analysis_system_prompt()

# ---------------------------------------------------------------------------
# Code extraction and validation
# ---------------------------------------------------------------------------

class CodeValidationError(ValueError):
    """Raised when LLM-generated code fails to parse as Python."""


def _extract_code(raw: str) -> str:
    """Extract Python source code from an LLM response.

    Handles three cases:
    1. Raw code with no markdown fences.
    2. Code wrapped in ```python ... ``` fences.
    3. Code wrapped in ``` ... ``` fences.
    """
    # Try to find a ```python ... ``` block.
    match = re.search(r"```python\s*\n(.*?)```", raw, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try a generic ``` ... ``` block.
    match = re.search(r"```\s*\n(.*?)```", raw, re.DOTALL)
    if match:
        return match.group(1).strip()

    # No fences — assume the entire response is code.
    return raw.strip()


def _validate_python(code: str) -> None:
    """Validate that *code* parses as valid Python."""
    import ast
    try:
        ast.parse(code)
    except SyntaxError as exc:
        raise CodeValidationError(f"Python syntax error: {exc}") from exc

    # Check for essential structure.
    if "def construct" not in code and "scene.render()" not in code:
        raise CodeValidationError(
            "Code must contain a 'def construct' method or a 'scene.render()' call."
        )


def _inject_tts_config(code: str, tts_config: dict) -> str:
    """Inject TTS_CONFIG definition into generated code.

    The LLM is told to reference ``TTS_CONFIG`` without defining it.
    This function inserts the actual definition after the sys.path.insert
    line so the code is self-contained when executed.
    """
    if not tts_config:
        # No TTS config — inject a disabled default so code doesn't NameError.
        tts_config = {"provider": "none"}

    config_line = f"TTS_CONFIG = {repr(tts_config)}\n"

    # Try to insert after the last sys.path.insert line.
    lines = code.split("\n")
    insert_idx = None
    for i, line in enumerate(lines):
        if "sys.path.insert" in line:
            insert_idx = i + 1

    if insert_idx is not None:
        lines.insert(insert_idx, "")
        lines.insert(insert_idx + 1, config_line.rstrip())
        return "\n".join(lines)

    # Fallback: prepend.
    return config_line + code


def _inject_media_dir(code: str, media_dir: str) -> str:
    """Inject manim media_dir setting into generated code.

    Sets ``config.media_dir`` so ALL manim artifacts (videos, tex, text,
    images, audio, partial movies) go into the specified directory.
    Inserted after the sys.path.insert line, before any Scene class.
    """
    # Normalize to absolute path string.
    from pathlib import Path
    abs_media_dir = str(Path(media_dir).resolve())
    media_line = (
        "from manim import config as _manim_config; "
        f"_manim_config.media_dir = {abs_media_dir!r}"
    )

    lines = code.split("\n")
    insert_idx = None
    for i, line in enumerate(lines):
        if "sys.path.insert" in line:
            insert_idx = i + 1

    if insert_idx is not None:
        lines.insert(insert_idx, "")
        lines.insert(insert_idx + 1, media_line)
        return "\n".join(lines)

    return media_line + "\n" + code


def _inject_headless_config(code: str) -> str:
    """Force Manim into non-interactive headless mode.

    Prevents the render process from opening a preview window or file
    browser on error, which would keep the subprocess alive indefinitely.
    """
    headless_line = (
        "from manim import config as _manim_config; "
        "_manim_config.preview = False; "
        "_manim_config.force_window = False; "
        "_manim_config.show_in_file_browser = False"
    )

    lines = code.split("\n")
    insert_idx = None
    for i, line in enumerate(lines):
        if "sys.path.insert" in line:
            insert_idx = i + 1

    if insert_idx is not None:
        lines.insert(insert_idx, "")
        lines.insert(insert_idx + 1, headless_line)
        return "\n".join(lines)

    return headless_line + "\n" + code


def _inject_force_exit_hook(code: str) -> str:
    """Install a sys.excepthook that forces immediate process exit on error.

    Manim's SceneFileWriter creates a non-daemon writer thread.  If
    ``construct()`` raises, the normal cleanup that joins this thread is
    skipped, so the Python interpreter hangs waiting for it.  This hook
    calls ``os._exit(1)`` to bypass thread cleanup and exit immediately.
    """
    hook_block = (
        "import sys\n"
        "import os\n"
        "import traceback\n"
        "\n"
        "def _phyanim_force_exit(exc_type, exc_value, tb):\n"
        "    traceback.print_exception(exc_type, exc_value, tb)\n"
        "    os._exit(1)\n"
        "\n"
        "sys.excepthook = _phyanim_force_exit\n"
    )

    lines = code.split("\n")
    insert_idx = None
    for i, line in enumerate(lines):
        if "sys.path.insert" in line:
            insert_idx = i + 1

    if insert_idx is not None:
        lines.insert(insert_idx, "")
        lines.insert(insert_idx + 1, hook_block.rstrip())
        return "\n".join(lines)

    return hook_block + code


def _rename_scene_class(code: str, new_name: str) -> str:
    """Rename the generated Scene class to *new_name*.

    manim names its output video file after the Scene class name
    (e.g. ``class Foo(Scene)`` → ``Foo.mp4``).  By renaming the class
    we control the output filename.

    This replaces the first ``class <Name>(`` that inherits from a Scene
    subclass (PhyAnimScene, PhyAnimCodeScene, PhyAnimHybridScene,
    PhyAnimationMultiLayerScene2D, or Scene).
    """
    # Match: class <Identifier>(<Base>):
    pattern = re.compile(
        r"class\s+(\w+)\s*\(\s*"
        r"(?:PhyAnim\w*|PhyAnimationMultiLayerScene2D|Scene)"
        r"\s*\)\s*:"
    )
    match = pattern.search(code)
    if match:
        old_name = match.group(1)
        # Replace the class definition.
        code = code[: match.start()] + code[match.start():].replace(
            f"class {old_name}(", f"class {new_name}(", 1
        )
        # Replace references to the old name (e.g. in __main__ block).
        code = code.replace(f"{old_name}().render()", f"{new_name}().render()")
    return code


__all__ = ["ANALYSIS_SYSTEM_PROMPT", "CodeValidationError"]

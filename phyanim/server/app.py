"""Flask server exposing the /generate_video endpoint.

Accepts a user prompt (and optional image URLs), calls the LLM to generate
Python animation code, renders it to a video, and returns the video file.
ALL generated files (scripts, videos, tex, audio, images) are confined
to the configured ``media_dir``.

Usage::

    from phyanim.server.config import ServerConfig, LLMProvider
    from phyanim.server.app import create_app

    config = ServerConfig(
        llm_provider=LLMProvider.DEEPSEEK,
        llm_api_key="sk-...",
        tts=TTSConfig(provider="minimax", api_key="...", voice_id="male-qn-qingse"),
        media_dir="/home/phyanim/media",
    )
    app = create_app(config)
    app.run(host="0.0.0.0", port=5000)

API:
    POST /generate_video
        Body: {"prompt": "...", "image_urls": ["..."]}

    Response: video file (mp4)
"""

from __future__ import annotations

import logging
import time
import traceback
from pathlib import Path
from typing import Any

from flask import Flask, request, send_file, jsonify

from phyanim.llm.planner import PhysicsLLMPlanner, CodeValidationError
from phyanim.llm.runner import render_code
from phyanim.server.config import ServerConfig

logger = logging.getLogger(__name__)


def create_app(config: ServerConfig) -> Flask:
    """Create and configure the Flask application.

    Parameters
    ----------
    config:
        Server configuration (LLM provider, TTS, media_dir, etc.).
    """
    app = Flask(__name__)
    app.config["SERVER_CONFIG"] = config

    @app.route("/generate_video", methods=["POST"])
    def generate_video() -> Any:
        """Generate a physics animation video from a natural language prompt.

        Request JSON body:
            prompt (str, required): Natural language physics problem.
            image_urls (list[str], optional): Image URLs for vision models.

        Returns:
            - On success: the rendered video file (mp4).
            - On error: JSON with error details.
        """
        data = request.get_json(silent=True) or {}
        prompt = (data.get("user_prompt") or data.get("prompt") or "").strip()
        image_urls = data.get("image_urls") or []

        if not prompt:
            return jsonify({"error": "Missing 'prompt' field"}), 400

        cfg: ServerConfig = app.config["SERVER_CONFIG"]

        # Generate a unique timestamp for this request.
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        scene_name = f"Scene_{timestamp}"
        script_name = f"generated_{timestamp}.py"

        try:
            # 1. Resolve media directory.
            media_dir = cfg.resolved_media_dir()
            manim_media = cfg.resolved_manim_media_dir()
            code_dir = cfg.resolved_code_dir()

            # 2. Create LLM client and planner.
            #    Inject TTS config and media_dir into generated code.
            client = cfg.make_llm_client()
            planner = PhysicsLLMPlanner(
                client,
                tts_config=cfg.to_tts_dict(),
                media_dir=str(manim_media),
            )

            # 3. Generate Python code from the prompt.
            logger.info("[%s] Generating code for prompt: %s", timestamp, prompt[:100])
            code = planner.plan(prompt, images=image_urls or None, scene_name=scene_name)
            logger.info("[%s] Code generated (%d chars)", timestamp, len(code))

            # 4. Render the code to a video.
            #    All manim output goes to <media_dir>/media/.
            logger.info("[%s] Rendering video...", timestamp)
            script_path = render_code(
                code,
                media_dir=str(media_dir),
                script_name=script_name,
                output_dir=str(code_dir),
            )

            # 5. Find the rendered video file.
            #    manim writes to <media_dir>/media/videos/.../<SceneName>.mp4
            video_search_dir = manim_media / "videos"
            video_files = list(video_search_dir.rglob("*.mp4"))
            # Exclude partial movie files.
            video_files = [f for f in video_files if "partial_movie" not in str(f)]

            if not video_files:
                return jsonify({"error": "No video file produced"}), 500

            # Return the most recently modified video.
            video_path = max(video_files, key=lambda f: f.stat().st_mtime)
            logger.info("[%s] Video ready: %s", timestamp, video_path)

            return send_file(
                str(video_path),
                mimetype="video/mp4",
                as_attachment=True,
                download_name=video_path.name,
            )

        except CodeValidationError as exc:
            logger.error("Code validation failed: %s", exc)
            return jsonify({"error": f"Code validation failed: {exc}"}), 422

        except Exception as exc:
            logger.error("Unexpected error: %s\n%s", exc, traceback.format_exc())
            return jsonify({"error": str(exc)}), 500

    @app.route("/health", methods=["GET"])
    def health() -> Any:
        """Health check endpoint."""
        cfg: ServerConfig = app.config["SERVER_CONFIG"]
        return jsonify({
            "status": "ok",
            "llm_provider": cfg.llm_provider.value,
            "llm_model": cfg.resolved_model(),
            "tts_enabled": cfg.tts.enabled,
            "media_dir": str(cfg.resolved_media_dir()),
        })

    return app

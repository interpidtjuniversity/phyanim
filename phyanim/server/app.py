"""Flask server for asynchronous physics animation generation.

API:
    POST /generate_video
        Body: {"prompt": "...", "image_urls": ["..."]}
        Returns a unique ``script_name`` immediately.

    GET /get_status?script_name=Scene_...
        Returns only the persisted job status.

    GET /get_video?script_name=Scene_...
        Returns the completed MP4 file.

    GET /get_code?script_name=Scene_...
        Returns job status or the generated render code.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any

from flask import Flask, jsonify, request, send_file
from sympy import N

from phyanim.server.config import ServerConfig
from phyanim.server.jobs import RenderJobManager, is_valid_script_name

logger = logging.getLogger(__name__)

_ACTIVE_STATUSES = frozenset({"queued", "generating", "rendering"})


def create_app(config: ServerConfig) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.json.ensure_ascii = False
    app.config["SERVER_CONFIG"] = config
    app.extensions["render_job_manager"] = RenderJobManager(config)

    def job_manager() -> RenderJobManager:
        return app.extensions["render_job_manager"]

    def requested_job() -> tuple[str | None, tuple[Any, int] | None]:
        script_name = request.args.get("script_name", "")
        if not is_valid_script_name(script_name):
            return None, (
                jsonify({"error": "Invalid or missing 'script_name' parameter"}),
                400,
            )
        return script_name, None

    def status_response(job: dict[str, Any]) -> tuple[Any, int]:
        return jsonify({
            "script_name": job["script_name"],
            "status": job["status"],
        }), 202

    def failure_response(job: dict[str, Any]) -> tuple[Any, int]:
        return jsonify({
            "script_name": job["script_name"],
            "status": "failed",
            "error_code": job.get("error_code", "render_failed"),
            "error": job.get(""),
        }), 500

    @app.route("/generate_video", methods=["POST"])
    def generate_video() -> Any:
        """Queue code generation and rendering, returning its identifier."""
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "Request body must be a JSON object"}), 400

        prompt_value = data.get("user_prompt") or data.get("prompt")
        if not isinstance(prompt_value, str) or not prompt_value.strip():
            return jsonify({"error": "Missing 'prompt' field"}), 400
        
        history_messages = data.get("history_messages")
        if history_messages is None:
            history_messages = []

        image_urls = data.get("image_urls")
        if image_urls is None:
            image_urls = []
        if not isinstance(image_urls, list) or not all(
            isinstance(url, str) for url in image_urls
        ):
            return jsonify({"error": "'image_urls' must be a list of strings"}), 400

        try:
            job = job_manager().submit(prompt_value.strip(), history_messages, image_urls)
            return jsonify({
                "script_name": job["script_name"],
                "status": "queued",
            }), 202
        except Exception as exc:
            logger.error("Unable to submit render job: %s\n%s", exc, traceback.format_exc())
            return jsonify({"error": "Unable to submit video generation job"}), 500

    @app.route("/get_status", methods=["GET"])
    def get_status() -> Any:
        """Return only the persisted status of a video generation job."""
        script_name, error = requested_job()
        if error:
            return error

        job = job_manager().get(script_name)
        if job is None:
            return jsonify({"error": "Video job not found"}), 404
        return jsonify({"status": job["status"], "error_code": job.get("error_code", None), "error": job.get("error", None)})

    @app.route("/get_video", methods=["GET"])
    def get_video() -> Any:
        """Return a job's MP4 when rendering has completed successfully."""
        script_name, error = requested_job()
        if error:
            return error

        manager = job_manager()
        job = manager.get(script_name)
        if job is None:
            return jsonify({"error": "Video job not found"}), 404
        if job["status"] in _ACTIVE_STATUSES:
            return jsonify({"error": "Video is not ready"}), 409
        if job["status"] == "failed":
            return jsonify({"error": job.get('error')}), 409

        video_path = manager.find_video(script_name)
        if video_path is None:
            return jsonify({"error": "Rendered video file not found"}), 404
        return send_file(
            str(video_path),
            mimetype="video/mp4",
            as_attachment=True,
            download_name=video_path.name,
        )

    @app.route("/get_code", methods=["GET"])
    def get_code() -> Any:
        """Return the generated Python render code for a job."""
        script_name, error = requested_job()
        if error:
            return error

        manager = job_manager()
        job = manager.get(script_name)
        if job is None:
            return jsonify({"error": "Video job not found"}), 404

        # 只获取源码，不获取注入后的代码
        code_path = manager.source_code_path(script_name)
        analysis_info_path = manager.analysis_info_path(script_name)
        if code_path.is_file():
            return jsonify({
                "script_name": script_name,
                "code": code_path.read_text(encoding="utf-8"),
                "analysis_info": analysis_info_path.read_text(encoding="utf-8"),
            })
        if job["status"] in _ACTIVE_STATUSES:
            return status_response(job)
        if job["status"] == "failed":
            return failure_response(job)
        return jsonify({"error": "Generated code not found"}), 404

    @app.route("/health", methods=["GET"])
    def health() -> Any:
        """Health check endpoint."""
        cfg: ServerConfig = app.config["SERVER_CONFIG"]
        return jsonify({
            "status": "ok",
            "llm_provider": cfg.llm_provider.value,
            "llm_model": cfg.resolved_model(),
            "tts_enabled": cfg.tts.enabled,
            "root_dir": str(cfg.resolved_root_dir()),
        })

    return app

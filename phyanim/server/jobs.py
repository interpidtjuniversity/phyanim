"""Persistent background jobs for LLM code generation and video rendering."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from phyanim.llm.planner import CodeValidationError, PhysicsLLMPlanner
from phyanim.llm.runner import start_render_process
from phyanim.server.config import ServerConfig

from phyanim.llm.runner import prepare_source_script

logger = logging.getLogger(__name__)

_ACTIVE_STATUSES = frozenset({"queued", "generating", "rendering"})
_SCRIPT_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")


def is_valid_script_name(script_name: object) -> bool:
    """Return whether *script_name* is a safe Python identifier and basename."""
    return isinstance(script_name, str) and bool(
        _SCRIPT_NAME_PATTERN.fullmatch(script_name)
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RenderJobManager:
    """Generate code and monitor render subprocesses in background threads."""

    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.root_dir = config.resolved_media_dir()
        self.code_dir = config.resolved_code_dir()
        self.source_code_dir = config.resolved_source_code_dir()
        self.manim_media_dir = config.resolved_manim_media_dir()
        self.jobs_dir = self.root_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._recover_jobs()

    def submit(self, prompt: str, history_messages: list[Any], image_urls: list[str]) -> dict[str, Any]:
        """Create a persisted job and start its worker thread."""
        script_name = self._new_script_name()
        now = _utc_now()
        job = {
            "script_name": script_name,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
        }
        with self._lock:
            self._jobs[script_name] = job
            self._persist(job)

        worker = threading.Thread(
            target=self._run_job,
            args=(script_name, prompt, history_messages, image_urls),
            name=f"render-{script_name}",
            daemon=True,
        )
        worker.start()
        return dict(job)

    def get(self, script_name: str) -> dict[str, Any] | None:
        """Return a snapshot of one job's persisted state."""
        with self._lock:
            job = self._jobs.get(script_name)
            return dict(job) if job else None

    def source_code_path(self, script_name: str) -> Path:
        return self.source_code_dir / f"{script_name}.py"

    def find_video(self, script_name: str) -> Path | None:
        """Find the exact completed video for a job."""
        video_dir = self.manim_media_dir / "videos"
        if not video_dir.exists():
            return None
        candidates = [
            path
            for path in video_dir.rglob(f"{script_name}.mp4")
            if not any("partial_movie" in part for part in path.parts)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)

    def _new_script_name(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        while True:
            name = f"Scene_{timestamp}_{uuid4().hex[:12]}"
            with self._lock:
                if name not in self._jobs and not self._job_path(name).exists():
                    return name

    def _run_job(
        self,
        script_name: str,
        prompt: str,
        history_messages: list[Any],
        image_urls: list[str],
    ) -> None:
        try:
            self._update(script_name, status="generating")
            client = self.config.make_llm_client()
            planner = PhysicsLLMPlanner(
                client,
                tts_config=self.config.to_tts_dict(),
                media_dir=str(self.manim_media_dir),
            )
            logger.info("[%s] Generating code", script_name)
            source_code, code = planner.plan(
                prompt,
                history_messages,
                images=image_urls or None,
                scene_name=script_name,
            )
            logger.info("[%s] Code generated (%d chars)", script_name, len(code))
            
            prepare_source_script(source_code, self.root_dir, f"{script_name}.py", self.source_code_dir)

            script_path, process = start_render_process( 
                code,
                media_dir=self.root_dir,
                script_name=f"{script_name}.py",
                output_dir=self.code_dir,
            )
            self._update(
                script_name,
                status="rendering",
                pid=process.pid,
                script_path=str(script_path),
            )
            logger.info("[%s] Rendering with PID %s", script_name, process.pid)
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                details = self._render_failure_details(
                    process.returncode, stdout, stderr
                )
                raise RuntimeError(details)

            video_path = self.find_video(script_name)
            if video_path is None:
                raise RuntimeError("Rendering exited successfully but produced no video")

            self._update(
                script_name,
                status="succeeded",
                video_path=str(video_path),
                pid=None,
                error=None,
            )
            logger.info("[%s] Video ready: %s", script_name, video_path)
        except CodeValidationError as exc:
            self._fail(script_name, "code_validation_failed", str(exc))
        except Exception as exc:
            logger.error(
                "[%s] Background job failed: %s\n%s",
                script_name,
                exc,
                traceback.format_exc(),
            )
            self._fail(script_name, "render_failed", str(exc))

    def _render_failure_details(
        self,
        returncode: int | None,
        stdout: str,
        stderr: str,
    ) -> str:
        output = (stderr or stdout or "").strip()
        if output:
            output = output[-2000:]
            return f"Rendering failed with exit code {returncode}: {output}"
        return f"Rendering failed with exit code {returncode}"

    def _fail(self, script_name: str, error_code: str, message: str) -> None:
        clean_message = " ".join(message.split())[:3000] or "Unknown failure"
        self._update(
            script_name,
            status="failed",
            error_code=error_code,
            error=clean_message,
            pid=None,
        )

    def _update(self, script_name: str, **changes: Any) -> None:
        with self._lock:
            job = self._jobs[script_name]
            job.update(changes)
            job["updated_at"] = _utc_now()
            self._persist(job)

    def _persist(self, job: dict[str, Any]) -> None:
        path = self._job_path(job["script_name"])
        temp_path = path.with_suffix(f".json.{uuid4().hex}.tmp")
        temp_path.write_text(
            json.dumps(job, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temp_path, path)

    def _job_path(self, script_name: str) -> Path:
        return self.jobs_dir / f"{script_name}.json"

    def _recover_jobs(self) -> None:
        for path in self.jobs_dir.glob("*.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
                script_name = job.get("script_name")
                if not is_valid_script_name(script_name):
                    raise ValueError("invalid script_name")
                self._jobs[script_name] = job
                if job.get("status") in _ACTIVE_STATUSES:
                    video_path = self.find_video(script_name)
                    if video_path:
                        self._update(
                            script_name,
                            status="succeeded",
                            video_path=str(video_path),
                            pid=None,
                            error=None,
                        )
                    else:
                        self._fail(
                            script_name,
                            "server_restarted",
                            "Rendering was interrupted by a server restart",
                        )
            except Exception as exc:
                logger.warning("Ignoring invalid job file %s: %s", path, exc)


__all__ = ["RenderJobManager", "is_valid_script_name"]

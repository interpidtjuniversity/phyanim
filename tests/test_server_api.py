"""Tests for asynchronous video generation HTTP endpoints."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from phyanim.server.app import create_app
from phyanim.server.config import ServerConfig
from phyanim.server.jobs import RenderJobManager


class _Planner:
    def __init__(self, client, tts_config=None, media_dir=None) -> None:
        self.media_dir = media_dir

    def plan(self, prompt, *, images=None, scene_name=None) -> str:
        return f"class {scene_name}:\n    def construct(self):\n        pass\n"


class _SuccessfulProcess:
    pid = 1234
    returncode = 0

    def communicate(self) -> tuple[str, str]:
        return "rendered", ""


class _FailedProcess:
    pid = 5678
    returncode = 2

    def communicate(self) -> tuple[str, str]:
        return "", "private renderer details"


def _wait_for_status(client, script_name: str, status: str) -> dict:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        response = client.get("/get_video", query_string={"script_name": script_name})
        if response.is_json and response.get_json().get("status") == status:
            return response.get_json()
        if status == "succeeded" and response.status_code == 200:
            return {"status": "succeeded"}
        time.sleep(0.01)
    pytest.fail(f"job {script_name} did not reach {status}")


def test_generate_code_and_video_in_background(tmp_path, monkeypatch) -> None:
    config = ServerConfig(media_dir=str(tmp_path))
    monkeypatch.setattr("phyanim.server.jobs.PhysicsLLMPlanner", _Planner)
    monkeypatch.setattr(config, "make_llm_client", lambda: object())

    def start_process(code, media_dir, script_name, output_dir):
        script_path = Path(output_dir) / script_name
        script_path.write_text(code, encoding="utf-8")
        scene_name = Path(script_name).stem
        video_path = Path(media_dir) / "media" / "videos" / "1080p60" / f"{scene_name}.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"video-data")
        return script_path, _SuccessfulProcess()

    monkeypatch.setattr("phyanim.server.jobs.start_render_process", start_process)
    app = create_app(config)
    client = app.test_client()

    response = client.post("/generate_video", json={"prompt": "animate", "image_urls": []})
    assert response.status_code == 202
    body = response.get_json()
    assert body["status"] == "queued"
    script_name = body["script_name"]
    assert script_name.startswith("Scene_")
    assert "." not in script_name

    _wait_for_status(client, script_name, "succeeded")

    code_response = client.get("/get_code", query_string={"script_name": script_name})
    assert code_response.status_code == 200
    assert f"class {script_name}" in code_response.get_json()["code"]

    video_response = client.get("/get_video", query_string={"script_name": script_name})
    assert video_response.status_code == 200
    assert video_response.mimetype == "video/mp4"
    assert video_response.data == b"video-data"


def test_get_code_returns_utf8_chinese_without_ascii_escaping(tmp_path) -> None:
    app = create_app(ServerConfig(media_dir=str(tmp_path)))
    manager = app.extensions["render_job_manager"]
    script_name = "Scene_chinese"
    manager._jobs[script_name] = {
        "script_name": script_name,
        "status": "succeeded",
    }
    code = '# 中文注释\ntitle = "简谐运动"\n'
    manager.code_path(script_name).write_text(code, encoding="utf-8")

    response = app.test_client().get(
        "/get_code",
        query_string={"script_name": script_name},
    )

    assert response.status_code == 200
    assert response.content_type == "application/json"
    assert response.get_json()["code"] == code
    assert "中文注释".encode("utf-8") in response.data
    assert b"\\u4e2d\\u6587" not in response.data


def test_generate_video_validates_request(tmp_path) -> None:
    client = create_app(ServerConfig(media_dir=str(tmp_path))).test_client()

    assert client.post("/generate_video").status_code == 400
    assert client.post("/generate_video", json={}).status_code == 400
    response = client.post(
        "/generate_video",
        json={"prompt": "animate", "image_urls": "not-a-list"},
    )
    assert response.status_code == 400


@pytest.mark.parametrize("endpoint", ["/get_video", "/get_code"])
def test_query_rejects_unsafe_or_unknown_names(tmp_path, endpoint) -> None:
    client = create_app(ServerConfig(media_dir=str(tmp_path))).test_client()

    assert client.get(endpoint, query_string={"script_name": "../secret"}).status_code == 400
    assert client.get(endpoint, query_string={"script_name": "Scene_unknown"}).status_code == 404


def test_get_video_reports_active_and_failed_jobs(tmp_path) -> None:
    app = create_app(ServerConfig(media_dir=str(tmp_path)))
    manager = app.extensions["render_job_manager"]
    now = "2026-07-26T00:00:00+00:00"
    manager._jobs["Scene_active"] = {
        "script_name": "Scene_active",
        "status": "rendering",
        "created_at": now,
        "updated_at": now,
    }
    manager._jobs["Scene_failed"] = {
        "script_name": "Scene_failed",
        "status": "failed",
        "error_code": "render_failed",
        "error": "renderer stopped",
        "created_at": now,
        "updated_at": now,
    }
    client = app.test_client()

    active = client.get("/get_video", query_string={"script_name": "Scene_active"})
    assert active.status_code == 202
    assert active.get_json()["status"] == "rendering"

    failed = client.get("/get_video", query_string={"script_name": "Scene_failed"})
    assert failed.status_code == 500
    assert failed.get_json()["error_code"] == "render_failed"


def test_background_process_failure_is_persisted_and_sanitized(tmp_path, monkeypatch) -> None:
    config = ServerConfig(media_dir=str(tmp_path))
    monkeypatch.setattr("phyanim.server.jobs.PhysicsLLMPlanner", _Planner)
    monkeypatch.setattr(config, "make_llm_client", lambda: object())

    def start_process(code, media_dir, script_name, output_dir):
        script_path = Path(output_dir) / script_name
        script_path.write_text(code, encoding="utf-8")
        return script_path, _FailedProcess()

    monkeypatch.setattr("phyanim.server.jobs.start_render_process", start_process)
    app = create_app(config)
    client = app.test_client()
    submitted = client.post("/generate_video", json={"prompt": "animate"}).get_json()

    failed = _wait_for_status(client, submitted["script_name"], "failed")
    assert failed["error_code"] == "render_failed"
    assert failed["error"] == "Video generation failed"
    assert "private renderer details" not in failed["error"]

    persisted = app.extensions["render_job_manager"].get(submitted["script_name"])
    assert "private renderer details" in persisted["error"]


def test_find_video_ignores_partial_and_unrelated_files(tmp_path) -> None:
    manager = RenderJobManager(ServerConfig(media_dir=str(tmp_path)))
    partial = (
        tmp_path
        / "media"
        / "videos"
        / "1080p60"
        / "partial_movie_files"
        / "Scene_target.mp4"
    )
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"partial")
    unrelated = tmp_path / "media" / "videos" / "1080p60" / "Scene_other.mp4"
    unrelated.write_bytes(b"other")

    assert manager.find_video("Scene_target") is None


def test_recover_interrupted_job_from_disk(tmp_path) -> None:
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    job = {
        "script_name": "Scene_interrupted",
        "status": "rendering",
        "created_at": "2026-07-26T00:00:00+00:00",
        "updated_at": "2026-07-26T00:00:00+00:00",
        "pid": 999,
    }
    (jobs_dir / "Scene_interrupted.json").write_text(json.dumps(job), encoding="utf-8")

    manager = RenderJobManager(ServerConfig(media_dir=str(tmp_path)))

    recovered = manager.get("Scene_interrupted")
    assert recovered["status"] == "failed"
    assert recovered["error_code"] == "server_restarted"


def test_recover_completed_video_from_disk(tmp_path) -> None:
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    job = {
        "script_name": "Scene_completed",
        "status": "rendering",
        "created_at": "2026-07-26T00:00:00+00:00",
        "updated_at": "2026-07-26T00:00:00+00:00",
    }
    (jobs_dir / "Scene_completed.json").write_text(json.dumps(job), encoding="utf-8")
    video = tmp_path / "media" / "videos" / "1080p60" / "Scene_completed.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"done")

    manager = RenderJobManager(ServerConfig(media_dir=str(tmp_path)))

    recovered = manager.get("Scene_completed")
    assert recovered["status"] == "succeeded"
    assert manager.find_video("Scene_completed") == video

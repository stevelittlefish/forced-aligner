"""Exercise the real job worker and HTTP contract, without the GPU circus."""

import json
import time
from threading import Event

import pytest
from fastapi.testclient import TestClient

from app.align import AlignResult, AudioDecodeError, LanguageUnsupported, Line, Word


def fake_result():
    return AlignResult(
        "en",
        2.5,
        [
            Line(
                "hello world",
                0.0,
                0.5,
                [
                    Word("hello", 0.0, 0.5),
                    Word("world", None, None),
                ],
            )
        ],
    )


def submit(client, params=None, audio=b"RIFFfake", **kw):
    return client.post(
        "/v1/align",
        files={"audio": ("v.wav", audio, "audio/wav")},
        data={"params": params or json.dumps({"text": "hello world"})},
        **kw,
    )


def wait_job(client, job_id):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        response = client.get(f"/v1/jobs/{job_id}")
        assert response.status_code == 200
        job = response.json()
        if job["state"] in ("succeeded", "failed"):
            return job
        time.sleep(0.01)
    pytest.fail("job did not finish")


def test_job_artifact_and_restart(make_app, monkeypatch):
    main = make_app(default_language="de")
    seen = []

    def fake(path, text, language):
        from pathlib import Path

        seen.append((Path(path).read_bytes(), text, language))
        return fake_result()

    monkeypatch.setattr(main.aligner, "align", fake)
    with TestClient(main.app) as client:
        response = submit(client)
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        job = wait_job(client, job_id)
        assert job["state"] == "succeeded"
        artifact = job["artifacts"][0]
        result = client.get(f"/v1/jobs/{job_id}/result/alignment.json")
        assert result.status_code == 200
        assert result.headers["content-type"] == "application/json"
        assert artifact == {
            "name": "alignment.json",
            "kind": "metadata",
            "content_type": "application/json",
            "bytes": len(result.content),
        }
        assert result.json()["lines"][0]["words"][1]["start"] is None
        assert seen == [(b"RIFFfake", "hello world", "de")]
        assert not (main.root / job_id / "input.audio").exists()
        assert client.get(f"/v1/jobs/{job_id}/result/status.json").status_code == 404
    # Completed artifacts survive a process restart until ASS harvests them.
    with TestClient(main.app) as client:
        assert client.get(f"/v1/jobs/{job_id}/result/alignment.json").status_code == 200


def test_serial_worker_keeps_http_responsive_and_bounds_queue(make_app, monkeypatch):
    main = make_app(max_pending_jobs=2)
    entered, release = Event(), Event()
    calls = []

    def fake(*args):
        calls.append(args)
        entered.set()
        assert release.wait(3)
        return fake_result()

    monkeypatch.setattr(main.aligner, "align", fake)
    with TestClient(main.app) as client:
        try:
            first = submit(client).json()["job_id"]
            assert entered.wait(2)
            second = submit(client).json()["job_id"]
            assert client.get("/health").status_code == 200
            assert client.get("/v1/info").json()["vram"]["cuda"] is False
            assert client.get(f"/v1/jobs/{first}").json()["state"] == "running"
            assert client.get(f"/v1/jobs/{second}").json()["state"] == "queued"
            assert len(calls) == 1
            assert submit(client).status_code == 503
            assert (
                client.get(f"/v1/jobs/{first}/result/alignment.json").status_code == 409
            )
        finally:
            release.set()
        assert wait_job(client, first)["state"] == "succeeded"
        assert wait_job(client, second)["state"] == "succeeded"
        assert len(calls) == 2
        assert submit(client).status_code == 202


@pytest.mark.parametrize(
    "error",
    [
        LanguageUnsupported("unknown language"),
        AudioDecodeError("bad audio"),
        RuntimeError("CUDA out of memory"),
    ],
)
def test_failure_releases_worker_and_upload(make_app, monkeypatch, error):
    main = make_app(max_pending_jobs=1)

    def boom(*args):
        raise error

    monkeypatch.setattr(main.aligner, "align", boom)
    with TestClient(main.app) as client:
        job_id = submit(client).json()["job_id"]
        job = wait_job(client, job_id)
        assert job["state"] == "failed"
        assert job["error"] == str(error)
        assert job["artifacts"] == []
        assert not (main.root / job_id / "input.audio").exists()
        monkeypatch.setattr(main.aligner, "align", lambda *args: fake_result())
        assert wait_job(client, submit(client).json()["job_id"])["state"] == "succeeded"


@pytest.mark.parametrize(
    "params",
    [
        "{broken",
        "[]",
        "null",
        '{"text": 4}',
        '{"text": " "}',
        '{"text": "hello", "language": []}',
        '{"text": "hello", "language": ""}',
    ],
)
def test_invalid_params(make_app, params):
    main = make_app()
    with TestClient(main.app) as client:
        assert submit(client, params=params).status_code == 400
        assert list(main.root.iterdir()) == []


def test_upload_limits_cleanup_and_unknown_jobs(make_app, monkeypatch):
    main = make_app(max_pending_jobs=1, max_upload_mb=1)
    monkeypatch.setattr(main.aligner, "align", lambda *args: fake_result())
    with TestClient(main.app) as client:
        assert submit(client, audio=b"").status_code == 400
        assert submit(client, audio=b"x" * (1024**2 + 1)).status_code == 413
        assert list(main.root.iterdir()) == []
        assert client.get("/v1/jobs/unknown").status_code == 404
        assert client.get("/v1/jobs/" + "0" * 32).status_code == 404
        assert submit(client).status_code == 202
        assert client.post("/align").status_code == 404
        assert client.get("/models").status_code == 404


def test_restart_marks_interrupted_jobs_failed(make_app):
    main = make_app()
    directory = main.root / ("a" * 32)
    directory.mkdir(parents=True)
    main.save_status(directory, {"job_id": directory.name, "state": "running"})
    (directory / "input.audio").write_bytes(b"abandoned")
    with TestClient(main.app) as client:
        job = client.get(f"/v1/jobs/{directory.name}").json()
        assert job["state"] == "failed"
        assert job["error"] == "service restarted"
        assert not (directory / "input.audio").exists()


def test_preload_and_failure_readiness(make_app, monkeypatch):
    main = make_app(preload_languages=("en",))
    warmed = []
    monkeypatch.setattr(main.aligner, "preload", warmed.append)
    with TestClient(main.app):
        assert warmed == ["en"]

    def boom(lang):
        raise RuntimeError("model failed")

    monkeypatch.setattr(main.aligner, "preload", boom)
    with pytest.raises(RuntimeError, match="model failed"), TestClient(main.app):
        pass


def test_auth_covers_submission_polling_and_download(make_app, monkeypatch):
    main = make_app(auth_token="secret")
    monkeypatch.setattr(main.aligner, "align", lambda *args: fake_result())
    with TestClient(main.app) as client:
        assert submit(client).status_code == 401
        assert client.get("/v1/jobs/" + "a" * 32).status_code == 401
        assert (
            client.get("/v1/jobs/" + "a" * 32 + "/result/alignment.json").status_code
            == 401
        )
        assert (
            submit(client, headers={"Authorization": "Bearer secret"}).status_code
            == 202
        )

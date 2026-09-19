"""ASS job API. The GPU does one thing at a time; HTTP needn't join the queue."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from threading import BoundedSemaphore, Lock
from uuid import uuid4

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from . import config
from .align import Aligner, AudioDecodeError, LanguageUnsupported, supported_languages

log = logging.getLogger("forced-aligner")
cfg = config.load()
aligner = Aligner(cfg)
root = Path(cfg.jobs_dir)
state_lock = Lock()
slots = BoundedSemaphore(cfg.max_pending_jobs)


def save_status(directory: Path, status: dict) -> None:
    # Publish whole documents, never half a success for ASS to harvest.
    with state_lock:
        temporary = directory / "status.tmp"
        temporary.write_text(json.dumps(status), encoding="utf-8")
        temporary.replace(directory / "status.json")


def run_job(directory: Path, text: str, language: str) -> None:
    status = {"job_id": directory.name, "state": "running", "artifacts": []}
    try:
        save_status(directory, status)
        result = aligner.align(str(directory / "input.audio"), text, language)
        output = directory / "alignment.json"
        output.write_text(json.dumps(asdict(result), allow_nan=False), encoding="utf-8")
        status.update(
            state="succeeded",
            artifacts=[
                {
                    "name": output.name,
                    "kind": "metadata",
                    "content_type": "application/json",
                    "bytes": output.stat().st_size,
                }
            ],
        )
    except Exception as exc:
        log.exception("alignment job %s failed", directory.name)
        status.update(state="failed", error=str(exc), artifacts=[])
    finally:
        try:
            (directory / "input.audio").unlink(missing_ok=True)
            save_status(directory, status)
        finally:
            slots.release()


def run_synchronous(directory: Path, text: str, language: str) -> JSONResponse:
    # The shared worker owns cleanup even if the caller gives up waiting.
    try:
        try:
            result = aligner.align(str(directory / "input.audio"), text, language)
        except LanguageUnsupported as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": f"unsupported language {language!r}: {exc}",
                    "supported": supported_languages(),
                },
            ) from exc
        except AudioDecodeError as exc:
            raise HTTPException(
                status_code=415, detail=f"audio could not be decoded: {exc}"
            ) from exc
        except Exception as exc:
            log.exception("synchronous alignment failed")
            raise HTTPException(
                status_code=500, detail=f"alignment failed: {exc}"
            ) from exc
        body = asdict(result)
        body["duration"] = round(result.duration, 3)
        return JSONResponse(body)
    finally:
        shutil.rmtree(directory, ignore_errors=True)
        slots.release()


@asynccontextmanager
async def lifespan(app: FastAPI):
    root.mkdir(parents=True, exist_ok=True)
    # A process restart cannot resume inference. Report failure instead of
    # letting ASS poll an abandoned 'running' job until the heat death of time.
    for path in root.glob("*/status.json"):
        status = json.loads(path.read_text(encoding="utf-8"))
        if status["state"] in ("queued", "running"):
            status.update(state="failed", error="service restarted", artifacts=[])
            save_status(path.parent, status)
            (path.parent / "input.audio").unlink(missing_ok=True)
    for lang in cfg.preload_languages:
        # Startup failure must fail readiness, not advertise a broken model.
        await asyncio.to_thread(aligner.preload, lang)
    app.state.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="align")
    try:
        yield
    finally:
        await asyncio.to_thread(app.state.worker.shutdown, wait=True)


app = FastAPI(title="forced-aligner", version="0.2.0", lifespan=lifespan)


@app.exception_handler(HTTPException)
async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
    body = exc.detail if isinstance(exc.detail, dict) else {"error": exc.detail}
    return JSONResponse(status_code=exc.status_code, content=body)


def check_auth(authorization: str | None) -> None:
    if cfg.auth_token and authorization != f"Bearer {cfg.auth_token}":
        raise HTTPException(status_code=401, detail="bad or missing bearer token")


def job_status(job_id: str) -> tuple[Path, dict]:
    # Only IDs we generate can address disk; no user-selected file paths.
    if len(job_id) != 32 or any(c not in "0123456789abcdef" for c in job_id):
        raise HTTPException(status_code=404, detail="unknown job")
    directory = root / job_id
    with state_lock:
        try:
            status = json.loads((directory / "status.json").read_text("utf-8"))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="unknown job") from exc
    return directory, status


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "device": cfg.device}


@app.get("/v1/info")
def info() -> dict:
    return {
        "model": "whisperx/wav2vec2",
        "device": cfg.device,
        "loaded_languages": aligner.loaded_languages(),
        "default_language": cfg.default_language,
        "max_loaded_languages": 1,
        "capabilities": ["align"],
        "eviction": "park",
        "parked": aligner.is_parked(),
        "vram": aligner.vram_stats(),
    }


@app.post("/park")
async def park() -> dict:
    # ASS's eviction control plane. No auth: like /health and /v1/info these are
    # node-local orchestration calls, and ASS's park client sends no bearer. The
    # move is blocking GPU work, so keep it off the event loop. ASS only parks a
    # backend with no in-flight jobs, so this never races a running align.
    return await asyncio.to_thread(aligner.park)


@app.post("/unpark")
async def unpark() -> dict:
    return await asyncio.to_thread(aligner.unpark)


@app.post("/v1/align", status_code=202)
async def submit(
    audio: UploadFile = File(...),
    params: str = Form(...),
    authorization: str | None = Header(default=None),
) -> dict:
    return await enqueue(audio, params, authorization)


@app.post("/align")
async def align_synchronously(
    audio: UploadFile = File(...),
    params: str = Form(...),
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    return await enqueue(audio, params, authorization, synchronous=True)


async def enqueue(
    audio: UploadFile,
    params: str,
    authorization: str | None,
    *,
    synchronous: bool = False,
) -> dict | JSONResponse:
    check_auth(authorization)
    try:
        p = json.loads(params)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="params is not valid JSON") from exc
    if not isinstance(p, dict):
        raise HTTPException(status_code=400, detail="params must be an object")
    text = p.get("text")
    language = p.get("language", cfg.default_language)
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=400, detail="params.text is required")
    if not isinstance(language, str) or not language.strip():
        raise HTTPException(status_code=400, detail="params.language must be a string")
    if not slots.acquire(blocking=False):
        raise HTTPException(status_code=503, detail="alignment queue is full")
    directory = root / uuid4().hex
    submitted = False
    try:
        directory.mkdir()
        size = 0
        with (directory / "input.audio").open("wb") as target:
            while chunk := await audio.read(1024 * 1024):
                size += len(chunk)
                if size > cfg.max_upload_mb * 1024 * 1024:
                    raise HTTPException(
                        status_code=413, detail="audio upload too large"
                    )
                target.write(chunk)
        if not size:
            raise HTTPException(status_code=400, detail="audio file is empty")
        status = {"job_id": directory.name, "state": "queued", "artifacts": []}
        if not synchronous:
            save_status(directory, status)
        future = app.state.worker.submit(
            run_synchronous if synchronous else run_job,
            directory,
            text.strip(),
            language.strip(),
        )
        submitted = True
        if synchronous:
            # Cancellation must not cancel queued work and strand its upload or
            # slot. The worker finishes and cleans up even after disconnection.
            pending = asyncio.wrap_future(future)
            try:
                return await asyncio.shield(pending)
            except asyncio.CancelledError:
                # Consume a later failure when nobody is left to receive it.
                pending.add_done_callback(lambda done: done.exception())
                raise
        return status
    finally:
        if not submitted:
            shutil.rmtree(directory, ignore_errors=True)
            slots.release()
        await audio.close()


@app.get("/v1/jobs/{job_id}")
def status(job_id: str, authorization: str | None = Header(default=None)) -> dict:
    check_auth(authorization)
    return job_status(job_id)[1]


@app.get("/v1/jobs/{job_id}/result/{name}")
def download(
    job_id: str,
    name: str,
    authorization: str | None = Header(default=None),
) -> FileResponse:
    check_auth(authorization)
    directory, job = job_status(job_id)
    if name != "alignment.json":
        raise HTTPException(status_code=404, detail="unknown artifact")
    if job["state"] != "succeeded":
        raise HTTPException(status_code=409, detail="job has no result")
    return FileResponse(directory / name, media_type="application/json", filename=name)

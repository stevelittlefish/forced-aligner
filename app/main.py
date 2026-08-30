"""FastAPI surface for the forced aligner.

One real endpoint (/align) plus /health and /models. Multipart in, JSON out,
mirroring how Demucs is called on this LAN.
"""

from __future__ import annotations

import json
import logging
import tempfile
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from . import config
from .align import (
    Aligner,
    AudioDecodeError,
    LanguageUnsupported,
    supported_languages,
)

log = logging.getLogger("forced-aligner")

cfg = config.load()
aligner = Aligner(cfg)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Warm the configured models before serving, so the first /align isn't slow
    # (default: English). Best-effort: a download/load failure here is logged but
    # doesn't stop the service — /align will retry lazily and /models shows what
    # actually loaded, rather than a bad HF fetch killing the container.
    for lang in cfg.preload_languages:
        try:
            log.info("preloading alignment model for %r...", lang)
            aligner.preload(lang)
            log.info("preloaded %r", lang)
        except Exception:
            log.exception("failed to preload %r; it will load lazily instead", lang)
    yield


app = FastAPI(title="forced-aligner", version="0.1.0", lifespan=lifespan)


@app.exception_handler(HTTPException)
async def _http_error(_: Request, exc: HTTPException) -> JSONResponse:
    # The spec's error bodies are {"error": ...} (plus "supported" on 422), not
    # FastAPI's default {"detail": ...}. A dict detail is already in that shape
    # and passes through; a string detail gets wrapped.
    body = exc.detail if isinstance(exc.detail, dict) else {"error": exc.detail}
    return JSONResponse(status_code=exc.status_code, content=body)


def _check_auth(authorization: str | None) -> None:
    if not cfg.auth_token:
        return
    expected = f"Bearer {cfg.auth_token}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="bad or missing bearer token")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "device": cfg.device,
        "models_loaded": aligner.loaded_languages(),
    }


@app.get("/models")
def models() -> dict:
    return {
        "device": cfg.device,
        "loaded": aligner.loaded_languages(),
        "default_language": cfg.default_language,
    }


@app.post("/align")
async def align(
    audio: UploadFile = File(...),
    params: str = Form(...),
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    _check_auth(authorization)

    try:
        p = json.loads(params)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400, detail=f"params is not valid JSON: {e}"
        ) from e

    text = (p.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="params.text is required")
    language = (p.get("language") or cfg.default_language).strip()

    # WhisperX loads audio from a path, so buffer the upload to a temp file.
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        data = await audio.read()
        if not data:
            raise HTTPException(status_code=400, detail="audio file is empty")
        tmp.write(data)
        tmp.flush()

        try:
            result = aligner.align(tmp.name, text, language)
        except LanguageUnsupported as e:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": f"unsupported language {language!r}: {e}",
                    "supported": supported_languages(),
                },
            ) from e
        except AudioDecodeError as e:
            raise HTTPException(
                status_code=415, detail=f"audio could not be decoded: {e}"
            ) from e
        except Exception as e:  # model load or alignment failure
            raise HTTPException(
                status_code=500, detail=f"alignment failed: {e}"
            ) from e

    return JSONResponse(
        {
            "language": result.language,
            "duration": round(result.duration, 3),
            "lines": [asdict(line) for line in result.lines],
        }
    )

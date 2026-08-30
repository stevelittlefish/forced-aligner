"""The HTTP surface: routes, auth, request validation, and the spec's error
shapes. The aligner itself is stubbed — we assert the service's contract, not
wav2vec2's output.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.align import AlignResult, AudioDecodeError, LanguageUnsupported, Line, Word


def _client(main):
    return TestClient(main.app, raise_server_exceptions=False)


def _params(text="hello world", **kw):
    return json.dumps({"text": text, **kw})


def _fake_result():
    return AlignResult(
        language="en",
        duration=2.5,
        lines=[
            Line(
                text="hello world",
                start=0.0,
                end=2.0,
                words=[Word("hello", 0.0, 0.5), Word("world", 0.6, 2.0)],
            )
        ],
    )


def test_health(make_app):
    c = _client(make_app())
    r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["device"] == "cpu"
    assert body["models_loaded"] == []


def test_models(make_app):
    c = _client(make_app())
    r = c.get("/models")
    assert r.status_code == 200
    assert r.json()["default_language"] == "en"


def test_align_happy_path(make_app, monkeypatch):
    main = make_app()
    monkeypatch.setattr(main.aligner, "align", lambda *a, **k: _fake_result())
    c = _client(main)
    r = c.post(
        "/align",
        files={"audio": ("v.wav", b"RIFFfake", "audio/wav")},
        data={"params": _params(language="en")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["language"] == "en"
    assert body["duration"] == 2.5
    assert body["lines"][0]["words"][0]["text"] == "hello"


def test_align_passes_default_language_when_omitted(make_app, monkeypatch):
    main = make_app(default_language="de")
    seen = {}

    def fake(path, text, language):
        seen["language"] = language
        return _fake_result()

    monkeypatch.setattr(main.aligner, "align", fake)
    c = _client(main)
    c.post(
        "/align",
        files={"audio": ("v.wav", b"x", "audio/wav")},
        data={"params": _params()},
    )
    assert seen["language"] == "de"


def test_align_bad_params_json(make_app):
    c = _client(make_app())
    r = c.post(
        "/align",
        files={"audio": ("v.wav", b"x", "audio/wav")},
        data={"params": "{not json"},
    )
    assert r.status_code == 400
    assert "error" in r.json()


def test_align_empty_text(make_app):
    c = _client(make_app())
    r = c.post(
        "/align",
        files={"audio": ("v.wav", b"x", "audio/wav")},
        data={"params": _params(text="   ")},
    )
    assert r.status_code == 400
    assert "error" in r.json()


def test_align_empty_audio(make_app):
    c = _client(make_app())
    r = c.post(
        "/align",
        files={"audio": ("v.wav", b"", "audio/wav")},
        data={"params": _params()},
    )
    assert r.status_code == 400
    assert "error" in r.json()


def test_align_unsupported_language(make_app, monkeypatch):
    main = make_app()

    def boom(*a, **k):
        raise LanguageUnsupported("no model for 'xx'")

    monkeypatch.setattr(main.aligner, "align", boom)
    monkeypatch.setattr(main, "supported_languages", lambda: ["en", "de"])
    c = _client(main)
    r = c.post(
        "/align",
        files={"audio": ("v.wav", b"x", "audio/wav")},
        data={"params": _params(language="xx")},
    )
    assert r.status_code == 422
    body = r.json()
    assert "error" in body
    assert body["supported"] == ["en", "de"]


def test_align_undecodable_audio_is_415(make_app, monkeypatch):
    main = make_app()

    def boom(*a, **k):
        raise AudioDecodeError("ffmpeg: invalid data")

    monkeypatch.setattr(main.aligner, "align", boom)
    c = _client(main)
    r = c.post(
        "/align",
        files={"audio": ("v.txt", b"not audio", "text/plain")},
        data={"params": _params()},
    )
    assert r.status_code == 415
    assert "error" in r.json()


def test_align_internal_failure_is_500(make_app, monkeypatch):
    main = make_app()

    def boom(*a, **k):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(main.aligner, "align", boom)
    c = _client(main)
    r = c.post(
        "/align",
        files={"audio": ("v.wav", b"x", "audio/wav")},
        data={"params": _params()},
    )
    assert r.status_code == 500
    assert "error" in r.json()


def test_auth_required_when_token_set(make_app, monkeypatch):
    main = make_app(auth_token="s3cret")
    monkeypatch.setattr(main.aligner, "align", lambda *a, **k: _fake_result())
    c = _client(main)

    # No token -> 401.
    r = c.post(
        "/align",
        files={"audio": ("v.wav", b"x", "audio/wav")},
        data={"params": _params()},
    )
    assert r.status_code == 401

    # Correct token -> 200.
    r = c.post(
        "/align",
        headers={"Authorization": "Bearer s3cret"},
        files={"audio": ("v.wav", b"x", "audio/wav")},
        data={"params": _params()},
    )
    assert r.status_code == 200

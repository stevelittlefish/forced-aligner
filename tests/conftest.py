"""Test fixtures.

The point of these fixtures is to exercise the HTTP surface and the word/line
mapping *without* the heavy ML stack: no torch, no whisperx, no GPU, no model
download. `app.main` loads config and builds an Aligner at import time, so we
patch `config.load` before importing it and stub the one method that would touch
whisperx (`Aligner.align`) per test.
"""

from __future__ import annotations

import importlib
import sys

import pytest

from app import config as _config


def _cfg(**over) -> _config.Config:
    base = {
        "host": "127.0.0.1",
        "port": 8830,
        "auth_token": "",
        "device": "cpu",
        "compute_type": "int8",
        "default_language": "en",
        "model_cache_dir": "",
        "vad_trim": False,
        "preload_languages": (),  # never touch the ML stack in tests
    }
    base.update(over)
    return _config.Config(**base)


def _fresh_main(monkeypatch, cfg: _config.Config):
    """Import a fresh app.main bound to `cfg`. Re-imported per fixture so a token
    in one test doesn't leak into another (config is read at module import)."""
    monkeypatch.setattr(_config, "load", lambda path=None: cfg)
    sys.modules.pop("app.main", None)
    import app.main as main

    importlib.reload(main)
    return main


@pytest.fixture
def make_app(monkeypatch):
    def _make(**cfg_over):
        return _fresh_main(monkeypatch, _cfg(**cfg_over))

    return _make

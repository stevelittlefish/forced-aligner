"""Model cache and telemetry tested with a fake ML stack."""

import sys
from types import SimpleNamespace

from conftest import _cfg

from app.align import Aligner


def test_language_switch_releases_previous_model_before_loading(monkeypatch):
    aligner = Aligner(_cfg())
    loads = []
    releases = []

    def load(language_code, **kw):
        assert aligner._models == {}
        loads.append(language_code)
        return object(), {}

    monkeypatch.setitem(sys.modules, "whisperx", SimpleNamespace(load_align_model=load))
    monkeypatch.setattr(aligner, "_release_vram", lambda: releases.append(True))
    aligner.preload("en")
    aligner.preload("en")
    aligner.preload("de")
    assert loads == ["en", "de"]
    assert len(releases) == 2
    assert aligner.loaded_languages() == ["de"]


def test_gpu_telemetry(monkeypatch):
    cuda = SimpleNamespace(
        is_available=lambda: True,
        memory_allocated=lambda device: 1024**2 * 100,
        memory_reserved=lambda device: 1024**2 * 200,
        max_memory_allocated=lambda device: 1024**2 * 300,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    assert Aligner(_cfg(device="cuda")).vram_stats() == {
        "cuda": True,
        "device": "cuda",
        "allocated_mb": 100,
        "reserved_mb": 200,
        "peak_mb": 300,
    }

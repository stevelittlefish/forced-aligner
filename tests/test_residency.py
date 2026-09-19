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


class FakeModel:
    """Records every device it's moved to, so a test can prove where it lives."""

    def __init__(self):
        self.moves: list[str] = []

    def to(self, device):
        self.moves.append(str(device))
        return self


def _load_fake(aligner, monkeypatch):
    """Wire up whisperx so preload/_get_model produce FakeModels, no VRAM release."""
    def load(language_code, **kw):
        return FakeModel(), {}

    monkeypatch.setitem(sys.modules, "whisperx", SimpleNamespace(load_align_model=load))
    monkeypatch.setattr(aligner, "_release_vram", lambda: None)


def test_park_moves_model_to_cpu_and_unpark_brings_it_back(monkeypatch):
    aligner = Aligner(_cfg(device="cuda"))
    _load_fake(aligner, monkeypatch)
    aligner.preload("en")
    model = aligner._models["en"][0]

    assert aligner.is_parked() is False
    res = aligner.park()
    assert res["parked"] is True and res["languages"] == ["en"]
    assert aligner.is_parked() is True
    assert model.moves[-1] == "cpu"

    res = aligner.unpark()
    assert res["unparked"] is True
    assert aligner.is_parked() is False
    assert model.moves[-1] == "cuda"


def test_park_is_a_noop_off_cuda(monkeypatch):
    aligner = Aligner(_cfg(device="cpu"))
    _load_fake(aligner, monkeypatch)
    aligner.preload("en")
    res = aligner.park()
    assert res["parked"] is False
    assert aligner.is_parked() is False
    assert aligner._models["en"][0].moves == []  # never touched


def test_get_model_restores_a_parked_model_before_use(monkeypatch):
    # ASS should unpark first, but a stray request must never run on a CPU model.
    aligner = Aligner(_cfg(device="cuda"))
    _load_fake(aligner, monkeypatch)
    aligner.preload("en")
    model = aligner._models["en"][0]
    aligner.park()
    assert model.moves[-1] == "cpu"

    aligner._get_model("en")  # the call align() makes right before inference
    assert aligner.is_parked() is False
    assert model.moves[-1] == "cuda"


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

"""Load config.toml. No environment variables — the file is the only source.

tomllib is stdlib from Python 3.11, so config costs no dependency.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config.toml"


@dataclass(frozen=True)
class Config:
    host: str
    port: int
    auth_token: str
    device: str
    compute_type: str
    default_language: str
    model_cache_dir: str
    vad_trim: bool


def load(path: Path | None = None) -> Config:
    p = path or DEFAULT_PATH
    if not p.exists():
        raise SystemExit(
            f"No config at {p}. Copy config.example.toml to config.toml and edit it."
        )
    with p.open("rb") as f:
        raw = tomllib.load(f)

    server = raw.get("server", {})
    align = raw.get("align", {})
    return Config(
        host=server.get("host", "0.0.0.0"),
        port=int(server.get("port", 8830)),
        auth_token=server.get("auth_token", ""),
        device=align.get("device", "cuda"),
        compute_type=align.get("compute_type", "float16"),
        default_language=align.get("default_language", "en"),
        model_cache_dir=align.get("model_cache_dir", ""),
        vad_trim=bool(align.get("vad_trim", True)),
    )

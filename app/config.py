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
    # Languages to load at startup instead of lazily on first request, so the
    # first /align isn't slow. A tuple because Config is frozen/hashable.
    preload_languages: tuple[str, ...]


def load(path: Path | None = None) -> Config:
    p = path or DEFAULT_PATH
    if not p.exists():
        raise SystemExit(
            f"No config at {p}. Copy config.example.toml to config.toml and edit it."
        )
    if p.is_dir():
        # A bind mount to a path that didn't exist on the host makes Docker
        # create it as an empty directory — so the file the container expects is
        # a dir and tomllib chokes. Say what actually happened, not IsADirectory.
        raise SystemExit(
            f"{p} is a directory, not a file. This usually means a Docker bind "
            f"mount pointed at a host path that didn't exist yet, so Docker "
            f"created it as a directory. Remove it and put the real config file "
            f"there before starting the container."
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
        preload_languages=tuple(align.get("preload_languages", ["en"])),
    )

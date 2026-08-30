"""Config loading, including the Docker bind-mount foot-gun where a missing host
path gets created as a directory."""

from __future__ import annotations

import pytest

from app import config


def test_missing_config_exits_with_hint(tmp_path):
    with pytest.raises(SystemExit) as e:
        config.load(tmp_path / "nope.toml")
    assert "config.example.toml" in str(e.value)


def test_config_dir_reports_bind_mount_cause(tmp_path):
    # Simulate Docker creating the mount target as a directory.
    d = tmp_path / "config.toml"
    d.mkdir()
    with pytest.raises(SystemExit) as e:
        config.load(d)
    assert "directory" in str(e.value)


def test_load_reads_values(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(
        '[server]\nport = 9000\nauth_token = "t"\n'
        '[align]\ndevice = "cpu"\nvad_trim = false\n'
    )
    cfg = config.load(p)
    assert cfg.port == 9000
    assert cfg.auth_token == "t"
    assert cfg.device == "cpu"
    assert cfg.vad_trim is False
    assert cfg.default_language == "en"  # default when omitted

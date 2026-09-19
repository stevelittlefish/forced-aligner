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
    assert cfg.preload_languages == ("en",)  # defaults to warming English


def test_preload_languages_parsed_as_tuple(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text('[align]\npreload_languages = ["en", "de"]\n')
    assert config.load(p).preload_languages == ("en", "de")


def test_preload_can_be_disabled(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[align]\npreload_languages = []\n")
    assert config.load(p).preload_languages == ()


@pytest.mark.parametrize("key", ["max_pending_jobs", "max_upload_mb"])
def test_job_limits_must_be_positive(tmp_path, key):
    path = tmp_path / "config.toml"
    path.write_text(f"[jobs]\n{key} = 0\n")
    with pytest.raises(ValueError, match=key):
        config.load(path)


def test_job_settings(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        '[jobs]\ndirectory = "/tmp/jobs"\n'
        'max_pending_jobs = 2\nmax_upload_mb = 128\n'
    )
    cfg = config.load(path)
    assert (cfg.jobs_dir, cfg.max_pending_jobs, cfg.max_upload_mb) == (
        "/tmp/jobs", 2, 128,
    )

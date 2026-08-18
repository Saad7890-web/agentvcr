from __future__ import annotations

from pathlib import Path

import pytest

from agentvcr.config import (
    DEFAULT_UPSTREAMS,
    ConfigError,
    Settings,
    load_settings,
    with_overrides,
)


def test_defaults(tmp_path: Path) -> None:
    settings = load_settings(environ={}, search_from=tmp_path)
    assert settings.host == "127.0.0.1"
    assert settings.port == 8484
    assert settings.mode == "record"
    assert settings.mismatch_policy == "warn"
    assert settings.db_path == Path(".agentvcr") / "agentvcr.db"
    assert settings.upstreams == DEFAULT_UPSTREAMS


def test_file_env_and_flags_layer_in_order(tmp_path: Path) -> None:
    (tmp_path / "agentvcr.toml").write_text(
        "\n".join(
            [
                "[agentvcr]",
                "port = 9000",
                "mode = 'replay'",
                "db_path = 'tapes/one.db'",
                "preset = 'groq'",
            ]
        )
    )
    from_file = load_settings(environ={}, search_from=tmp_path)
    assert from_file.port == 9000
    assert from_file.mode == "replay"
    assert from_file.db_path == Path("tapes/one.db")
    assert from_file.upstream_for("openai") == "https://api.groq.com/openai/v1"
    # a preset only overrides the formats it names
    assert from_file.upstream_for("anthropic") == DEFAULT_UPSTREAMS["anthropic"]

    from_env = load_settings(
        environ={"AGENTVCR_PORT": "9100", "AGENTVCR_UPSTREAM_OPENAI": "http://localhost:11434/v1"},
        search_from=tmp_path,
    )
    assert from_env.port == 9100
    assert from_env.mode == "replay"  # still from the file
    assert from_env.upstream_for("openai") == "http://localhost:11434/v1"

    from_flags = load_settings(
        environ={"AGENTVCR_PORT": "9100"},
        overrides={"port": 9200, "host": None},
        search_from=tmp_path,
    )
    assert from_flags.port == 9200
    assert from_flags.host == "127.0.0.1"  # None overrides are ignored


def test_config_file_is_found_by_walking_up(tmp_path: Path) -> None:
    (tmp_path / "agentvcr.toml").write_text("[agentvcr]\nport = 9300\n")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    settings = load_settings(environ={}, search_from=nested)
    assert settings.port == 9300
    assert settings.config_path == tmp_path / "agentvcr.toml"


@pytest.mark.parametrize(
    "overrides",
    [{"mode": "rewind"}, {"mismatch_policy": "shout"}, {"port": 0}],
)
def test_invalid_values_are_rejected(tmp_path: Path, overrides: dict) -> None:
    with pytest.raises(ConfigError):
        load_settings(environ={}, overrides=overrides, search_from=tmp_path)


def test_unknown_preset_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_settings(environ={"AGENTVCR_PRESET": "nope"}, search_from=tmp_path)


def test_with_overrides_validates() -> None:
    settings = Settings()
    assert with_overrides(settings, port=1234, mode=None).port == 1234
    with pytest.raises(ConfigError):
        with_overrides(settings, mode="nope")


def test_upstream_for_unknown_provider() -> None:
    with pytest.raises(ConfigError):
        Settings().upstream_for("cohere")

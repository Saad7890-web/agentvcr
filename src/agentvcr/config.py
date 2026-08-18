"""Settings resolution: defaults < agentvcr.toml < environment < CLI flags.

A single :class:`Settings` object is threaded through the CLI, the server app and the
core modules, so nothing else has to know where a value came from.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

try:  # Python >= 3.11
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on 3.10 only
    import tomli as tomllib  # type: ignore[no-redef]

CONFIG_FILENAME = "agentvcr.toml"
DEFAULT_DB_DIR = ".agentvcr"
DEFAULT_DB_NAME = "agentvcr.db"

MODES = ("record", "replay", "fork", "passthrough")
MISMATCH_POLICIES = ("warn", "strict", "live-on-miss")

#: Wire formats the proxy speaks. The path prefix (``/openai``, ``/anthropic``) picks
#: the format; the upstream host behind it is configuration.
PROVIDERS = ("openai", "anthropic")

DEFAULT_UPSTREAMS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
}

#: One-flag upstream bundles so free-tier demos need no config file.
PRESETS: dict[str, dict[str, str]] = {
    "openai": {"openai": "https://api.openai.com/v1"},
    "anthropic": {"anthropic": "https://api.anthropic.com"},
    "groq": {"openai": "https://api.groq.com/openai/v1"},
    "gemini": {"openai": "https://generativelanguage.googleapis.com/v1beta/openai"},
}


class ConfigError(ValueError):
    """Raised when a config file, env var or flag holds a value we cannot honor."""


@dataclass(frozen=True)
class Settings:
    """Fully resolved runtime configuration."""

    host: str = "127.0.0.1"
    port: int = 8484
    db_path: Path = field(default_factory=lambda: Path(DEFAULT_DB_DIR) / DEFAULT_DB_NAME)
    mode: str = "record"
    mismatch_policy: str = "warn"
    upstreams: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_UPSTREAMS))
    #: Seconds of silence after which an implicitly-grouped run is considered finished.
    idle_timeout_s: float = 300.0
    #: Store the raw SSE chunk tape alongside the accumulated response.
    record_chunks: bool = True
    config_path: Path | None = None

    def upstream_for(self, provider: str) -> str:
        """Base URL to forward ``provider``-format requests to."""
        try:
            return self.upstreams[provider]
        except KeyError:
            raise ConfigError(
                f"unknown provider {provider!r}; expected one of {PROVIDERS}"
            ) from None

    def validate(self) -> Settings:
        if self.mode not in MODES:
            raise ConfigError(f"invalid mode {self.mode!r}; expected one of {MODES}")
        if self.mismatch_policy not in MISMATCH_POLICIES:
            raise ConfigError(
                f"invalid mismatch policy {self.mismatch_policy!r}; "
                f"expected one of {MISMATCH_POLICIES}"
            )
        if not 0 < self.port < 65536:
            raise ConfigError(f"invalid port {self.port!r}")
        unknown = set(self.upstreams) - set(PROVIDERS)
        if unknown:
            raise ConfigError(f"unknown upstream provider(s): {sorted(unknown)}")
        return self


def find_config_file(start: Path | None = None) -> Path | None:
    """Nearest ``agentvcr.toml`` walking up from ``start`` (default: cwd)."""
    here = (start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        candidate = directory / CONFIG_FILENAME
        if candidate.is_file():
            return candidate
    return None


def _from_file(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    section = raw.get("agentvcr", raw)
    values: dict[str, Any] = {}
    for key in (
        "host",
        "port",
        "mode",
        "mismatch_policy",
        "idle_timeout_s",
        "record_chunks",
    ):
        if key in section:
            values[key] = section[key]
    if "db_path" in section:
        values["db_path"] = Path(section["db_path"])
    if "preset" in section:
        values["upstreams"] = _preset_upstreams(section["preset"])
    upstreams = section.get("upstreams")
    if isinstance(upstreams, dict):
        values.setdefault("upstreams", {})
        values["upstreams"] = {**values.get("upstreams", {}), **upstreams}
    return values


def _from_env(environ: dict[str, str]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if "AGENTVCR_HOST" in environ:
        values["host"] = environ["AGENTVCR_HOST"]
    if "AGENTVCR_PORT" in environ:
        values["port"] = _int(environ["AGENTVCR_PORT"], "AGENTVCR_PORT")
    if "AGENTVCR_DB" in environ:
        values["db_path"] = Path(environ["AGENTVCR_DB"])
    if "AGENTVCR_MODE" in environ:
        values["mode"] = environ["AGENTVCR_MODE"]
    if "AGENTVCR_MISMATCH_POLICY" in environ:
        values["mismatch_policy"] = environ["AGENTVCR_MISMATCH_POLICY"]
    if "AGENTVCR_IDLE_TIMEOUT" in environ:
        values["idle_timeout_s"] = _float(environ["AGENTVCR_IDLE_TIMEOUT"], "AGENTVCR_IDLE_TIMEOUT")
    if "AGENTVCR_RECORD_CHUNKS" in environ:
        values["record_chunks"] = _bool(environ["AGENTVCR_RECORD_CHUNKS"], "AGENTVCR_RECORD_CHUNKS")
    if "AGENTVCR_PRESET" in environ:
        values["upstreams"] = _preset_upstreams(environ["AGENTVCR_PRESET"])
    for provider in PROVIDERS:
        key = f"AGENTVCR_UPSTREAM_{provider.upper()}"
        if key in environ:
            values.setdefault("upstreams", {})
            values["upstreams"] = {**values["upstreams"], provider: environ[key]}
    return values


def _preset_upstreams(name: str) -> dict[str, str]:
    try:
        return dict(PRESETS[name])
    except KeyError:
        raise ConfigError(f"unknown preset {name!r}; expected one of {sorted(PRESETS)}") from None


def _int(value: str, label: str) -> int:
    try:
        return int(value)
    except ValueError:
        raise ConfigError(f"{label} must be an integer, got {value!r}") from None


def _bool(value: str, label: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{label} must be a boolean, got {value!r}")


def _float(value: str, label: str) -> float:
    try:
        return float(value)
    except ValueError:
        raise ConfigError(f"{label} must be a number, got {value!r}") from None


def load_settings(
    *,
    config_path: Path | None = None,
    environ: dict[str, str] | None = None,
    overrides: dict[str, Any] | None = None,
    search_from: Path | None = None,
) -> Settings:
    """Resolve settings from all layers.

    ``overrides`` carries CLI flags and wins over everything; ``None`` values in it are
    ignored so callers can pass optional flags straight through.
    """
    environ = os.environ if environ is None else environ
    path = config_path or find_config_file(search_from)

    values: dict[str, Any] = {}
    if path is not None:
        values.update(_from_file(path))
    _merge(values, _from_env(dict(environ)))
    _merge(values, {k: v for k, v in (overrides or {}).items() if v is not None})

    upstreams = {**DEFAULT_UPSTREAMS, **values.pop("upstreams", {})}
    settings = Settings(upstreams=upstreams, config_path=path, **values)
    return settings.validate()


def _merge(base: dict[str, Any], incoming: dict[str, Any]) -> None:
    """Layer ``incoming`` onto ``base``, merging the upstream map instead of replacing it."""
    for key, value in incoming.items():
        if key == "upstreams" and isinstance(value, dict):
            base["upstreams"] = {**base.get("upstreams", {}), **value}
        else:
            base[key] = value


def with_overrides(settings: Settings, **overrides: Any) -> Settings:
    """Return a copy of ``settings`` with non-``None`` fields replaced."""
    clean = {k: v for k, v in overrides.items() if v is not None}
    return replace(settings, **clean).validate()

"""Settings for the Kemory provider.

The API key, URL and organisation pin come from Hermes's secret scope; everything else comes from
$HERMES_HOME/kemory.json. No other file is read.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from agent.secret_scope import get_secret

DEFAULT_BASE_URL = "https://api.kemory.s9n.ai"
CONFIG_FILENAME = "kemory.json"
CONFIG_KEYS = (
    "base_url",
    "org_id",
    "auto_recall",
    "auto_capture",
    "gateways",
    "max_recall_results",
    "recall_timeout",
    "write_timeout",
    "capture_min_new_prompts",
    "context_max_chars",
)

API_KEY_ENV = "KEMORY_API_KEY"
URL_ENV = "KEMORY_URL"
ORG_ID_ENV = "KEMORY_ORG_ID"

# Plain http is only for a Kemory server on the same machine.
_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost"})


class ConfigError(Exception):
    """The provider cannot be configured; the message is meant for the user."""


@dataclass(frozen=True)
class KemoryConfig:
    api_key: str = field(repr=False)
    base_url: str
    org_pin: str | None
    auto_recall: bool = True
    auto_capture: bool = True
    gateways: bool = False
    max_recall_results: int = 6
    recall_timeout: float = 3.0
    write_timeout: float = 20.0
    capture_min_new_prompts: int = 3
    context_max_chars: int = 4000


def read_config_file(hermes_home: str) -> dict[str, Any]:
    """Return the parsed kemory.json object, or {} when it is missing or not a JSON object."""
    try:
        parsed = json.loads((Path(hermes_home) / CONFIG_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return parsed


def load_config(hermes_home: str) -> KemoryConfig:
    """Build the configuration for one Hermes home; raises only ConfigError."""
    api_key = _secret(API_KEY_ENV)
    if not api_key:
        raise ConfigError(
            f"{API_KEY_ENV} is not set. Create a Kemory API key and set {API_KEY_ENV}."
        )

    file_values = read_config_file(hermes_home)
    base_url = _base_url(_secret(URL_ENV) or _text(file_values.get("base_url")) or DEFAULT_BASE_URL)
    org_pin = _secret(ORG_ID_ENV) or _text(file_values.get("org_id")) or None

    return KemoryConfig(
        api_key=api_key,
        base_url=base_url,
        org_pin=org_pin,
        auto_recall=_flag(file_values, "auto_recall", True),
        auto_capture=_flag(file_values, "auto_capture", True),
        gateways=_flag(file_values, "gateways", False),
        max_recall_results=_integer(file_values, "max_recall_results", 6, 1, 20),
        recall_timeout=_number(file_values, "recall_timeout", 3.0, 0.5, 7.5),
        write_timeout=_number(file_values, "write_timeout", 20.0, 1.0, 60.0),
        capture_min_new_prompts=_integer(file_values, "capture_min_new_prompts", 3, 1, 12),
        context_max_chars=_integer(file_values, "context_max_chars", 4000, 500, None),
    )


def _secret(name: str) -> str:
    """Read a setting through Hermes's secret scope, which is the only way a plugin may read one."""
    try:
        value = get_secret(name)
    except Exception as exc:  # the scope raises when it is unbound under a multiplexed gateway
        raise ConfigError(f"{name} could not be read from Hermes's secret scope: {exc}") from exc
    return value.strip() if isinstance(value, str) else ""


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _base_url(raw: str) -> str:
    url = raw.removesuffix("/")
    try:
        parts = urlsplit(url)
        host = parts.hostname
        _ = parts.port  # raises ValueError for a malformed port
    except ValueError:
        host = None
    if host and (parts.scheme == "https" or (parts.scheme == "http" and host in _LOCAL_HOSTS)):
        return url
    raise ConfigError(
        f"The Kemory URL must start with https:// (http:// is allowed only for 127.0.0.1 and "
        f"localhost). Check {URL_ENV} and base_url in {CONFIG_FILENAME}."
    )


def _flag(values: dict[str, Any], key: str, default: bool) -> bool:
    value = values.get(key)
    return value if isinstance(value, bool) else default


def _integer(values: dict[str, Any], key: str, default: int, low: int, high: int | None) -> int:
    value = values.get(key)
    # bool is a subclass of int but is never a meaningful count.
    if not isinstance(value, int) or isinstance(value, bool):
        return default
    value = max(low, value)
    return value if high is None else min(high, value)


def _number(values: dict[str, Any], key: str, default: float, low: float, high: float) -> float:
    value = values.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        return default
    if isinstance(value, float) and not math.isfinite(value):
        return default
    # Clamp before converting so an enormous integer cannot overflow float().
    return float(min(high, max(low, value)))

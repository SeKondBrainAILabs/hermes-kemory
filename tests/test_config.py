import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest

from hermes_kemory.config import (
    CONFIG_FILENAME,
    CONFIG_KEYS,
    DEFAULT_BASE_URL,
    ConfigError,
    KemoryConfig,
    load_config,
    read_config_file,
)

API_KEY = "kmk_test_0123456789"


@pytest.fixture(autouse=True)
def api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KEMORY_API_KEY", API_KEY)


def write_config(home: Path, content: Any) -> None:
    text = content if isinstance(content, str) else json.dumps(content)
    (home / CONFIG_FILENAME).write_text(text)


def test_constants() -> None:
    assert DEFAULT_BASE_URL == "https://api.kemory.s9n.ai"
    assert CONFIG_FILENAME == "kemory.json"
    assert CONFIG_KEYS == (
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


def test_defaults(hermes_home: Path) -> None:
    config = load_config(str(hermes_home))
    assert config == KemoryConfig(api_key=API_KEY, base_url=DEFAULT_BASE_URL, org_pin=None)
    assert config.auto_recall is True
    assert config.auto_capture is True
    assert config.gateways is False
    assert config.max_recall_results == 6
    assert config.recall_timeout == 3.0
    assert config.write_timeout == 20.0
    assert config.capture_min_new_prompts == 3
    assert config.context_max_chars == 4000


def test_url_precedence_env_then_file_then_default(
    hermes_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert load_config(str(hermes_home)).base_url == DEFAULT_BASE_URL

    write_config(hermes_home, {"base_url": "https://file.example.com/"})
    assert load_config(str(hermes_home)).base_url == "https://file.example.com"

    monkeypatch.setenv("KEMORY_URL", "https://env.example.com//")
    # Only one trailing slash is removed.
    assert load_config(str(hermes_home)).base_url == "https://env.example.com/"

    monkeypatch.setenv("KEMORY_URL", "")
    assert load_config(str(hermes_home)).base_url == "https://file.example.com"


@pytest.mark.parametrize(
    "url",
    [
        "http://api.example.com",
        "http://localhost.example.com",
        "http://127.0.0.1.example.com",
        "ftp://example.com",
        "api.example.com",
        "file:///etc/passwd",
    ],
)
def test_rejects_non_https_except_localhost(
    url: str, hermes_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KEMORY_URL", url)
    with pytest.raises(ConfigError, match="https"):
        load_config(str(hermes_home))


@pytest.mark.parametrize(
    "url",
    [
        "https://api.example.com",
        "http://127.0.0.1",
        "http://127.0.0.1:8080",
        "http://localhost",
        "http://localhost:9000",
        "http://localhost:9000/",
    ],
)
def test_accepts_https_and_local_http(
    url: str, hermes_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KEMORY_URL", url)
    assert load_config(str(hermes_home)).base_url == url.removesuffix("/")


def test_file_url_is_checked_too(hermes_home: Path) -> None:
    write_config(hermes_home, {"base_url": "http://example.com"})
    with pytest.raises(ConfigError, match="https"):
        load_config(str(hermes_home))


@pytest.mark.parametrize("value", [None, ""])
def test_missing_key_raises_config_error_naming_env_var(
    value: str | None, hermes_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if value is None:
        monkeypatch.delenv("KEMORY_API_KEY")
    else:
        monkeypatch.setenv("KEMORY_API_KEY", value)
    with pytest.raises(ConfigError, match="KEMORY_API_KEY"):
        load_config(str(hermes_home))


def test_org_pin_precedence_env_then_file(
    hermes_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert load_config(str(hermes_home)).org_pin is None

    write_config(hermes_home, {"org_id": "org-file"})
    assert load_config(str(hermes_home)).org_pin == "org-file"

    monkeypatch.setenv("KEMORY_ORG_ID", "org-env")
    assert load_config(str(hermes_home)).org_pin == "org-env"

    # An empty value means no pin, and the empty env value does not hide the file.
    monkeypatch.setenv("KEMORY_ORG_ID", "")
    assert load_config(str(hermes_home)).org_pin == "org-file"

    write_config(hermes_home, {"org_id": ""})
    assert load_config(str(hermes_home)).org_pin is None


def test_numeric_settings_clamped_and_bad_types_default(hermes_home: Path) -> None:
    write_config(
        hermes_home,
        {
            "max_recall_results": 500,
            "recall_timeout": 60,
            "write_timeout": 0.1,
            "capture_min_new_prompts": 0,
            "context_max_chars": 10,
        },
    )
    config = load_config(str(hermes_home))
    assert config.max_recall_results == 20
    assert config.recall_timeout == 7.5
    assert config.write_timeout == 1.0
    assert config.capture_min_new_prompts == 1
    assert config.context_max_chars == 500

    write_config(
        hermes_home,
        {
            "max_recall_results": -4,
            "recall_timeout": 0,
            "write_timeout": 1000,
            "capture_min_new_prompts": 99,
        },
    )
    config = load_config(str(hermes_home))
    assert config.max_recall_results == 1
    assert config.recall_timeout == 0.5
    assert config.write_timeout == 60.0
    assert config.capture_min_new_prompts == 12

    write_config(
        hermes_home,
        {
            "max_recall_results": "abc",
            "recall_timeout": [1],
            "write_timeout": True,
            "capture_min_new_prompts": {"a": 1},
            "context_max_chars": None,
            "auto_recall": "yes",
            "auto_capture": 1,
            "gateways": "true",
        },
    )
    config = load_config(str(hermes_home))
    assert config == KemoryConfig(api_key=API_KEY, base_url=DEFAULT_BASE_URL, org_pin=None)


def test_numeric_types_are_kept(hermes_home: Path) -> None:
    write_config(
        hermes_home,
        {"max_recall_results": 9, "recall_timeout": 2.5, "write_timeout": 30},
    )
    config = load_config(str(hermes_home))
    assert config.max_recall_results == 9
    assert isinstance(config.max_recall_results, int)
    assert config.recall_timeout == 2.5
    assert config.write_timeout == 30.0
    assert isinstance(config.write_timeout, float)


def test_boolean_settings(hermes_home: Path) -> None:
    write_config(hermes_home, {"auto_recall": False, "auto_capture": False, "gateways": True})
    config = load_config(str(hermes_home))
    assert config.auto_recall is False
    assert config.auto_capture is False
    assert config.gateways is True


@pytest.mark.parametrize("content", ["{not json", "[1, 2]", '"text"', "null", "42", ""])
def test_bad_config_file_falls_back_to_defaults(hermes_home: Path, content: str) -> None:
    write_config(hermes_home, content)
    assert read_config_file(str(hermes_home)) == {}
    assert load_config(str(hermes_home)) == KemoryConfig(
        api_key=API_KEY, base_url=DEFAULT_BASE_URL, org_pin=None
    )


def test_deeply_nested_config_file_gives_defaults(hermes_home: Path) -> None:
    write_config(hermes_home, "[" * 200_000 + "]" * 200_000)
    assert read_config_file(str(hermes_home)) == {}
    assert load_config(str(hermes_home)) == KemoryConfig(
        api_key=API_KEY, base_url=DEFAULT_BASE_URL, org_pin=None
    )


def test_missing_config_file_and_directory(hermes_home: Path, tmp_path: Path) -> None:
    assert read_config_file(str(hermes_home)) == {}
    assert read_config_file(str(tmp_path / "does-not-exist")) == {}


def test_config_file_that_is_a_directory_gives_defaults(hermes_home: Path) -> None:
    (hermes_home / CONFIG_FILENAME).mkdir()
    assert read_config_file(str(hermes_home)) == {}


def test_read_config_file_returns_the_object_and_unknown_keys_are_ignored(
    hermes_home: Path,
) -> None:
    write_config(hermes_home, {"gateways": True, "something_else": [1]})
    assert read_config_file(str(hermes_home)) == {
        "gateways": True,
        "something_else": [1],
    }
    assert load_config(str(hermes_home)).gateways is True


def test_repr_hides_api_key() -> None:
    config = KemoryConfig(api_key=API_KEY, base_url=DEFAULT_BASE_URL, org_pin=None)
    assert API_KEY not in repr(config)
    assert API_KEY not in str(config)
    assert "api_key" not in repr(config)


def test_config_is_frozen() -> None:
    config = KemoryConfig(api_key=API_KEY, base_url=DEFAULT_BASE_URL, org_pin=None)
    with pytest.raises(AttributeError):
        config.gateways = True  # type: ignore[misc]


def test_config_keys_name_config_fields() -> None:
    names = {f.name for f in fields(KemoryConfig)}
    # org_id in the file is the org_pin field; every other file key is a field of the same name.
    assert {"org_pin" if key == "org_id" else key for key in CONFIG_KEYS} <= names


def test_secret_scope_failure_becomes_config_error(
    hermes_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(name: str, default: str | None = None) -> str | None:
        raise RuntimeError("no scope")

    monkeypatch.setattr("hermes_kemory.config.get_secret", boom)
    with pytest.raises(ConfigError, match="KEMORY_API_KEY"):
        load_config(str(hermes_home))


def test_cli_login_file_is_not_used(
    hermes_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other_home = tmp_path / "user-home"
    (other_home / ".kemory").mkdir(parents=True)
    (other_home / ".kemory" / "credentials-prod").write_text('{"api_key": "from-cli-login"}')
    monkeypatch.setenv("HOME", str(other_home))
    monkeypatch.delenv("KEMORY_API_KEY")
    with pytest.raises(ConfigError):
        load_config(str(hermes_home))

"""Shared fixtures. The autouse fixture keeps a developer's own setup out of every test."""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from .fake_kemory import FakeKemory


@pytest.fixture(autouse=True)
def hermes_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point HERMES_HOME at an empty tmp dir and drop every Kemory setting from the environment."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for name in list(os.environ):
        if name.startswith("KEMORY_"):
            monkeypatch.delenv(name)
    # The Hermes checkout has a plugins/ dir that Hermes would scan as project plugins.
    monkeypatch.delenv("HERMES_ENABLE_PROJECT_PLUGINS", raising=False)
    yield home


@pytest.fixture(autouse=True)
def no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests talk to a fake server on 127.0.0.1; a developer's proxy must not intercept that."""
    for name in ("http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.upper(), raising=False)


@pytest.fixture
def fake_kemory() -> Iterator[FakeKemory]:
    with FakeKemory() as server:
        yield server

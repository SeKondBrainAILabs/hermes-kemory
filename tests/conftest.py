"""Shared fixtures. The autouse fixture keeps a developer's own setup out of every test."""

import json
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from agent.memory_manager import MemoryManager

from hermes_kemory.provider import KemoryMemoryProvider

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


API_KEY = "kmk_test_0123456789abcdef"


@dataclass
class Running:
    """A provider driven the way Hermes drives it: through a MemoryManager."""

    manager: MemoryManager
    provider: KemoryMemoryProvider
    server: FakeKemory
    home: Path

    def prefetch(self, query: str) -> str:
        return self.manager.prefetch_all(query, session_id="s1")

    def tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        result = json.loads(self.manager.handle_tool_call(name, args))
        assert isinstance(result, dict)
        return result


StartProvider = Callable[..., Running]


@pytest.fixture
def start_provider(
    fake_kemory: FakeKemory, hermes_home: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[StartProvider]:
    """Start a provider against the fake server, with settings for kemory.json as keywords.

    platform and agent_context are what Hermes would pass to initialize(); key=None leaves the
    API key unset, initialize=False only registers the provider, and url replaces the fake
    server's address.
    """
    managers: list[MemoryManager] = []

    def start(
        *,
        platform: str | None = "cli",
        agent_context: str | None = "primary",
        key: str | None = API_KEY,
        initialize: bool = True,
        url: str | None = None,
        settings: dict[str, Any] | None = None,
        **config: Any,
    ) -> Running:
        if key is not None:
            monkeypatch.setenv("KEMORY_API_KEY", key)
        monkeypatch.setenv("KEMORY_URL", url or fake_kemory.url)
        values = {**(settings or {}), **config}
        if values:
            (hermes_home / "kemory.json").write_text(json.dumps(values))
        provider = KemoryMemoryProvider()
        manager = MemoryManager()
        managers.append(manager)
        manager.add_provider(provider)
        if initialize:
            kwargs: dict[str, Any] = {"hermes_home": str(hermes_home)}
            if platform is not None:
                kwargs["platform"] = platform
            if agent_context is not None:
                kwargs["agent_context"] = agent_context
            manager.initialize_all(session_id="s1", **kwargs)
        return Running(manager, provider, fake_kemory, hermes_home)

    yield start
    for manager in managers:
        manager.shutdown_all()

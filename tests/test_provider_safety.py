"""Safety: Kemory trouble never reaches the agent, and the provider is off where it should be."""

import builtins
import json
import logging
import os
import stat
import time
from pathlib import Path
from typing import Any

import pytest

import hermes_kemory
from hermes_kemory.provider import (
    LOCAL_PLATFORMS,
    NO_CAPTURE_CONTEXTS,
    RECALL_ONLY_PLATFORMS,
    KemoryMemoryProvider,
)

from .conftest import API_KEY, StartProvider
from .fake_kemory import KEY_ORG_ID, FakeKemory

ME_ORGS = "/api/v1/me/orgs"
SEARCH = "/api/v1/memories/search"
CONTEXT = "/api/v1/user/context"
HIT = {"items": [{"memory_id": "m1", "namespace": "user:profile", "content": "Likes tea."}]}
TOOL_CALLS: list[tuple[str, dict[str, Any]]] = [
    ("kemory_search", {"query": "tea"}),
    ("kemory_store", {"content": "Likes tea"}),
    ("kemory_forget", {"memory_id": "m1"}),
    ("kemory_context", {}),
]
TOOL_NAMES = {name for name, _ in TOOL_CALLS}


def bodies_of(server: FakeKemory, method: str, path: str) -> list[Any]:
    return [r for r in server.requests if r.method == method and r.path.startswith(path)]


def test_missing_key_unavailable_and_no_requests(
    start_provider: StartProvider, fake_kemory: FakeKemory, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    running = start_provider(key=None)

    assert running.provider.is_available() is False
    assert "KEMORY_API_KEY" in running.provider.unavailable_reason()
    assert running.prefetch("what do I like?") == ""
    for name, args in TOOL_CALLS:
        result = running.tool(name, args)
        assert "KEMORY_API_KEY" in result["error"]
    assert running.provider.get_tool_schemas() == []
    assert running.provider.system_prompt_block() == ""
    assert running.provider.recall_status() is None
    assert fake_kemory.requests == []
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len([r for r in warnings if "KEMORY_API_KEY" in r.getMessage()]) == 1


def test_available_with_a_key_without_any_request_or_file_read(
    start_provider: StartProvider,
    fake_kemory: FakeKemory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    running = start_provider(key=API_KEY, initialize=False)
    opened: list[object] = []
    real_open = builtins.open

    def spy(file: Any, *args: Any, **kwargs: Any) -> Any:
        opened.append(file)
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy)
    monkeypatch.setattr(Path, "read_text", lambda *a, **k: opened.append(a) or "")

    assert running.provider.is_available() is True
    assert running.provider.unavailable_reason() == ""

    assert opened == []
    assert fake_kemory.requests == []


def test_a_bad_url_leaves_the_provider_off_and_logged_once(
    start_provider: StartProvider,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    running = start_provider(initialize=False)
    monkeypatch.setenv("KEMORY_URL", "http://example.com")
    caplog.set_level(logging.WARNING)

    running.manager.initialize_all(session_id="s1", hermes_home=str(running.home), platform="cli")

    assert running.prefetch("anything") == ""
    assert "must start with https://" in running.tool("kemory_search", {"query": "x"})["error"]
    assert len([r for r in caplog.records if "must start with https://" in r.getMessage()]) == 1
    assert running.server.requests == []


def test_prefetch_timeout_returns_within_deadline(
    start_provider: StartProvider, fake_kemory: FakeKemory
) -> None:
    fake_kemory.reply("POST", SEARCH, 200, HIT, delay=10, times=1)
    running = start_provider(recall_timeout=1)
    running.provider.wait_for_context(3)

    started = time.monotonic()
    block = running.prefetch("something slow")
    elapsed = time.monotonic() - started

    assert block == ""
    assert elapsed < 1.5
    fake_kemory.reply("POST", SEARCH, 200, HIT)
    assert "Likes tea." in running.prefetch("something quick")


@pytest.mark.parametrize("failing", [ME_ORGS, SEARCH])
def test_401_inert_with_one_warning(
    start_provider: StartProvider,
    fake_kemory: FakeKemory,
    caplog: pytest.LogCaptureFixture,
    failing: str,
) -> None:
    caplog.set_level(logging.WARNING)
    method = "GET" if failing == ME_ORGS else "POST"
    fake_kemory.reply(method, failing, 401, {"detail": "Invalid API key"})
    running = start_provider()
    running.provider.wait_for_context(3)
    first = running.prefetch("question one")
    sent_so_far = len(fake_kemory.requests)

    assert first == ""
    assert running.prefetch("question two") == ""
    for name, args in TOOL_CALLS:
        assert "KEMORY_API_KEY" in running.tool(name, args)["error"]
    assert len(fake_kemory.requests) == sent_so_far
    named = [r for r in caplog.records if "KEMORY_API_KEY" in r.getMessage()]
    assert len(named) == 1
    assert named[0].levelno == logging.WARNING


@pytest.mark.parametrize("hosts", [["eu.kemory.example"], None])
def test_403_residency_inert_names_target_hosts(
    start_provider: StartProvider,
    fake_kemory: FakeKemory,
    caplog: pytest.LogCaptureFixture,
    hosts: list[str] | None,
) -> None:
    caplog.set_level(logging.WARNING)
    fake_kemory.reply(
        "POST",
        SEARCH,
        403,
        {"detail": {"code": "ORG_WRONG_RESIDENCY_CELL", "target_hosts": hosts}},
    )
    running = start_provider()
    running.provider.wait_for_context(3)

    assert running.prefetch("question one") == ""
    sent_so_far = len(fake_kemory.requests)
    assert running.prefetch("question two") == ""

    for name, args in TOOL_CALLS:
        error = running.tool(name, args)["error"]
        assert "KEMORY_URL" in error
        assert ("eu.kemory.example" in error) if hosts else ("no target hosts" in error)
    assert len(fake_kemory.requests) == sent_so_far
    assert len([r for r in caplog.records if "KEMORY_URL" in r.getMessage()]) == 1


@pytest.mark.parametrize("via", ["file", "env"])
def test_org_pin_mismatch_blocks_writes_and_names_both_ids(
    start_provider: StartProvider,
    fake_kemory: FakeKemory,
    monkeypatch: pytest.MonkeyPatch,
    via: str,
) -> None:
    if via == "env":
        monkeypatch.setenv("KEMORY_ORG_ID", "org-other-9")
        running = start_provider()
    else:
        running = start_provider(org_id="org-other-9")
    running.provider.wait_for_context(3)

    assert running.prefetch("something to search for") == ""

    for name, args in TOOL_CALLS:
        error = running.tool(name, args)["error"]
        assert "org-other-9" in error
        assert KEY_ORG_ID in error
    assert fake_kemory.paths() == [ME_ORGS]


@pytest.mark.parametrize("platform", ["cli", "tui", "desktop", "acp", "local", "", None])
def test_local_surfaces_on(
    start_provider: StartProvider, fake_kemory: FakeKemory, platform: str | None
) -> None:
    fake_kemory.reply("POST", SEARCH, 200, HIT)
    running = start_provider(platform=platform, agent_context="primary")

    prompt = running.provider.system_prompt_block()
    for tool in TOOL_NAMES:
        assert tool in prompt
    assert {s["name"] for s in running.provider.get_tool_schemas()} == TOOL_NAMES
    assert "- [user:profile] Likes tea." in running.prefetch("what do I like?")
    assert running.tool("kemory_search", {"query": "tea"})["results"]


def test_cron_recalls_with_hermes_init_kwargs(
    start_provider: StartProvider, fake_kemory: FakeKemory
) -> None:
    fake_kemory.reply("POST", SEARCH, 200, HIT)
    # Hermes starts a cron job with platform "cron" and agent_context "cron", not "cli".
    running = start_provider(platform="cron", agent_context="cron")

    assert "- [user:profile] Likes tea." in running.prefetch("what do I like?")
    assert running.tool("kemory_search", {"query": "tea"})["results"]
    assert running.tool("kemory_store", {"content": "Likes tea"})["stored"] is True
    assert {s["name"] for s in running.provider.get_tool_schemas()} == TOOL_NAMES


@pytest.mark.parametrize(
    "platform",
    [
        "telegram",
        "discord",
        "slack",
        "api_server",
        "webhook",
        "msgraph_webhook",
        "gateway",
        "kanban",
        "codex",
        "tool",
        "xyz",
    ],
)
def test_messaging_and_server_platforms_off_by_default(
    start_provider: StartProvider, fake_kemory: FakeKemory, platform: str
) -> None:
    fake_kemory.reply("POST", SEARCH, 200, HIT)
    running = start_provider(platform=platform)

    assert running.provider.get_tool_schemas() == []
    assert running.provider.system_prompt_block() == ""
    assert running.prefetch("what do I like?") == ""
    assert running.provider.recall_status() is None
    for name, args in TOOL_CALLS:
        assert "gateways" in running.tool(name, args)["error"]
    assert fake_kemory.requests == []


@pytest.mark.parametrize("platform", ["telegram", "api_server", "xyz"])
def test_gateway_opt_in(
    start_provider: StartProvider, fake_kemory: FakeKemory, platform: str
) -> None:
    fake_kemory.reply("POST", SEARCH, 200, HIT)
    running = start_provider(platform=platform, gateways=True)

    assert {s["name"] for s in running.provider.get_tool_schemas()} == TOOL_NAMES
    assert running.provider.system_prompt_block() != ""
    assert "- [user:profile] Likes tea." in running.prefetch("what do I like?")
    assert running.tool("kemory_search", {"query": "tea"})["results"]


def test_auto_recall_off_sends_no_search_but_keeps_the_tools(
    start_provider: StartProvider, fake_kemory: FakeKemory
) -> None:
    fake_kemory.reply("POST", SEARCH, 200, HIT)
    running = start_provider(auto_recall=False)

    assert running.prefetch("what do I like?") == ""
    assert running.provider.wait_for_context(1)
    assert running.tool("kemory_search", {"query": "tea"})["results"]
    assert len(bodies_of(fake_kemory, "POST", SEARCH)) == 1
    assert bodies_of(fake_kemory, "GET", CONTEXT) == []


GARBAGE = b"\x00not json{"


def _break_server(server: FakeKemory, how: str) -> None:
    if how == "garbage":
        for method, path in (
            ("GET", ME_ORGS),
            ("GET", CONTEXT),
            ("POST", SEARCH),
            ("POST", "/api/v1/memories"),
            ("DELETE", "/api/v1/memories/m1"),
        ):
            server.reply(method, path, 200, GARBAGE)
    elif how == "shapes":
        server.reply("GET", CONTEXT, 200, {"namespaces": "nope"})
        server.reply(
            "POST", SEARCH, 200, {"items": [None, 5, {"content": ["x"]}, {"content": "ok"}]}
        )
        server.reply("POST", "/api/v1/memories", 200, [1, 2])
        server.reply("DELETE", "/api/v1/memories/m1", 200, "text")
    elif how == "html":
        for method, path in (
            ("GET", CONTEXT),
            ("POST", SEARCH),
            ("POST", "/api/v1/memories"),
            ("DELETE", "/api/v1/memories/m1"),
        ):
            server.reply(method, path, 502, "<html>bad gateway</html>")


@pytest.mark.parametrize("how", ["garbage", "shapes", "html", "refused"])
def test_no_public_method_raises_on_garbage_or_refused(
    start_provider: StartProvider,
    fake_kemory: FakeKemory,
    how: str,
) -> None:
    _break_server(fake_kemory, how)
    # Nothing listens on port 1.
    url = "http://127.0.0.1:1" if how == "refused" else None
    running = start_provider(recall_timeout=1, write_timeout=1, url=url)
    provider = running.provider
    provider.wait_for_context(3)

    assert isinstance(provider.prefetch("a question about tea"), str)
    for name, args in TOOL_CALLS:
        result = json.loads(provider.handle_tool_call(name, args))
        assert isinstance(result, dict)
        if how != "shapes":
            assert "error" in result
    assert isinstance(provider.system_prompt_block(), str)
    assert isinstance(provider.get_tool_schemas(), list)
    provider.recall_status()
    provider.queue_prefetch("q")
    provider.sync_turn("hello", "hi", session_id="s1", messages=None, turn_author=None)
    provider.on_session_end([])
    provider.on_session_switch("s2", parent_session_id="s1", reset=False)
    assert provider.on_pre_compress([{"role": "user", "content": "x"}]) == ""
    provider.shutdown()
    assert isinstance(provider.get_status_config({}), dict)


def test_every_public_method_survives_nonsense_arguments() -> None:
    provider = KemoryMemoryProvider()
    junk: list[Any] = [None, 5, [], {}, object()]

    provider.initialize("s1", hermes_home=junk[1], platform=junk[0], agent_context=junk[2])
    provider.initialize("s1")
    for value in junk:
        assert isinstance(provider.prefetch(value), str)
        assert isinstance(provider.handle_tool_call(value, value), str)
        assert isinstance(provider.handle_tool_call("kemory_search", value), str)
        provider.sync_turn(value, value, session_id=value, messages=value, turn_author=value)
        provider.on_session_end(value)
        provider.on_session_switch(value, parent_session_id=value, reset=value)
        provider.on_pre_compress(value)
        assert isinstance(provider.get_status_config(value), dict)
        provider.save_config(value, value)
    provider.backup_paths()
    provider.shutdown()


def test_register_gives_new_instance_each_call() -> None:
    class Context:
        def __init__(self) -> None:
            self.providers: list[object] = []

        def register_memory_provider(self, provider: object) -> None:
            self.providers.append(provider)

    ctx = Context()
    hermes_kemory.register(ctx)
    hermes_kemory.register(ctx)

    first, second = ctx.providers
    assert isinstance(first, KemoryMemoryProvider)
    assert isinstance(second, KemoryMemoryProvider)
    assert first is not second
    assert first.name == "kemory"


def test_config_schema_asks_for_the_key_in_the_environment() -> None:
    schema = {field["key"]: field for field in KemoryMemoryProvider().get_config_schema()}

    assert schema["api_key"]["secret"] is True
    assert schema["api_key"]["required"] is True
    assert schema["api_key"]["env_var"] == "KEMORY_API_KEY"
    assert "base_url" in schema
    assert "gateways" in schema
    assert not any(field.get("env_var") for key, field in schema.items() if key != "api_key")


def test_save_config_writes_known_non_secret_keys_only(hermes_home: Path) -> None:
    path = hermes_home / "kemory.json"
    path.write_text(json.dumps({"org_id": "org-1", "recall_timeout": 2, "extra": "kept out"}))
    path.chmod(0o644)

    KemoryMemoryProvider().save_config(
        {
            "api_key": "kmk_never_written",
            "base_url": "https://kemory.example",
            "gateways": "true",
            "auto_capture": False,
            "unknown": 1,
            "org_id": None,
        },
        str(hermes_home),
    )

    saved = json.loads(path.read_text())
    assert saved == {
        "org_id": "org-1",
        "recall_timeout": 2,
        "extra": "kept out",
        "base_url": "https://kemory.example",
        "gateways": True,
        "auto_capture": False,
    }
    assert "kmk_never_written" not in path.read_text()
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert [p.name for p in hermes_home.iterdir()] == ["kemory.json"]


def test_save_config_creates_a_private_file(hermes_home: Path) -> None:
    KemoryMemoryProvider().save_config({"gateways": "false"}, str(hermes_home))

    path = hermes_home / "kemory.json"
    assert json.loads(path.read_text()) == {"gateways": False}
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_status_config_needs_no_network_and_shows_what_is_known(
    start_provider: StartProvider, fake_kemory: FakeKemory
) -> None:
    fresh = KemoryMemoryProvider()
    assert fresh.get_status_config({}) == {
        "base_url": "https://api.kemory.s9n.ai",
        "org_id": "not looked up yet",
        "last_error": "none",
    }
    configured = fresh.get_status_config({"base_url": "https://configured.example"})
    assert configured["base_url"] == "https://configured.example"

    running = start_provider(initialize=False)
    status = running.provider.get_status_config({})
    assert status["base_url"] == fake_kemory.url
    assert status["org_id"] == "not looked up yet"
    assert fake_kemory.requests == []

    running.manager.initialize_all(session_id="s1", hermes_home=str(running.home), platform="cli")
    running.provider.wait_for_context(3)
    status = running.provider.get_status_config({})
    assert status["org_id"] == KEY_ORG_ID
    assert status["last_error"] == "none"


def test_constants_are_the_documented_ones() -> None:
    assert frozenset({"", "cli", "tui", "desktop", "acp", "local"}) == LOCAL_PLATFORMS
    assert frozenset({"cron"}) == RECALL_ONLY_PLATFORMS
    assert frozenset({"cron", "subagent", "flush"}) == NO_CAPTURE_CONTEXTS

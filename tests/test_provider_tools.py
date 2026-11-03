"""The four memory tools, driven through the manager as Hermes calls them."""

import json
import time
from typing import Any

import pytest

from .conftest import Running, StartProvider
from .fake_kemory import FakeKemory

SEARCH = "/api/v1/memories/search"
MEMORIES = "/api/v1/memories"
CONTEXT = "/api/v1/user/context"


def sent(server: FakeKemory, method: str, path: str) -> list[Any]:
    return [r for r in server.requests if r.method == method and r.path == path]


def test_search_tool(start_provider: StartProvider, fake_kemory: FakeKemory) -> None:
    fake_kemory.reply(
        "POST",
        SEARCH,
        200,
        {
            "items": [
                {"memory_id": "m1", "namespace": "user:profile", "content": "Likes tea.", "x": 1},
                {"memory_id": "m1", "namespace": "user:profile", "content": "a duplicate id"},
                {"memory_id": "m2", "namespace": "work", "content": "y" * 1500},
                {"memory_id": "m3", "namespace": "work", "content": ""},
            ]
        },
    )
    running = start_provider(max_recall_results=5)

    result = running.tool(
        "kemory_search", {"query": "drinks", "namespace": "user:profile", "limit": 3}
    )

    assert result == {
        "results": [
            {"memory_id": "m1", "namespace": "user:profile", "content": "Likes tea."},
            {"memory_id": "m2", "namespace": "work", "content": "y" * 1500},
        ]
    }
    (request,) = sent(fake_kemory, "POST", SEARCH)
    assert request.body == {
        "query": "drinks",
        "limit": 3,
        "search_mode": "hybrid",
        "namespace": "user:profile",
    }


@pytest.mark.parametrize(
    ("limit", "expected"), [(3, 3), (0, 1), (99, 20), ("7", 6), (True, 6), (None, 6)]
)
def test_search_tool_limit_is_one_to_twenty(
    start_provider: StartProvider, fake_kemory: FakeKemory, limit: Any, expected: int
) -> None:
    running = start_provider()

    running.tool("kemory_search", {"query": "drinks", "limit": limit})

    (request,) = sent(fake_kemory, "POST", SEARCH)
    assert request.body["limit"] == expected
    assert "namespace" not in request.body


def test_search_tool_cuts_long_query(
    start_provider: StartProvider, fake_kemory: FakeKemory
) -> None:
    fake_kemory.search_items = [{"memory_id": "m1", "namespace": "n", "content": "found"}]
    running = start_provider()
    query = "é" * 3000

    result = running.tool("kemory_search", {"query": query})

    (request,) = sent(fake_kemory, "POST", SEARCH)
    assert request.body["query"] == "é" * 1000
    assert result == {"results": [{"memory_id": "m1", "namespace": "n", "content": "found"}]}


def test_store_tool_redacts_and_returns_id(
    start_provider: StartProvider, fake_kemory: FakeKemory
) -> None:
    running = start_provider()

    # Called on the provider itself: through the manager, Hermes would redact the text first.
    result = json.loads(
        running.provider.handle_tool_call(
            "kemory_store", {"content": "My deploy key is api_key=abcdef123456 for staging"}
        )
    )

    assert result == {"stored": True, "memory_id": "mem-1", "namespace": "general"}
    (request,) = sent(fake_kemory, "POST", MEMORIES)
    assert request.body == {
        "content": "My deploy key is [REDACTED] for staging",
        "content_type": "text",
        "metadata": {"source": "hermes", "capture": "tool"},
    }

    running.tool("kemory_store", {"content": "Prefers dark mode", "namespace": "user:profile"})
    assert sent(fake_kemory, "POST", MEMORIES)[1].body["namespace"] == "user:profile"


def test_forget_tool_quotes_id(start_provider: StartProvider, fake_kemory: FakeKemory) -> None:
    running = start_provider()

    result = running.tool("kemory_forget", {"memory_id": "a/b c?d#e"})

    assert result == {"forgotten": True, "memory_id": "a/b c?d#e"}
    (request,) = [r for r in fake_kemory.requests if r.method == "DELETE"]
    assert request.path == "/api/v1/memories/a%2Fb%20c%3Fd%23e"


def test_context_tool(start_provider: StartProvider, fake_kemory: FakeKemory) -> None:
    fake_kemory.reply(
        "GET",
        CONTEXT,
        200,
        {
            "namespaces": [
                {"namespace": "user:profile", "summary": "Prefers short answers."},
                {"namespace": "empty", "summary": ""},
                {"namespace": "work", "summary": "Builds a plugin."},
            ]
        },
    )
    running = start_provider()

    result = running.tool("kemory_context", {})

    context = result["context"]
    assert context.splitlines()[1:] == [
        "- [user:profile] Prefers short answers.",
        "- [work] Builds a plugin.",
    ]
    assert [r.path for r in fake_kemory.requests if r.method == "GET"].count(
        "/api/v1/user/context?depth=l3"
    ) == 2  # the one from initialize() and the tool's own


def test_search_and_context_use_recall_timeout_store_and_forget_use_write_timeout(
    start_provider: StartProvider, fake_kemory: FakeKemory
) -> None:
    for method, path in (("POST", SEARCH), ("GET", CONTEXT), ("POST", MEMORIES)):
        fake_kemory.reply(method, path, 200, {}, delay=3)
    fake_kemory.reply("DELETE", "/api/v1/memories/m1", 204, delay=3)
    running = start_provider(recall_timeout=0.5, write_timeout=2.0)
    running.provider.wait_for_context(2)
    # Warm up the organisation lookup so only the scripted calls are timed.
    running.tool("kemory_search", {"query": "warm up"})

    def took(name: str, args: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        started = time.monotonic()
        result = running.tool(name, args)
        return time.monotonic() - started, result

    seconds, result = took("kemory_search", {"query": "slow"})
    assert 0.4 < seconds < 1.2
    assert "error" in result
    seconds, result = took("kemory_context", {})
    assert 0.4 < seconds < 1.2
    assert "error" in result
    seconds, result = took("kemory_store", {"content": "slow"})
    assert 1.8 < seconds < 2.8
    assert "error" in result
    seconds, result = took("kemory_forget", {"memory_id": "m1"})
    assert 1.8 < seconds < 2.8
    assert "error" in result


@pytest.mark.parametrize(
    ("tool", "args", "method", "path", "status", "body"),
    [
        ("kemory_store", {"content": "x"}, "POST", MEMORIES, 409, {"detail": "Too similar"}),
        ("kemory_search", {"query": "x"}, "POST", SEARCH, 500, {"detail": "boom"}),
        ("kemory_forget", {"memory_id": "x"}, "DELETE", "/api/v1/memories/x", 404, None),
        ("kemory_context", {}, "GET", CONTEXT, 429, None),
    ],
)
def test_errors_return_json_error(
    start_provider: StartProvider,
    fake_kemory: FakeKemory,
    tool: str,
    args: dict[str, Any],
    method: str,
    path: str,
    status: int,
    body: Any,
) -> None:
    fake_kemory.reply(method, path, status, body)
    running = start_provider()

    result = running.tool(tool, args)

    assert set(result) == {"error"}
    assert str(status) in result["error"]
    if body:
        assert body["detail"] in result["error"]


def test_bad_arguments_and_unknown_tools_return_json_error(start_provider: StartProvider) -> None:
    running = start_provider()
    provider = running.provider

    for name, args in (
        ("kemory_search", {}),
        ("kemory_search", {"query": "  "}),
        ("kemory_search", {"query": 5}),
        ("kemory_store", {"content": ""}),
        ("kemory_forget", {}),
    ):
        result = json.loads(provider.handle_tool_call(name, args))
        assert "is required" in result["error"], (name, args)

    unknown = json.loads(provider.handle_tool_call("kemory_destroy", {}))
    assert "kemory_destroy" in unknown["error"]
    assert "error" in json.loads(provider.handle_tool_call("kemory_search", ["x"]))  # type: ignore[arg-type]


def test_tool_schemas_match_the_contract(start_provider: StartProvider) -> None:
    running = start_provider()

    schemas = {s["name"]: s["parameters"] for s in running.provider.get_tool_schemas()}

    assert set(schemas) == {"kemory_search", "kemory_store", "kemory_forget", "kemory_context"}
    assert schemas["kemory_search"]["required"] == ["query"]
    assert set(schemas["kemory_search"]["properties"]) == {"query", "namespace", "limit"}
    assert schemas["kemory_search"]["properties"]["limit"] == {
        "type": "integer",
        "minimum": 1,
        "maximum": 20,
        "description": schemas["kemory_search"]["properties"]["limit"]["description"],
    }
    assert schemas["kemory_store"]["required"] == ["content"]
    assert set(schemas["kemory_store"]["properties"]) == {"content", "namespace"}
    assert schemas["kemory_forget"]["required"] == ["memory_id"]
    assert schemas["kemory_context"]["properties"] == {}
    assert "required" not in schemas["kemory_context"]
    assert {s["name"] for s in running.manager.get_all_tool_schemas()} == set(schemas)


def test_tools_routed_through_manager(start_provider: StartProvider) -> None:
    running: Running = start_provider()

    assert running.manager.get_all_tool_names() == {
        "kemory_search",
        "kemory_store",
        "kemory_forget",
        "kemory_context",
    }

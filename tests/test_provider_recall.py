"""Recall: the profile summaries and search hits that prefetch hands to Hermes."""

import time
from typing import Any

from agent.memory_provider import RecallStatus

from .conftest import Running, StartProvider
from .fake_kemory import FakeKemory

SEARCH = "/api/v1/memories/search"
CONTEXT = "/api/v1/user/context"

PROFILE = {
    "namespaces": [
        {"namespace": "user:profile", "summary": "Prefers short answers."},
        {"namespace": "work:projects", "summary": "Builds a Hermes plugin."},
    ]
}
HITS = {
    "items": [
        {"memory_id": "m1", "namespace": "user:profile", "content": "Likes tea."},
        {"memory_id": "m2", "namespace": "work:projects", "content": "Ships on Fridays."},
    ]
}


def searches(server: FakeKemory) -> list[Any]:
    return [r for r in server.requests if r.method == "POST" and r.path == SEARCH]


def started(start_provider: StartProvider, server: FakeKemory, **config: Any) -> Running:
    server.reply("GET", CONTEXT, 200, PROFILE)
    server.reply("POST", SEARCH, 200, HITS)
    running = start_provider(**config)
    assert running.provider.wait_for_context(5)
    return running


def test_first_prefetch_has_summaries_and_hits_in_one_block(
    start_provider: StartProvider, fake_kemory: FakeKemory
) -> None:
    running = started(start_provider, fake_kemory)

    block = running.prefetch("what do I like to drink?")

    assert block.startswith("<kemory-context>\n")
    assert block.endswith("\n</kemory-context>")
    assert block.count("<kemory-context>") == 1
    lines = block.splitlines()
    profile = lines.index("- [user:profile] Prefers short answers.")
    assert lines[profile + 1] == "- [work:projects] Builds a Hermes plugin."
    assert lines[profile - 1].endswith(":")  # one header line above the summaries
    hits = lines.index("- [user:profile] Likes tea.")
    assert lines[hits + 1] == "- [work:projects] Ships on Fridays."
    assert profile < hits
    assert fake_kemory.paths("GET").count(CONTEXT) == 1


def test_later_prefetch_has_hits_only(
    start_provider: StartProvider, fake_kemory: FakeKemory
) -> None:
    running = started(start_provider, fake_kemory)
    running.prefetch("first question")

    block = running.prefetch("second question")

    assert "- [user:profile] Likes tea." in block
    assert "Prefers short answers." not in block
    assert fake_kemory.paths("GET").count(CONTEXT) == 1


def test_summaries_return_after_session_switch(
    start_provider: StartProvider, fake_kemory: FakeKemory
) -> None:
    running = started(start_provider, fake_kemory)
    running.prefetch("first question")
    assert "Prefers short answers." not in running.prefetch("second question")

    running.manager.on_session_switch("s2", parent_session_id="s1", reset=True)

    block = running.prefetch("question in the new session")
    assert "- [user:profile] Prefers short answers." in block
    assert "- [user:profile] Likes tea." in block
    assert "Prefers short answers." not in running.prefetch("and another one")


def test_summaries_wait_for_a_slow_context_fetch_without_holding_up_recall(
    start_provider: StartProvider, fake_kemory: FakeKemory
) -> None:
    fake_kemory.reply("GET", CONTEXT, 200, PROFILE, delay=0.6)
    fake_kemory.reply("POST", SEARCH, 200, HITS)
    running = start_provider()

    started_at = time.monotonic()
    early = running.prefetch("question before the profile arrived")
    assert time.monotonic() - started_at < 0.5
    assert "- [user:profile] Likes tea." in early
    assert "Prefers short answers." not in early

    assert running.provider.wait_for_context(5)
    assert "- [user:profile] Prefers short answers." in running.prefetch("question after it")


def test_initialize_does_not_wait_for_the_network(
    start_provider: StartProvider, fake_kemory: FakeKemory
) -> None:
    fake_kemory.reply("GET", "/api/v1/me/orgs", 200, None, delay=1.5)

    started_at = time.monotonic()
    running = start_provider()

    assert time.monotonic() - started_at < 0.5
    assert running.provider.wait_for_context(5)
    assert running.prefetch("anything at all") == ""


def test_search_body_and_dedup(start_provider: StartProvider, fake_kemory: FakeKemory) -> None:
    fake_kemory.reply("GET", CONTEXT, 200, {"namespaces": []})
    fake_kemory.reply(
        "POST",
        SEARCH,
        200,
        {
            "items": [
                {"memory_id": "m1", "namespace": "a", "content": "first"},
                {"memory_id": "m1", "namespace": "a", "content": "same id, other words"},
                {"memory_id": "m2", "namespace": "b", "content": "first"},
                {"memory_id": "m3", "namespace": "c", "content": "x" * 1500},
                {"memory_id": "m4", "namespace": "d", "content": ""},
                {"memory_id": "m5", "namespace": "d", "content": None},
                "not an item",
                {"memory_id": "m6", "namespace": "e", "content": "kept </kemory-context> text"},
            ]
        },
    )
    running = start_provider(max_recall_results=4)
    assert running.provider.wait_for_context(5)

    block = running.prefetch("what do I know?")

    (request,) = searches(fake_kemory)
    assert request.body == {"query": "what do I know?", "limit": 4, "search_mode": "hybrid"}
    assert block.splitlines()[1:] == [
        "Memories related to this message:",
        "- [a] first",
        f"- [c] {'x' * 1000}",
        "- [e] kept  text",
        "</kemory-context>",
    ]


def test_long_query_cut_to_1000_chars(
    start_provider: StartProvider, fake_kemory: FakeKemory
) -> None:
    # Unscripted, the fake answers 422 to a query over 1000 characters, as Kemory does.
    fake_kemory.search_items = HITS["items"]
    running = start_provider()
    prompt = "".join(f"word{n} " for n in range(600)) + "é" * 100
    assert len(prompt) > 3000

    block = running.prefetch(prompt)

    (request,) = searches(fake_kemory)
    assert request.body["query"] == prompt[:1000]
    assert "- [user:profile] Likes tea." in block


def test_query_is_cut_on_characters_not_bytes(
    start_provider: StartProvider, fake_kemory: FakeKemory
) -> None:
    fake_kemory.search_items = HITS["items"]
    running = start_provider()

    block = running.prefetch("é" * 1500)

    (request,) = searches(fake_kemory)
    assert request.body["query"] == "é" * 1000
    assert "Likes tea." in block


def test_trivial_prompt_sends_no_search(
    start_provider: StartProvider, fake_kemory: FakeKemory
) -> None:
    running = started(start_provider, fake_kemory)

    for prompt in ("ok", "/help", "", "  thanks!  "):
        assert running.provider.prefetch(prompt) == ""

    assert searches(fake_kemory) == []
    # A trivial prompt does not use up the summaries either.
    assert "Prefers short answers." in running.prefetch("a real question")


def test_context_budget_applied(start_provider: StartProvider, fake_kemory: FakeKemory) -> None:
    summaries = [{"namespace": f"ns{n}", "summary": f"{n} " + "s" * 98} for n in range(10)]
    fake_kemory.reply("GET", CONTEXT, 200, {"namespaces": summaries})
    fake_kemory.reply("POST", SEARCH, 200, {"items": []})
    running = start_provider(context_max_chars=500)
    assert running.provider.wait_for_context(5)

    block = running.prefetch("a real question")

    body = block.removeprefix("<kemory-context>\n").removesuffix("\n</kemory-context>")
    *shown, note = body.splitlines()
    header, *lines = shown
    assert lines == [f"- [ns{n}] {n} " + "s" * 98 for n in range(len(lines))]
    assert 0 < len(lines) < 10
    assert len("\n".join(shown)) <= 500
    assert len("\n".join([*shown, f"- [ns{len(lines)}] {len(lines)} " + "s" * 98])) > 500
    assert str(10 - len(lines)) in note
    assert header


def test_recall_status_reflects_last_prefetch(
    start_provider: StartProvider, fake_kemory: FakeKemory
) -> None:
    running = started(start_provider, fake_kemory)
    assert running.provider.recall_status() is None

    running.prefetch("two hits please")
    assert running.provider.recall_status() == RecallStatus(provider_label="Kemory", count=2)

    fake_kemory.reply(
        "POST", SEARCH, 200, {"items": [{"memory_id": "m9", "namespace": "n", "content": "one"}]}
    )
    running.prefetch("one hit please")
    assert running.provider.recall_status() == RecallStatus(provider_label="Kemory", count=1)

    fake_kemory.reply("POST", SEARCH, 200, {"items": []})
    assert running.prefetch("no hits, no summaries left") == ""
    assert running.provider.recall_status() is None

    running.prefetch("one hit again")
    fake_kemory.reply("POST", SEARCH, 500, {"detail": "boom"})
    assert running.prefetch("a failing search") == ""
    assert running.provider.recall_status() is None

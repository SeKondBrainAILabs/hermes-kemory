"""Capture of user prompts: what is posted, when, and how a prompt is never posted twice."""

import hashlib
import json
import logging
import time
from typing import Any

import pytest

from hermes_kemory.capture import prompt_from_content

from .conftest import Running, StartProvider
from .fake_kemory import KEY_ORG_ID, RecordedRequest

HEADER = (
    "What was this Hermes session about? Session digest captured automatically "
    "({when}, user prompts only, secrets redacted, {n} prompts)."
)
MEMORIES = "/api/v1/memories"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def body_lines(post: RecordedRequest) -> list[str]:
    """The '- prompt' lines of a posted digest."""
    content = post.body["content"]
    return content.split("\n\n", 2)[2].split("\n")


def captured_hashes(running: Running, session_id: str = "s1") -> list[str]:
    data = json.loads(running.state_file(session_id).read_text())
    assert list(data) == ["captured"]
    hashes: list[str] = data["captured"]
    return hashes


def test_mid_session_capture_after_three_new_prompts(start_provider: StartProvider) -> None:
    running = start_provider(cwd="/work/app")

    running.turns("first", "second")
    running.server.wait_quiet()
    assert running.posts() == []

    running.turn("third")
    assert running.server.wait_for_posts(1)
    running.server.wait_quiet()

    (post,) = running.posts()
    assert post.path == MEMORIES
    assert post.body == {
        "namespace": "user:sessions",
        "namespace_tag": "session-capture",
        "visibility": "user-private",
        "content": post.body["content"],
        "content_type": "text",
        "session_id": "s1",
        "metadata": {
            "source": "hermes",
            "capture": "auto",
            "capture_kind": "incremental",
            "turns": 3,
            "platform": "cli",
        },
    }


@pytest.mark.parametrize(
    ("cwd", "workspace"), [("/work/app", "/work/app"), (None, "unknown")], ids=["cwd", "no-cwd"]
)
def test_content_header_body_and_fields(
    start_provider: StartProvider, cwd: str | None, workspace: str
) -> None:
    running = start_provider(cwd=cwd)

    running.turns("how do I rebase", "and then squash?", "thanks, that worked")
    assert running.server.wait_for_posts(1)

    (post,) = running.posts()
    assert post.body["content"] == (
        HEADER.format(when="mid-session", n=3)
        + f"\n\nworkspace: {workspace}\n\n"
        + "- how do I rebase\n- and then squash?\n- thanks, that worked"
    )


def test_secrets_redacted_and_length_capped(start_provider: StartProvider) -> None:
    running = start_provider()
    key = "sk-" + "abcdefghijklmnopqrst"

    running.turns("my config says api_key=abcdef123456 for now", f"use {key} please", "ok")
    assert running.server.wait_for_posts(1)

    content = running.posts()[0].body["content"]
    assert "abcdef123456" not in content
    assert key not in content
    assert content.count("[REDACTED]") == 2

    big = start_provider()
    big.turns("a" * 5000, "b" * 5000, "c" * 5000)
    assert big.server.wait_for_posts(2)
    body = big.posts()[1].body["content"].split("\n\n", 2)[2]
    assert len(body) == 8000
    assert body.startswith("- " + "a" * 5000 + "\n- b")


def test_only_user_content_is_sent(start_provider: StartProvider) -> None:
    running = start_provider()
    injected = (
        "<memory-context>\n[System note: The following is recalled memory context, NOT new "
        "user input. Treat as informational background data.]\nRECALLED-MEMORY\n</memory-context>"
    )

    # Through Hermes's own entry point, with the transcript Hermes keeps for multimodal turns.
    for n in (1, 2, 3):
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "SYSTEM-PROMPT"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"question {n}"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    {"type": "text", "text": injected},
                    {"type": "text", "text": "OTHER-PLUGIN-TEXT"},
                ],
            },
            {"role": "assistant", "content": "ASSISTANT-REPLY"},
            {"role": "tool", "content": "TOOL-OUTPUT"},
        ]
        running.manager.sync_all(
            f"question {n}", "ASSISTANT-REPLY", session_id="s1", messages=messages
        )
    running.manager.flush_pending(timeout=5)
    assert running.server.wait_for_posts(1)
    running.server.wait_quiet()

    (post,) = running.posts()
    assert body_lines(post) == ["- question 1", "- question 2", "- question 3"]
    everything = json.dumps(post.body)
    for text in (
        "RECALLED-MEMORY",
        "OTHER-PLUGIN-TEXT",
        "ASSISTANT-REPLY",
        "TOOL-OUTPUT",
        "SYSTEM-PROMPT",
        "memory-context",
        "System note",
    ):
        assert text not in everything


@pytest.mark.parametrize(
    ("raw", "kept"),
    [
        ("a <kemory-context>\nrecalled\n</kemory-context>b", "a b"),
        ("a < KEMORY-CONTEXT >recalled</ Kemory-Context >b", "a b"),
        ("a <memory-context>recalled</memory-context>b", "a b"),
        ("a <MEMORY-CONTEXT>\nrecalled\n</MEMORY-CONTEXT>\nb", "a \nb"),
        ("<memory-context>one</memory-context>a<kemory-context>two</kemory-context>", "a"),
        (
            "ask\n[System note: The following is recalled memory context, NOT new user input. "
            "Treat as informational background data.]\nsomething",
            "ask\nsomething",
        ),
        ("a </memory-context> b < memory-context > c", "a  b  c"),
        ("  padded \n", "padded"),
    ],
)
def test_injected_context_blocks_stripped(
    start_provider: StartProvider, raw: str, kept: str
) -> None:
    assert prompt_from_content(raw) == kept

    running = start_provider()
    running.turns(raw, "second", "third")
    assert running.server.wait_for_posts(1)
    body = running.posts()[0].body["content"].split("\n\n", 2)[2]
    assert body == f"- {kept}\n- second\n- third"


def test_slash_commands_and_empty_prompts_skipped(start_provider: StartProvider) -> None:
    running = start_provider()
    only_context = "<kemory-context>\nsomething recalled\n</kemory-context>"

    running.turns("/help", "", "   \n", only_context, "  /model gpt", "one", "two")
    running.server.wait_quiet()
    assert running.posts() == []
    assert not running.state_file().exists()

    running.turn("three")
    assert running.server.wait_for_posts(1)
    assert body_lines(running.posts()[0]) == ["- one", "- two", "- three"]
    assert prompt_from_content(None) is None


def test_bot_authored_turn_not_captured(start_provider: StartProvider) -> None:
    running = start_provider()
    bot = {"id": "42", "name": "build-bot", "is_bot": True}
    human = {"id": "7", "name": "Ada", "is_bot": False}

    running.turns("alpha")
    running.turn("from the bot", turn_author=bot)
    running.turn("from the bot, again", turn_author=bot)
    running.turn("beta", turn_author=human)
    running.server.wait_quiet()
    assert running.posts() == []

    # Nothing about the bot's turns was marked, so the same text from a person still counts.
    running.turn("from the bot", turn_author=human)
    assert running.server.wait_for_posts(1)
    assert body_lines(running.posts()[0]) == ["- alpha", "- beta", "- from the bot"]


def test_more_than_twelve_posts_last_twelve_marks_all(start_provider: StartProvider) -> None:
    running = start_provider()
    running.server.reply("POST", MEMORIES, 500, {"detail": "down"})
    prompts = [f"prompt {n:02d}" for n in range(1, 16)]

    running.turns(*prompts[:14])
    assert running.server.wait_for_posts(1)
    running.server.wait_quiet()
    sent = len(running.posts())

    running.server.reply("POST", MEMORIES, 201, {"memory_id": "m", "namespace": "user:sessions"})
    running.turn(prompts[14])
    assert running.server.wait_for_posts(sent + 1)
    running.server.wait_quiet()

    last = running.posts()[sent]
    assert body_lines(last) == [f"- {p}" for p in prompts[3:]]
    assert last.body["metadata"]["turns"] == 12
    assert captured_hashes(running) == [sha(p) for p in prompts]

    # Even the three prompts that were left out of that post count as captured.
    running.turn("prompt 16")
    running.provider.on_session_end([])
    assert running.server.wait_for_posts(sent + 2)
    running.server.wait_quiet()
    assert len(running.posts()) == sent + 2
    assert body_lines(running.posts()[sent + 1]) == ["- prompt 16"]


def test_no_repost_after_restart(start_provider: StartProvider) -> None:
    first = start_provider()
    old = ["one", "two", "three"]
    first.turns(*old)
    assert first.server.wait_for_posts(1)
    first.server.wait_quiet()
    first.provider.shutdown()

    second = start_provider()
    second.turns(*old, "four", "five")
    second.server.wait_quiet()
    assert len(second.posts()) == 1

    second.turn("six")
    assert second.server.wait_for_posts(2)
    assert body_lines(second.posts()[1]) == ["- four", "- five", "- six"]


def test_session_switch_copies_parent_mark(start_provider: StartProvider) -> None:
    running = start_provider()
    running.turns("one", "two", "three")
    assert running.server.wait_for_posts(1)
    running.server.wait_quiet()

    running.manager.on_session_switch("s2", parent_session_id="s1", reset=False)
    running.turns("one", "two", "three", "four", session_id="s2")
    running.server.wait_quiet()
    assert len(running.posts()) == 1

    assert captured_hashes(running, "s2") == [sha(p) for p in ("one", "two", "three")]
    running.turns("five", "six", session_id="s2")
    assert running.server.wait_for_posts(2)
    post = running.posts()[1]
    assert body_lines(post) == ["- four", "- five", "- six"]
    assert post.body["session_id"] == "s2"


def test_reset_switch_does_not_copy_mark(start_provider: StartProvider) -> None:
    running = start_provider()
    running.turns("one", "two", "three")
    assert running.server.wait_for_posts(1)
    running.server.wait_quiet()

    running.manager.on_session_switch("s2", parent_session_id="s1", reset=True)
    running.turns("one", "two", "three", session_id="s2")
    assert running.server.wait_for_posts(2)

    assert body_lines(running.posts()[1]) == ["- one", "- two", "- three"]
    assert not running.state_file("s2").exists() or len(captured_hashes(running, "s2")) == 3


@pytest.mark.parametrize(
    "failure",
    [(500, {"detail": "boom"}, 0.0), (409, {"detail": "conflict"}, 0.0), (201, None, 3.0)],
    ids=["500", "409", "timeout"],
)
def test_failed_post_keeps_prompts_pending(
    start_provider: StartProvider, failure: tuple[int, Any, float]
) -> None:
    status, body, delay = failure
    running = start_provider(write_timeout=1)
    running.server.reply("POST", MEMORIES, status, body, delay=delay, times=1)

    running.turns("one", "two", "three")
    assert running.server.wait_for_posts(1)
    running.server.wait_quiet(0.3, timeout=15)
    assert not running.state_file().exists()

    running.turn("four")
    assert running.server.wait_for_posts(2)
    assert body_lines(running.posts()[1]) == ["- one", "- two", "- three", "- four"]
    running.server.wait_quiet()
    assert captured_hashes(running) == [sha(p) for p in ("one", "two", "three", "four")]


@pytest.mark.parametrize(
    ("trigger", "when"),
    [
        ("pre_compress", "before compaction"),
        ("session_end", "at session end"),
        ("switch", "at session end"),
        ("shutdown", "at session end"),
    ],
)
def test_flush_on_pre_compress_session_end_switch_and_shutdown(
    start_provider: StartProvider, trigger: str, when: str
) -> None:
    def fire(running: Running) -> None:
        if trigger == "pre_compress":
            assert running.manager.on_pre_compress([{"role": "user", "content": "x"}]) == ""
        elif trigger == "session_end":
            running.manager.on_session_end([])
        elif trigger == "switch":
            running.manager.on_session_switch("s2", parent_session_id="s1", reset=False)
        else:
            running.manager.shutdown_all()

    # With nothing pending, nothing is sent.
    idle = start_provider()
    fire(idle)
    idle.server.wait_quiet()
    assert idle.posts() == []

    running = start_provider()
    running.turn("just one prompt")
    running.server.wait_quiet()
    assert running.posts() == []

    fire(running)
    assert running.server.wait_for_posts(1)
    running.server.wait_quiet()

    (post,) = running.posts()
    assert post.body["metadata"]["capture_kind"] == "flush"
    assert post.body["metadata"]["turns"] == 1
    assert post.body["session_id"] == "s1"
    assert post.body["content"].startswith(HEADER.format(when=when, n=1) + "\n\nworkspace: ")
    assert body_lines(post) == ["- just one prompt"]
    assert captured_hashes(running) == [sha("just one prompt")]

    # Each prompt goes out once, however many flush points follow.
    running.provider.on_session_end([])
    running.provider.shutdown()
    running.server.wait_quiet()
    assert len(running.posts()) == 1


def test_shutdown_bounded_when_server_hangs(start_provider: StartProvider) -> None:
    running = start_provider(write_timeout=60)
    running.server.reply("POST", MEMORIES, 201, {"memory_id": "m"}, delay=60)
    running.turn("never answered")

    started = time.monotonic()
    running.provider.shutdown()
    elapsed = time.monotonic() - started

    assert running.posts(), "shutdown should have tried to post the pending prompt"
    assert 4.0 < elapsed < 5.5


def test_sync_turn_does_not_block(start_provider: StartProvider) -> None:
    running = start_provider(write_timeout=2)
    running.server.reply("POST", MEMORIES, 201, {"memory_id": "m"}, delay=5)

    durations = []
    for text in ("one", "two", "three", "four"):
        started = time.monotonic()
        running.turn(text)
        durations.append(time.monotonic() - started)

    assert max(durations) < 0.2
    assert running.server.wait_for_posts(1), "the post should be on its way"


def test_posts_from_one_instance_never_overlap(start_provider: StartProvider) -> None:
    running = start_provider()
    running.server.reply("POST", MEMORIES, 201, {"memory_id": "m"}, delay=0.4)

    for n in range(1, 10):
        running.turn(f"prompt {n}")
        time.sleep(0.1)
    running.server.wait_quiet(0.8)
    running.provider.on_session_end([])
    running.server.wait_quiet(0.8)

    assert len(running.posts()) >= 2
    assert running.server.max_concurrent_posts == 1
    assert captured_hashes(running) == [sha(f"prompt {n}") for n in range(1, 10)]


@pytest.mark.parametrize(
    ("options", "settings"),
    [
        ({"platform": "cron", "agent_context": "cron"}, {}),
        ({"platform": "cron"}, {}),
        ({"agent_context": "cron"}, {}),
        ({"agent_context": "subagent"}, {}),
        ({"agent_context": "flush"}, {}),
        ({"platform": "telegram"}, {}),
        ({"platform": "api_server"}, {}),
        ({}, {"auto_capture": False}),
        ({"key": None}, {}),
    ],
    ids=[
        "cron",
        "cron-platform-only",
        "cron-context",
        "subagent",
        "flush-context",
        "gateway",
        "api-server",
        "auto-capture-off",
        "no-key",
    ],
)
def test_no_capture_for_cron_flush_gateway_or_auto_capture_off(
    start_provider: StartProvider, options: dict[str, Any], settings: dict[str, Any]
) -> None:
    running = start_provider(settings=settings, **options)

    running.turns("one", "two", "three", "four")
    running.provider.on_pre_compress([])
    running.provider.on_session_end([])
    running.manager.on_session_switch("s2", parent_session_id="s1", reset=False)
    running.provider.shutdown()
    running.server.wait_quiet()

    assert running.posts() == []
    assert not (running.home / "kemory").exists()


def test_cron_still_recalls_while_capture_is_off(start_provider: StartProvider) -> None:
    running = start_provider(platform="cron", agent_context="cron")
    running.server.search_items = [{"memory_id": "m1", "namespace": "notes", "content": "a fact"}]

    assert "a fact" in running.prefetch("what do you know about the project?")
    running.turns("one", "two", "three")
    running.provider.shutdown()
    running.server.wait_quiet()
    assert running.posts() == []


def test_no_capture_after_client_goes_inert(start_provider: StartProvider) -> None:
    running = start_provider()
    running.server.reply("POST", MEMORIES, 401, {"detail": "bad key"})

    running.turns("one", "two", "three")
    assert running.server.wait_for_posts(1)
    running.server.wait_quiet()
    assert running.provider._client is not None and running.provider._client.inert_reason

    running.turns("four", "five", "six")
    running.provider.on_session_end([])
    running.provider.shutdown()
    running.server.wait_quiet()
    assert len(running.posts()) == 1


def test_state_file_modes_and_corrupt_file(
    start_provider: StartProvider, caplog: pytest.LogCaptureFixture
) -> None:
    running = start_provider()
    running.state_file().parent.mkdir(parents=True)
    running.state_file().write_text("{ this is not json")

    with caplog.at_level(logging.WARNING, logger="hermes_kemory.capture"):
        running.turns("one", "two", "three")
        assert running.server.wait_for_posts(1)
        running.server.wait_quiet()
        running.turns("four", "five", "six")
        assert running.server.wait_for_posts(2)
        running.server.wait_quiet()

    # The unreadable file counted as empty, was reported once, and was replaced.
    assert sum("unreadable" in r.getMessage() for r in caplog.records) == 1
    assert body_lines(running.posts()[0]) == ["- one", "- two", "- three"]
    assert captured_hashes(running) == [
        sha(p) for p in ("one", "two", "three", "four", "five", "six")
    ]

    state_dir = running.home / "kemory" / "captured"
    assert (running.home / "kemory").stat().st_mode & 0o777 == 0o700
    assert state_dir.stat().st_mode & 0o777 == 0o700
    name = hashlib.sha256(b"s1").hexdigest()[:32]
    assert [p.name for p in state_dir.iterdir()] == [f"{name}.json"]
    assert running.state_file().stat().st_mode & 0o777 == 0o600


def test_state_file_keeps_at_most_1000_hashes_oldest_dropped(start_provider: StartProvider) -> None:
    running = start_provider()
    old = [sha(f"old {n}") for n in range(1000)]
    running.state_file().parent.mkdir(parents=True)
    running.state_file().write_text(json.dumps({"captured": old}))

    running.turns("one", "two", "three")
    assert running.server.wait_for_posts(1)
    running.server.wait_quiet()

    assert captured_hashes(running) == [*old[3:], sha("one"), sha("two"), sha("three")]


@pytest.mark.parametrize("trigger", ["incremental", "flush"])
def test_capture_requests_carry_org_header(start_provider: StartProvider, trigger: str) -> None:
    running = start_provider()

    if trigger == "incremental":
        running.turns("one", "two", "three")
    else:
        running.turn("one")
        running.provider.on_session_end([])
    assert running.server.wait_for_posts(1)

    (post,) = running.posts()
    assert post.headers["X-Organization-ID"] == KEY_ORG_ID
    assert post.headers["X-API-Key"]
    assert running.server.paths()[0] == "/api/v1/me/orgs"

import json
import logging
import platform
import threading
import time
from collections.abc import Callable
from typing import Any

import pytest

from hermes_kemory.client import KemoryClient, KemoryError, KemoryInert
from hermes_kemory.config import KemoryConfig

from .fake_kemory import KEY_ORG_ID, FakeKemory

API_KEY = "kmk_live_SECRETSECRET0123"
SEARCH = "/api/v1/memories/search"
ME_ORGS = "/api/v1/me/orgs"


def make_client(
    server_url: str,
    *,
    org_pin: str | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> KemoryClient:
    config = KemoryConfig(api_key=API_KEY, base_url=server_url, org_pin=org_pin)
    return KemoryClient(config, clock=clock)


def search(client: KemoryClient, deadline: float = 5.0) -> Any:
    return client.request("POST", SEARCH, body={"query": "hello"}, deadline=deadline)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_every_request_carries_key_and_client_headers(fake_kemory: FakeKemory) -> None:
    client = make_client(fake_kemory.url)
    search(client)
    client.request("DELETE", "/api/v1/memories/abc", deadline=5)

    assert fake_kemory.paths() == [ME_ORGS, SEARCH, "/api/v1/memories/abc"]
    for recorded in fake_kemory.requests:
        assert recorded.headers["x-api-key"] == API_KEY
        assert recorded.headers["User-Agent"] == "hermes-kemory/0.1.0"
        assert recorded.headers["Accept"] == "application/json"
        client_header = recorded.headers["X-Kemory-Client"]
        assert client_header.startswith("hermes/")
        assert client_header.endswith(f"({platform.system()}; {platform.machine()})")
    lookup, post, delete = fake_kemory.requests
    assert post.headers["Content-Type"] == "application/json"
    assert post.body == {"query": "hello"}
    assert "content-type" not in lookup.headers
    assert "content-type" not in delete.headers


def test_org_header_on_every_request_after_me_orgs(fake_kemory: FakeKemory) -> None:
    client = make_client(fake_kemory.url)
    search(client)
    client.request("GET", "/api/v1/user/context?depth=l3", deadline=5)
    client.request("POST", "/api/v1/memories", body={"content": "x"}, deadline=5)

    assert client.org_id == KEY_ORG_ID
    lookup, *others = fake_kemory.requests
    assert lookup.path == ME_ORGS
    assert "x-organization-id" not in lookup.headers
    assert len(others) == 3
    assert {r.headers["X-Organization-ID"] for r in others} == {KEY_ORG_ID}
    assert others[1].path == "/api/v1/user/context?depth=l3"


def test_active_org_is_preferred_and_first_row_is_the_fallback(fake_kemory: FakeKemory) -> None:
    rows = [{"org_id": "org-a", "active": False}, {"org_id": "org-b", "active": True}]
    fake_kemory.reply("GET", ME_ORGS, 200, rows, times=1)
    first = make_client(fake_kemory.url)
    search(first)
    assert first.org_id == "org-b"

    fake_kemory.reply("GET", ME_ORGS, 200, [{"org_id": "org-a"}, {"org_id": "org-c"}], times=1)
    second = make_client(fake_kemory.url)
    search(second)
    assert second.org_id == "org-a"
    assert fake_kemory.requests[-1].headers["X-Organization-ID"] == "org-a"


def test_me_orgs_called_once_per_client(fake_kemory: FakeKemory) -> None:
    client = make_client(fake_kemory.url)
    errors: list[BaseException] = []

    def call() -> None:
        try:
            search(client)
        except BaseException as exc:
            errors.append(exc)

    workers = [threading.Thread(target=call) for _ in range(6)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    assert errors == []
    assert fake_kemory.paths().count(ME_ORGS) == 1
    assert fake_kemory.paths().count(SEARCH) == 6

    search(client)
    assert fake_kemory.paths().count(ME_ORGS) == 1
    other = make_client(fake_kemory.url)
    search(other)
    assert fake_kemory.paths().count(ME_ORGS) == 2


def test_org_lookup_failure_skips_then_retries_after_60s(fake_kemory: FakeKemory) -> None:
    clock = FakeClock()
    client = make_client(fake_kemory.url, clock=clock)
    fake_kemory.reply("GET", ME_ORGS, 503, {"detail": "down"}, times=1)

    with pytest.raises(KemoryError) as first:
        search(client)
    assert first.value.kind == "org_lookup"
    assert fake_kemory.paths() == [ME_ORGS]

    clock.now += 59
    with pytest.raises(KemoryError) as second:
        search(client)
    assert second.value.kind == "org_lookup"
    assert str(second.value) == str(first.value)
    assert fake_kemory.paths() == [ME_ORGS]

    clock.now += 2
    search(client)
    assert fake_kemory.paths() == [ME_ORGS, ME_ORGS, SEARCH]
    assert fake_kemory.requests[-1].headers["X-Organization-ID"] == KEY_ORG_ID


@pytest.mark.parametrize(
    "script",
    [
        pytest.param((200, []), id="empty-list"),
        pytest.param((200, {"orgs": []}), id="not-a-list"),
        pytest.param((200, [{"name": "no id"}]), id="row-without-org-id"),
        pytest.param((200, "not json at all"), id="not-json"),
        pytest.param((403, {"detail": "forbidden"}), id="forbidden"),
        pytest.param((429, None), id="rate-limited"),
    ],
)
def test_unusable_org_lookup_answers_are_org_lookup_errors(
    fake_kemory: FakeKemory, script: tuple[int, Any]
) -> None:
    fake_kemory.reply("GET", ME_ORGS, *script)
    client = make_client(fake_kemory.url)
    with pytest.raises(KemoryError) as exc_info:
        search(client)
    assert exc_info.value.kind == "org_lookup"
    assert not isinstance(exc_info.value, KemoryInert)
    assert client.inert_reason is None
    assert fake_kemory.paths() == [ME_ORGS]


def test_org_lookup_that_cannot_connect_is_an_org_lookup_error() -> None:
    with FakeKemory() as gone:
        url = gone.url
    clock = FakeClock()
    client = make_client(url, clock=clock)
    with pytest.raises(KemoryError) as exc_info:
        search(client)
    assert exc_info.value.kind == "org_lookup"


def test_org_lookup_that_times_out_is_an_org_lookup_error(fake_kemory: FakeKemory) -> None:
    fake_kemory.reply("GET", ME_ORGS, 200, [], delay=5)
    client = make_client(fake_kemory.url)
    started = time.monotonic()
    with pytest.raises(KemoryError) as exc_info:
        search(client, deadline=0.5)
    assert exc_info.value.kind == "org_lookup"
    assert time.monotonic() - started < 1.5


def test_org_pin_mismatch_goes_inert_and_names_both_ids(fake_kemory: FakeKemory) -> None:
    client = make_client(fake_kemory.url, org_pin="org-other")
    with pytest.raises(KemoryInert) as exc_info:
        search(client)
    assert exc_info.value.kind == "org_mismatch"
    assert "org-other" in str(exc_info.value)
    assert KEY_ORG_ID in str(exc_info.value)
    assert client.inert_reason == str(exc_info.value)
    assert fake_kemory.paths() == [ME_ORGS]

    with pytest.raises(KemoryInert):
        client.request("POST", "/api/v1/memories", body={"content": "x"}, deadline=5)
    with pytest.raises(KemoryInert):
        client.request("DELETE", "/api/v1/memories/abc", deadline=5)
    assert fake_kemory.paths() == [ME_ORGS]


def test_org_pin_that_matches_the_key_org_is_allowed(fake_kemory: FakeKemory) -> None:
    client = make_client(fake_kemory.url, org_pin=KEY_ORG_ID)
    assert search(client) == {"items": []}
    assert client.inert_reason is None


def test_401_goes_inert_without_further_requests(fake_kemory: FakeKemory) -> None:
    fake_kemory.reply("POST", SEARCH, 401, {"detail": "bad key"})
    client = make_client(fake_kemory.url)
    with pytest.raises(KemoryInert) as exc_info:
        search(client)
    assert exc_info.value.kind == "unauthorized"
    assert exc_info.value.status == 401
    assert "KEMORY_API_KEY" in str(exc_info.value)
    assert "await approval" in str(exc_info.value)
    assert client.inert_reason == str(exc_info.value)
    sent = len(fake_kemory.requests)

    with pytest.raises(KemoryInert):
        client.request("GET", "/api/v1/user/context", deadline=5)
    with pytest.raises(KemoryInert):
        search(client)
    assert len(fake_kemory.requests) == sent


def test_401_from_me_orgs_goes_inert(fake_kemory: FakeKemory) -> None:
    fake_kemory.reply("GET", ME_ORGS, 401, {"detail": "pending_approval"})
    client = make_client(fake_kemory.url)
    with pytest.raises(KemoryInert) as exc_info:
        search(client)
    assert exc_info.value.kind == "unauthorized"
    with pytest.raises(KemoryInert):
        search(client)
    assert fake_kemory.paths() == [ME_ORGS]


@pytest.mark.parametrize(
    ("target_hosts", "expected"),
    [
        pytest.param(["eu.kemory.example"], '["eu.kemory.example"]', id="list"),
        pytest.param({"api": "eu.kemory.example"}, '{"api": "eu.kemory.example"}', id="object"),
        pytest.param(None, "no target hosts were given", id="null"),
    ],
)
def test_403_residency_goes_inert_and_names_target_hosts(
    fake_kemory: FakeKemory, target_hosts: Any, expected: str
) -> None:
    detail = {"code": "ORG_WRONG_RESIDENCY_CELL", "target_hosts": target_hosts}
    fake_kemory.reply("POST", SEARCH, 403, {"detail": detail})
    client = make_client(fake_kemory.url)
    with pytest.raises(KemoryInert) as exc_info:
        search(client)
    assert exc_info.value.kind == "wrong_region"
    assert expected in str(exc_info.value)
    assert "KEMORY_URL" in str(exc_info.value)
    sent = len(fake_kemory.requests)

    with pytest.raises(KemoryInert):
        search(client)
    assert len(fake_kemory.requests) == sent


def test_403_residency_from_me_orgs_goes_inert(fake_kemory: FakeKemory) -> None:
    detail = {"code": "ORG_WRONG_RESIDENCY_CELL", "target_hosts": ["eu.kemory.example"]}
    fake_kemory.reply("GET", ME_ORGS, 403, {"detail": detail})
    client = make_client(fake_kemory.url)
    with pytest.raises(KemoryInert) as exc_info:
        search(client)
    assert exc_info.value.kind == "wrong_region"
    assert fake_kemory.paths() == [ME_ORGS]


@pytest.mark.parametrize(
    ("status", "body"),
    [
        pytest.param(403, {"detail": "Forbidden"}, id="plain-403"),
        pytest.param(403, {"detail": {"code": "OTHER"}}, id="other-403-code"),
        pytest.param(404, None, id="404"),
        pytest.param(409, {"detail": "Namespace conflict"}, id="409"),
        pytest.param(422, {"detail": [{"msg": "bad"}]}, id="422"),
    ],
)
def test_other_client_errors_are_skippable_and_not_inert(
    fake_kemory: FakeKemory, status: int, body: Any
) -> None:
    fake_kemory.reply("POST", SEARCH, status, body, times=1)
    client = make_client(fake_kemory.url)
    with pytest.raises(KemoryError) as exc_info:
        search(client)
    assert not isinstance(exc_info.value, KemoryInert)
    assert exc_info.value.kind == "client"
    assert exc_info.value.status == status
    assert str(status) in str(exc_info.value)
    assert client.inert_reason is None
    assert search(client) == {"items": []}


def test_client_error_message_carries_the_servers_reason(fake_kemory: FakeKemory) -> None:
    fake_kemory.reply("POST", "/api/v1/memories", 409, {"detail": "Namespace conflict"})
    client = make_client(fake_kemory.url)
    with pytest.raises(KemoryError) as exc_info:
        client.request("POST", "/api/v1/memories", body={"content": "x"}, deadline=5)
    assert "Namespace conflict" in str(exc_info.value)


def test_search_query_over_1000_chars_is_a_client_error(fake_kemory: FakeKemory) -> None:
    client = make_client(fake_kemory.url)
    with pytest.raises(KemoryError) as exc_info:
        client.request("POST", SEARCH, body={"query": "x" * 1001}, deadline=5)
    assert (exc_info.value.kind, exc_info.value.status) == ("client", 422)
    client.request("POST", SEARCH, body={"query": "x" * 1000}, deadline=5)


def test_429_5xx_and_refused_connection_are_skippable(fake_kemory: FakeKemory) -> None:
    client = make_client(fake_kemory.url)
    cases = [(429, "rate_limited"), (500, "server"), (503, "server")]
    for status, kind in cases:
        fake_kemory.reply("POST", SEARCH, status, {"detail": "slow down"}, times=1)
        with pytest.raises(KemoryError) as exc_info:
            search(client)
        assert (exc_info.value.kind, exc_info.value.status) == (kind, status)
    assert client.inert_reason is None
    assert search(client) == {"items": []}

    with FakeKemory() as server:
        connected = make_client(server.url)
        search(connected)
    # The server is gone but the organisation is known, so the request itself is refused.
    with pytest.raises(KemoryError) as network_error:
        search(connected)
    assert network_error.value.kind == "network"
    assert connected.inert_reason is None


def test_non_json_success_body_is_bad_response(fake_kemory: FakeKemory) -> None:
    fake_kemory.reply("POST", SEARCH, 200, "<html>gateway page</html>", times=1)
    client = make_client(fake_kemory.url)
    with pytest.raises(KemoryError) as exc_info:
        search(client)
    assert (exc_info.value.kind, exc_info.value.status) == ("bad_response", 200)
    assert client.inert_reason is None
    assert search(client) == {"items": []}


def test_empty_success_body_is_none(fake_kemory: FakeKemory) -> None:
    client = make_client(fake_kemory.url)
    assert client.request("DELETE", "/api/v1/memories/abc", deadline=5) is None


def test_timeout_returns_within_deadline(fake_kemory: FakeKemory) -> None:
    fake_kemory.reply("POST", SEARCH, 200, {"items": []}, delay=10)
    client = make_client(fake_kemory.url)
    started = time.monotonic()
    with pytest.raises(KemoryError) as exc_info:
        search(client, deadline=1.0)
    elapsed = time.monotonic() - started
    assert exc_info.value.kind == "timeout"
    assert elapsed < 1.5
    assert client.inert_reason is None

    fake_kemory.reply("POST", SEARCH, 200, {"items": []})
    assert search(client) == {"items": []}


def test_deadline_covers_the_org_lookup_and_the_request_together(
    fake_kemory: FakeKemory,
) -> None:
    fake_kemory.reply("GET", ME_ORGS, 200, [{"org_id": KEY_ORG_ID}], delay=0.6)
    fake_kemory.reply("POST", SEARCH, 200, {"items": []}, delay=0.6)
    client = make_client(fake_kemory.url)
    started = time.monotonic()
    with pytest.raises(KemoryError) as exc_info:
        search(client, deadline=1.0)
    assert exc_info.value.kind == "timeout"
    assert time.monotonic() - started < 1.5


def test_redirect_not_followed(fake_kemory: FakeKemory) -> None:
    with FakeKemory() as elsewhere:
        fake_kemory.reply(
            "POST", SEARCH, 302, None, headers={"Location": f"{elsewhere.url}/stolen"}
        )
        client = make_client(fake_kemory.url)
        with pytest.raises(KemoryError) as exc_info:
            search(client)
        assert (exc_info.value.kind, exc_info.value.status) == ("client", 302)
        assert elsewhere.requests == []
    assert client.inert_reason is None


def test_api_key_never_logged_or_in_errors(
    fake_kemory: FakeKemory, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    messages: list[str] = []

    def collect(client: KemoryClient, **kwargs: Any) -> None:
        try:
            client.request(kwargs.pop("method", "POST"), kwargs.pop("path", SEARCH), **kwargs)
        except KemoryError as exc:
            messages.append(str(exc))
            assert API_KEY not in repr(exc)

    # A server that echoes the key back in its error text must not get it into messages or logs.
    for status, body in [
        (400, {"detail": f"bad key {API_KEY}"}),
        (401, {"detail": f"invalid {API_KEY}"}),
    ]:
        fake_kemory.reply("POST", SEARCH, status, body)
        collect(make_client(fake_kemory.url), body={"query": "q"}, deadline=5)
    fake_kemory.reply("POST", SEARCH, 500, {"detail": API_KEY})
    collect(make_client(fake_kemory.url), body={"query": "q"}, deadline=5)
    fake_kemory.reply("POST", SEARCH, 403, {"detail": {"code": "ORG_WRONG_RESIDENCY_CELL"}})
    collect(make_client(fake_kemory.url), body={"query": "q"}, deadline=5)
    fake_kemory.reply("POST", SEARCH, 200, API_KEY)
    collect(make_client(fake_kemory.url), body={"query": "q"}, deadline=5)
    collect(make_client(fake_kemory.url, org_pin="org-other"), body={"query": "q"}, deadline=5)
    fake_kemory.reply("POST", SEARCH, 200, {"items": []}, delay=5)
    collect(make_client(fake_kemory.url), body={"query": "q"}, deadline=0.3)
    with FakeKemory() as gone:
        down = make_client(gone.url)
    collect(down, body={"query": "q"}, deadline=5)

    assert len(messages) == 8
    assert caplog.records
    for record in caplog.records:
        assert API_KEY not in record.getMessage()
        assert API_KEY not in repr(record.args)
        assert record.exc_text is None or API_KEY not in record.exc_text
    assert all(API_KEY not in message for message in messages)


def test_each_error_kind_warns_once(
    fake_kemory: FakeKemory, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="hermes_kemory.client")
    client = make_client(fake_kemory.url)
    for status in (429, 429, 500, 500, 404, 404):
        fake_kemory.reply("POST", SEARCH, status, None, times=1)
        with pytest.raises(KemoryError):
            search(client)
    fake_kemory.reply("POST", SEARCH, 200, "not json", times=2)
    for _ in range(2):
        with pytest.raises(KemoryError):
            search(client)

    levels: dict[str, list[int]] = {}
    for record in caplog.records:
        kind = record.args[0] if isinstance(record.args, tuple) else None
        assert isinstance(kind, str)
        levels.setdefault(kind, []).append(record.levelno)
    assert levels == {
        "rate_limited": [logging.WARNING, logging.DEBUG],
        "server": [logging.WARNING, logging.DEBUG],
        "client": [logging.WARNING, logging.DEBUG],
        "bad_response": [logging.WARNING, logging.DEBUG],
    }
    assert client.last_error is not None
    assert client.last_error.startswith("bad_response:")


def test_inert_and_org_lookup_errors_also_warn_once(
    fake_kemory: FakeKemory, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="hermes_kemory.client")
    fake_kemory.reply("POST", SEARCH, 401, None, times=1)
    client = make_client(fake_kemory.url)
    for _ in range(3):
        with pytest.raises(KemoryInert):
            search(client)
    clock = FakeClock()
    flaky = make_client(fake_kemory.url, clock=clock)
    fake_kemory.reply("GET", ME_ORGS, 500, None, times=2)
    for pause in (0, 61, 0):
        clock.now += pause
        with pytest.raises(KemoryError):
            search(flaky)

    by_kind: dict[str, list[int]] = {}
    for record in caplog.records:
        assert isinstance(record.args, tuple)
        by_kind.setdefault(str(record.args[0]), []).append(record.levelno)
    assert by_kind["unauthorized"] == [logging.WARNING, logging.DEBUG, logging.DEBUG]
    assert by_kind["org_lookup"][0] == logging.WARNING
    assert set(by_kind["org_lookup"][1:]) == {logging.DEBUG}


def test_unexpected_failures_surface_as_kemory_error(fake_kemory: FakeKemory) -> None:
    client = make_client(fake_kemory.url)
    with pytest.raises(KemoryError) as exc_info:
        client.request("POST", SEARCH, body={"query": object()}, deadline=5)
    assert exc_info.value.kind == "client"
    assert json.dumps(str(exc_info.value))


def test_join_bounds_a_call_that_ignores_its_socket_timeout(
    fake_kemory: FakeKemory, monkeypatch: pytest.MonkeyPatch
) -> None:
    # urllib's timeout applies to each socket operation, so a server that keeps trickling bytes
    # never trips it; only the deadline on the worker bounds such a call.
    client = make_client(fake_kemory.url)
    search(client)
    release = threading.Event()

    def trickle(*args: Any, **kwargs: Any) -> None:
        release.wait(5)

    monkeypatch.setattr(client, "_http", trickle)
    started = time.monotonic()
    try:
        with pytest.raises(KemoryError) as exc_info:
            search(client, deadline=0.5)
        assert exc_info.value.kind == "timeout"
        assert time.monotonic() - started < 1.5
    finally:
        release.set()

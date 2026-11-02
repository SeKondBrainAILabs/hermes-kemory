"""A small stand-in for the Kemory API, for tests that must not reach the real one."""

import json
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Any
from urllib.parse import urlsplit

KEY_ORG_ID = "org-key-0001"

# Kemory rejects a search query longer than this with a 422.
MAX_SEARCH_QUERY_CHARS = 1000


class Headers(Mapping[str, str]):
    """Request headers with case-insensitive names."""

    def __init__(self, pairs: list[tuple[str, str]]) -> None:
        self._values = {name.lower(): value for name, value in pairs}

    def __getitem__(self, name: str) -> str:
        return self._values[name.lower()]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


@dataclass(frozen=True)
class RecordedRequest:
    method: str
    path: str  # with the query string
    headers: Headers
    body: Any  # parsed JSON, or None when there was no (valid) JSON body


@dataclass
class _Scripted:
    method: str
    path: str
    status: int
    body: Any
    delay: float
    times: int | None
    headers: dict[str, str]


class _Server(ThreadingHTTPServer):
    # A handler sleeping on a scripted delay must never hold up stop().
    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        """Clients that give up on a delayed answer are expected; keep their errors quiet."""


class FakeKemory:
    def __init__(self) -> None:
        self.requests: list[RecordedRequest] = []
        self._scripted: list[_Scripted] = []
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._created = 0
        self._server = _Server(("127.0.0.1", 0), _make_handler(self))
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def __enter__(self) -> "FakeKemory":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stopped.set()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def reply(
        self,
        method: str,
        path: str,
        status: int,
        body: Any = None,
        *,
        delay: float = 0.0,
        times: int | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Script an answer for method and path (with or without the query string).

        The newest matching script wins and is dropped after `times` uses (None: for good).
        A str or bytes body is sent as it is, anything else as JSON, None as an empty body.
        """
        with self._lock:
            self._scripted.append(
                _Scripted(method.upper(), path, status, body, delay, times, headers or {})
            )

    def paths(self, method: str | None = None) -> list[str]:
        """The recorded paths without their query strings, in order."""
        return [
            urlsplit(r.path).path for r in self.requests if method is None or r.method == method
        ]

    def _answer(self, request: RecordedRequest) -> _Scripted:
        route = urlsplit(request.path).path
        with self._lock:
            for scripted in reversed(self._scripted):
                if scripted.method != request.method or scripted.path not in (request.path, route):
                    continue
                if scripted.times is not None:
                    scripted.times -= 1
                    if scripted.times <= 0:
                        self._scripted.remove(scripted)
                return scripted
            return self._default(request, route)

    def _default(self, request: RecordedRequest, route: str) -> _Scripted:
        def answer(status: int, body: Any = None) -> _Scripted:
            return _Scripted(request.method, route, status, body, 0.0, None, {})

        method = request.method
        if (method, route) == ("GET", "/api/v1/me/orgs"):
            return answer(
                200,
                [
                    {
                        "org_id": KEY_ORG_ID,
                        "name": "Key org",
                        "org_type": "personal",
                        "role": "owner",
                        "active": True,
                    }
                ],
            )
        if (method, route) == ("GET", "/api/v1/user/context"):
            return answer(
                200,
                {
                    "namespaces": [
                        {"namespace": "user:profile", "summary": "Prefers short answers."}
                    ]
                },
            )
        if (method, route) == ("POST", "/api/v1/memories/search"):
            query = request.body.get("query") if isinstance(request.body, dict) else None
            if isinstance(query, str) and len(query) > MAX_SEARCH_QUERY_CHARS:
                return answer(
                    422, {"detail": [{"loc": ["body", "query"], "msg": "String too long"}]}
                )
            return answer(200, {"items": []})
        if (method, route) == ("POST", "/api/v1/memories"):
            self._created += 1
            namespace = request.body.get("namespace") if isinstance(request.body, dict) else None
            return answer(
                201, {"memory_id": f"mem-{self._created}", "namespace": namespace or "general"}
            )
        if method == "DELETE" and route.startswith("/api/v1/memories/"):
            return answer(204)
        return answer(404, {"detail": "Not Found"})


def _make_handler(fake: FakeKemory) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _handle(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                body = json.loads(raw) if raw else None
            except ValueError:
                body = None
            request = RecordedRequest(
                self.command, self.path, Headers(list(self.headers.items())), body
            )
            fake.requests.append(request)
            scripted = fake._answer(request)
            if scripted.delay and fake._stopped.wait(scripted.delay):
                return
            payload = _encode(scripted.body)
            self.send_response(scripted.status)
            for name, value in scripted.headers.items():
                self.send_header(name, value)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        do_GET = do_POST = do_DELETE = do_PUT = do_PATCH = _handle

        def log_message(self, format: str, *args: Any) -> None:
            pass

    return Handler


def _encode(body: Any) -> bytes:
    if body is None:
        return b""
    if isinstance(body, bytes):
        return body
    if isinstance(body, str):
        return body.encode("utf-8")
    return json.dumps(body).encode("utf-8")

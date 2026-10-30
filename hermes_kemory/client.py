"""HTTP client for the Kemory API.

Every call has a hard deadline, carries the organisation header and never raises anything but
KemoryError. A client that learns its key is unusable (rejected, or in another region, or pinned to
a different organisation) goes inert and sends nothing more.
"""

import json
import logging
import platform
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agent.memory_provider import spawn_context_thread

from . import __version__
from .config import API_KEY_ENV, URL_ENV, KemoryConfig

logger = logging.getLogger(__name__)

ME_ORGS_PATH = "/api/v1/me/orgs"

# After a failed organisation lookup the client stays quiet this long, so a down API is not
# hammered by every prompt.
ORG_LOOKUP_RETRY_SECONDS = 60.0

_RESIDENCY_CODE = "ORG_WRONG_RESIDENCY_CELL"
_MAX_BODY_BYTES = 8 * 1024 * 1024
_MAX_DETAIL_CHARS = 200


class KemoryError(Exception):
    """A Kemory call failed and should be skipped; the message is safe to show to the user.

    kind is one of 'timeout', 'network', 'rate_limited', 'server', 'client', 'org_lookup' and
    'bad_response'. status is the HTTP status when there was one.
    """

    def __init__(self, kind: str, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.status = status


class KemoryInert(KemoryError):
    """The key cannot work, so the client stops sending requests.

    kind is one of 'unauthorized', 'wrong_region' and 'org_mismatch'. str() is the one line that
    tells the user what to fix.
    """


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    """Refuse redirects: urllib would resend X-API-Key to wherever the server points."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


@dataclass(frozen=True)
class _Reply:
    status: int
    body: bytes


class KemoryClient:
    def __init__(
        self, config: KemoryConfig, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._config = config
        # The clock only times the organisation-lookup pause; deadlines always use real time.
        self._clock = clock
        self._opener = urllib.request.build_opener(_NoRedirects)
        self._headers = {
            "X-API-Key": config.api_key,
            "User-Agent": f"hermes-kemory/{__version__}",
            "X-Kemory-Client": f"hermes/{_hermes_version()} ({_os_name()}; {_machine()})",
            "Accept": "application/json",
        }
        # state_lock guards the fields below; lookup_lock lets one thread at a time look the
        # organisation up while the others wait within their own deadlines.
        self._state_lock = threading.Lock()
        self._lookup_lock = threading.Lock()
        self._org_id: str | None = None
        self._inert: KemoryInert | None = None
        self._lookup_failure: tuple[float, KemoryError] | None = None
        self._seen_kinds: set[str] = set()
        self._last_error: str | None = None

    @property
    def org_id(self) -> str | None:
        """The organisation the API key belongs to, once it has been looked up."""
        with self._state_lock:
            return self._org_id

    @property
    def inert_reason(self) -> str | None:
        with self._state_lock:
            return str(self._inert) if self._inert else None

    @property
    def last_error(self) -> str | None:
        with self._state_lock:
            return self._last_error

    def request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        deadline: float,
    ) -> Any:
        """Call Kemory and return the parsed JSON (None for an empty body).

        deadline is the number of seconds the whole call may take, the organisation lookup
        included. Raises only KemoryError, or KemoryInert once the key is known to be unusable.
        """
        end = time.monotonic() + deadline
        try:
            org_id = self._ensure_org(end)
            return self._exchange(method, path, org_id, body, end)
        except KemoryError as exc:
            self._record(exc)
            raise
        except Exception as exc:  # the contract is KemoryError only, whatever goes wrong inside
            wrapped = KemoryError("client", f"Kemory request could not be sent: {self._scrub(exc)}")
            self._record(wrapped)
            raise wrapped from exc

    def _ensure_org(self, end: float) -> str:
        """Return the key's organisation, looking it up on first use."""
        known = self._known_org()
        if known:
            return known
        if not self._lookup_lock.acquire(timeout=max(0.0, end - time.monotonic())):
            raise KemoryError("timeout", "Kemory did not answer in time.")
        try:
            known = self._known_org()
            if known:
                return known
            try:
                org_id = self._lookup_org(end)
            except KemoryInert:
                raise
            except KemoryError as exc:
                with self._state_lock:
                    self._lookup_failure = (self._clock() + ORG_LOOKUP_RETRY_SECONDS, exc)
                raise
            mismatch = self._pin_mismatch(org_id)
            with self._state_lock:
                self._lookup_failure = None
                self._org_id = org_id
                # Set together with the org, so a thread waiting on the lock cannot slip a
                # request through between the lookup and the refusal.
                if mismatch:
                    self._inert = mismatch
            if mismatch:
                raise mismatch
            return org_id
        finally:
            self._lookup_lock.release()

    def _known_org(self) -> str | None:
        """Return the cached organisation, or raise when the client may not send anything."""
        with self._state_lock:
            if self._inert:
                raise KemoryInert(self._inert.kind, str(self._inert), self._inert.status)
            if self._org_id:
                return self._org_id
            failure = self._lookup_failure
            if failure and self._clock() < failure[0]:
                raise KemoryError(failure[1].kind, str(failure[1]), failure[1].status)
        return None

    def _lookup_org(self, end: float) -> str:
        try:
            rows = self._exchange("GET", ME_ORGS_PATH, None, None, end)
        except KemoryInert:
            raise
        except KemoryError as exc:
            raise KemoryError(
                "org_lookup", f"Kemory organisation lookup failed: {exc}", exc.status
            ) from exc
        org_id = _pick_org(rows)
        if org_id is None:
            raise KemoryError(
                "org_lookup", "Kemory organisation lookup returned no usable organisation."
            )
        return org_id

    def _pin_mismatch(self, org_id: str) -> KemoryInert | None:
        pin = self._config.org_pin
        if not pin or pin == org_id:
            return None
        return KemoryInert(
            "org_mismatch",
            f"Kemory is set to organisation {pin} but this API key belongs to organisation "
            f"{org_id}, so nothing is sent. Fix the organisation setting or use a key from "
            f"organisation {pin}.",
        )

    def _exchange(
        self, method: str, path: str, org_id: str | None, body: dict[str, Any] | None, end: float
    ) -> Any:
        """Send one request and return its parsed JSON; org_id is None only for the lookup."""
        headers = dict(self._headers)
        if org_id is not None:
            headers["X-Organization-ID"] = org_id
        data: bytes | None = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        reply = self._send(method, self._config.base_url + path, headers, data, end)
        return self._interpret(reply)

    def _send(
        self, method: str, url: str, headers: dict[str, str], data: bytes | None, end: float
    ) -> _Reply:
        remaining = end - time.monotonic()
        if remaining <= 0:
            raise KemoryError("timeout", "Kemory did not answer in time.")
        outcome: list[_Reply | KemoryError] = []

        def work() -> None:
            try:
                outcome.append(self._http(method, url, headers, data, remaining))
            except KemoryError as exc:
                outcome.append(exc)
            except Exception as exc:  # keep the worker from dying without an outcome
                outcome.append(KemoryError("network", f"Kemory call failed: {self._scrub(exc)}"))

        worker = spawn_context_thread(work, name="kemory-http", daemon=True)
        worker.start()
        # urlopen's timeout applies to each socket operation, so only this join bounds the call;
        # a worker that overruns is abandoned and its late result is never read.
        worker.join(max(0.0, end - time.monotonic()))
        if worker.is_alive() or not outcome:
            raise KemoryError("timeout", "Kemory did not answer in time.")
        result = outcome[0]
        if isinstance(result, KemoryError):
            raise result
        return result

    def _http(
        self, method: str, url: str, headers: dict[str, str], data: bytes | None, timeout: float
    ) -> _Reply:
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with self._opener.open(req, timeout=timeout) as resp:
                return _Reply(resp.status, _read_limited(resp))
        except urllib.error.HTTPError as exc:
            try:
                return _Reply(exc.code, _read_limited(exc))
            finally:
                exc.close()
        except TimeoutError:
            raise KemoryError("timeout", "Kemory did not answer in time.") from None
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise KemoryError("timeout", "Kemory did not answer in time.") from None
            raise KemoryError(
                "network", f"Could not reach Kemory: {self._scrub(exc.reason)}"
            ) from None
        except OSError as exc:
            raise KemoryError("network", f"Could not reach Kemory: {self._scrub(exc)}") from None

    def _interpret(self, reply: _Reply) -> Any:
        status = reply.status
        if 200 <= status < 300:
            if not reply.body.strip():
                return None
            try:
                return json.loads(reply.body)
            except ValueError:
                raise KemoryError(
                    "bad_response",
                    f"Kemory answered HTTP {status} with a body that is not JSON.",
                    status,
                ) from None
        payload = _parse_json(reply.body)
        if status == 401:
            raise KemoryInert(
                "unauthorized",
                f"Kemory rejected the API key (HTTP 401). Check {API_KEY_ENV}; a newly created "
                "key may still await approval.",
                status,
            )
        if status == 403 and _is_residency(payload):
            raise KemoryInert("wrong_region", _residency_message(payload), status)
        detail = self._detail(payload)
        suffix = f": {detail}" if detail else ""
        if status == 429:
            raise KemoryError("rate_limited", f"Kemory is rate limiting (HTTP 429){suffix}", status)
        if status >= 500:
            raise KemoryError(
                "server", f"Kemory had a server error (HTTP {status}){suffix}", status
            )
        raise KemoryError("client", f"Kemory refused the request (HTTP {status}){suffix}", status)

    def _record(self, exc: KemoryError) -> None:
        """Remember the failure; only the first of each kind is worth a WARNING."""
        with self._state_lock:
            if isinstance(exc, KemoryInert) and self._inert is None:
                self._inert = exc
            first = exc.kind not in self._seen_kinds
            self._seen_kinds.add(exc.kind)
            self._last_error = f"{exc.kind}: {exc}"
        logger.log(
            logging.WARNING if first else logging.DEBUG,
            "Kemory call skipped (%s): %s",
            exc.kind,
            exc,
        )

    def _detail(self, payload: Any) -> str:
        """A short server-supplied reason (FastAPI's {"detail": "..."}), when there is one."""
        if isinstance(payload, dict) and isinstance(payload.get("detail"), str):
            return self._scrub(payload["detail"])[:_MAX_DETAIL_CHARS]
        return ""

    def _scrub(self, value: object) -> str:
        """Text for a message or log line, with the API key removed in case a server echoed it."""
        return str(value).replace(self._config.api_key, "[REDACTED]")


def _read_limited(resp: Any) -> bytes:
    data: bytes = resp.read(_MAX_BODY_BYTES + 1)
    if len(data) > _MAX_BODY_BYTES:
        raise KemoryError("bad_response", "Kemory answered with an oversized body.", resp.status)
    return data


def _parse_json(raw: bytes) -> Any:
    try:
        return json.loads(raw)
    except ValueError:
        return None


def _pick_org(rows: Any) -> str | None:
    """The org_id of the row marked active, else of the first row."""
    if not isinstance(rows, list):
        return None
    dicts = [row for row in rows if isinstance(row, dict)]
    chosen = next((row for row in dicts if row.get("active") is True), dicts[0] if dicts else None)
    org_id = chosen.get("org_id") if chosen else None
    return org_id if isinstance(org_id, str) and org_id else None


def _is_residency(payload: Any) -> bool:
    detail = payload.get("detail") if isinstance(payload, dict) else None
    return isinstance(detail, dict) and detail.get("code") == _RESIDENCY_CODE


def _residency_message(payload: Any) -> str:
    hosts = payload["detail"].get("target_hosts")
    where = "no target hosts were given" if hosts is None else f"target hosts: {json.dumps(hosts)}"
    return (
        f"This API key's organisation is hosted in another Kemory region ({where}). "
        f"Set {URL_ENV} to the right host."
    )


def _hermes_version() -> str:
    try:
        import hermes_cli

        return str(hermes_cli.__version__) or "unknown"
    except Exception:  # any failure to read the version just means it is not reported
        return "unknown"


def _os_name() -> str:
    return platform.system() or "unknown"


def _machine() -> str:
    return platform.machine() or "unknown"

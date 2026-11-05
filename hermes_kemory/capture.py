"""Capture what the user typed in a Hermes session into Kemory, at most once per prompt.

Prompts are collected per session as Hermes reports each turn. Once enough new ones are waiting, or
a flush point is reached (compression, session end or switch, shutdown), one digest of the latest
prompts is posted by a single writer thread. A prompt is remembered as captured, by its hash, only
after Kemory accepted the digest, so a failed post is simply tried again with the next capture.
"""

import hashlib
import json
import logging
import os
import queue
import re
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from agent.memory_provider import spawn_context_thread

from .client import KemoryClient, KemoryError, KemoryInert
from .config import KemoryConfig
from .redact import redact

__all__ = [
    "CAPTURE_NAMESPACE",
    "MAX_BODY_CHARS",
    "MAX_CAPTURED_HASHES",
    "MAX_PROMPTS_PER_POST",
    "MEMORIES_PATH",
    "Capture",
    "prompt_from_content",
    "write_private_json",
]

logger = logging.getLogger(__name__)

MEMORIES_PATH = "/api/v1/memories"
CAPTURE_NAMESPACE = "user:sessions"
CAPTURE_TAG = "session-capture"
MAX_PROMPTS_PER_POST = 12
MAX_BODY_CHARS = 8000
MAX_CAPTURED_HASHES = 1000
# How long shutdown waits for the writer to finish its queue.
SHUTDOWN_JOIN_SECONDS = 5.0

INCREMENTAL: Final = "incremental"
FLUSH: Final = "flush"
WHEN_MID_SESSION = "mid-session"
WHEN_BEFORE_COMPACTION = "before compaction"
WHEN_SESSION_END = "at session end"

# What Hermes and this plugin wrap around recalled memory. All of it is Kemory's own output coming
# back, so none of it may be saved again as something the user said. The patterns are ported from
# Hermes rather than imported: the module that holds them has moved between releases.
_KEMORY_BLOCK_RE = re.compile(
    r"<\s*kemory-context\s*>[\s\S]*?</\s*kemory-context\s*>", re.IGNORECASE
)
_MEMORY_BLOCK_RE = re.compile(
    r"<\s*memory-context\s*>[\s\S]*?</\s*memory-context\s*>", re.IGNORECASE
)
_SYSTEM_NOTE_RE = re.compile(
    r"\[System note:\s*The following is recalled memory context,\s*NOT new user input\.[^\]]*\]\s*",
    re.IGNORECASE,
)
_FENCE_TAG_RE = re.compile(r"</?\s*memory-context\s*>", re.IGNORECASE)
_INJECTED_PATTERNS = (_KEMORY_BLOCK_RE, _MEMORY_BLOCK_RE, _SYSTEM_NOTE_RE, _FENCE_TAG_RE)

# What the writer thread is told to do: post a session's pending prompts, adopt another session's
# captured set, or stop.
_STOP: Final = object()


@dataclass(frozen=True)
class _Post:
    session_id: str
    kind: str  # INCREMENTAL or FLUSH
    when: str


@dataclass(frozen=True)
class _Adopt:
    parent: str
    child: str


def prompt_from_content(user_content: object) -> str | None:
    """The prompt worth capturing from one turn's user text, or None.

    Recalled memory and the notes around it are removed first; a prompt that is then empty, or
    that is a slash command, is not captured.
    """
    if not isinstance(user_content, str):
        return None
    text = user_content
    for pattern in _INJECTED_PATTERNS:
        text = pattern.sub("", text)
    text = text.strip()
    if not text or text.startswith("/"):
        return None
    return text


def write_private_json(path: Path, values: dict[str, Any]) -> None:
    """Replace path with values as JSON that only its owner can read."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            os.fchmod(tmp.fileno(), 0o600)
            json.dump(values, tmp, indent=2, sort_keys=True)
            tmp.write("\n")
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


class Capture:
    """Pending and captured prompts for the sessions of one provider instance."""

    def __init__(
        self,
        client: KemoryClient,
        config: KemoryConfig,
        hermes_home: str,
        *,
        platform: str,
        cwd: str | None,
    ) -> None:
        self._client = client
        self._config = config
        self._state_dir = Path(hermes_home) / "kemory" / "captured"
        self._platform = platform
        self._cwd = cwd
        self._lock = threading.Lock()
        # Hashes of captured prompts per session, oldest first (a dict keeps insertion order).
        self._captured: dict[str, dict[str, None]] = {}
        # Prompts waiting for a post per session, oldest first, as (hash, text).
        self._pending: dict[str, list[tuple[str, str]]] = {}
        self._jobs: queue.Queue[object] = queue.Queue()
        # Post jobs that are queued and not started yet, so a slow server cannot pile up copies.
        self._queued: set[_Post] = set()
        self._worker: threading.Thread | None = None
        self._closed = False
        self._warned_corrupt: set[Path] = set()

    # -- Called from the provider -------------------------------------------

    def add_turn(self, session_id: str, user_content: str, turn_author: object) -> None:
        """Note one turn's prompt; queue a post when enough new prompts are waiting."""
        if isinstance(turn_author, dict) and turn_author.get("is_bot"):
            return
        prompt = prompt_from_content(user_content)
        if prompt is None or not session_id or self._client.inert_reason:
            return
        digest = _hash(prompt)
        with self._lock:
            if self._closed:
                return
            if digest in self._captured_for(session_id):
                return
            pending = self._pending.setdefault(session_id, [])
            if any(known == digest for known, _ in pending):
                return
            pending.append((digest, prompt))
            del pending[:-MAX_CAPTURED_HASHES]
            ready = len(pending) >= self._config.capture_min_new_prompts
        if ready:
            self._enqueue(_Post(session_id, INCREMENTAL, WHEN_MID_SESSION))

    def flush(self, when: str, *, session_ids: tuple[str, ...] | None = None) -> None:
        """Queue a post of whatever is pending: for the given sessions, or for all of them."""
        with self._lock:
            if self._closed:
                return
            targets = session_ids if session_ids is not None else tuple(self._pending)
        for session_id in targets:
            self._enqueue(_Post(session_id, FLUSH, when))

    def switch(self, old: str, new: str, *, parent: str, reset: bool) -> None:
        """The session id changed: flush the old session and, unless it is a reset, carry over
        what it had captured so the same prompts are not posted again under the new id."""
        olds = tuple(dict.fromkeys(s for s in (old, parent) if s and s != new))
        self.flush(WHEN_SESSION_END, session_ids=olds)
        if reset or not new:
            return
        for source in olds:
            self._adopt(source, new)
            # The flush above may still be on its way; adopt again once it has run.
            self._enqueue(_Adopt(source, new))

    def shutdown(self, timeout: float = SHUTDOWN_JOIN_SECONDS) -> None:
        """Queue a last post of everything pending and wait for the writer, but not for long."""
        with self._lock:
            if self._closed:
                return
            targets = tuple(self._pending)
        for session_id in targets:
            self._enqueue(_Post(session_id, FLUSH, WHEN_SESSION_END))
        with self._lock:
            self._closed = True
            worker = self._worker
        if worker is None:
            return
        self._jobs.put(_STOP)
        worker.join(timeout)
        if worker.is_alive():
            logger.warning("Kemory capture did not finish in %.0f s and was left behind", timeout)

    # -- The writer thread --------------------------------------------------

    def _enqueue(self, job: object) -> None:
        with self._lock:
            if self._closed and not isinstance(job, _Post):
                return
            if isinstance(job, _Post):
                if job in self._queued:
                    return
                self._queued.add(job)
            if self._worker is None:
                self._worker = spawn_context_thread(
                    self._run_jobs, name="kemory-capture", daemon=True
                )
                self._worker.start()
        self._jobs.put(job)

    def _run_jobs(self) -> None:
        while True:
            job = self._jobs.get()
            if job is _STOP:
                return
            try:
                if isinstance(job, _Post):
                    with self._lock:
                        self._queued.discard(job)
                    self._post(job)
                elif isinstance(job, _Adopt):
                    self._adopt(job.parent, job.child)
            except Exception:
                logger.warning("Kemory capture failed and was skipped", exc_info=True)

    def _post(self, job: _Post) -> None:
        with self._lock:
            captured = self._captured_for(job.session_id)
            pending = [
                (h, p) for h, p in self._pending.get(job.session_id, []) if h not in captured
            ]
            self._pending[job.session_id] = pending
            if not pending:
                return
            if job.kind == INCREMENTAL and len(pending) < self._config.capture_min_new_prompts:
                return
            batch = list(pending)
        prompts = [text for _, text in batch[-MAX_PROMPTS_PER_POST:]]
        body = {
            "namespace": CAPTURE_NAMESPACE,
            "namespace_tag": CAPTURE_TAG,
            "visibility": "user-private",
            "content": self._content(prompts, job.when),
            "content_type": "text",
            "session_id": job.session_id,
            "metadata": {
                "source": "hermes",
                "capture": "auto",
                "capture_kind": job.kind,
                "turns": len(prompts),
                "platform": self._platform,
            },
        }
        try:
            self._client.request(
                "POST", MEMORIES_PATH, body=body, deadline=self._config.write_timeout
            )
        except KemoryInert:
            return  # the client has said why; nothing more is sent
        except KemoryError as exc:
            # The prompts stay pending and go out with the next capture.
            logger.debug("Kemory capture skipped (%s): %s", exc.kind, exc)
            return
        done = {h for h, _ in batch}
        with self._lock:
            self._pending[job.session_id] = [
                (h, p) for h, p in self._pending.get(job.session_id, []) if h not in done
            ]
            self._mark(job.session_id, [h for h, _ in batch])

    def _content(self, prompts: list[str], when: str) -> str:
        body = redact("\n".join(f"- {prompt}" for prompt in prompts))[:MAX_BODY_CHARS]
        return (
            "What was this Hermes session about? Session digest captured automatically "
            f"({when}, user prompts only, secrets redacted, {len(prompts)} prompts).\n\n"
            f"workspace: {self._cwd or 'unknown'}\n\n"
            f"{body}"
        )

    # -- The high-water mark ------------------------------------------------

    def _state_path(self, session_id: str) -> Path:
        name = hashlib.sha256(session_id.encode("utf-8", errors="replace")).hexdigest()[:32]
        return self._state_dir / f"{name}.json"

    def _captured_for(self, session_id: str) -> dict[str, None]:
        """The captured hashes of a session, read from disk the first time. Hold the lock."""
        known = self._captured.get(session_id)
        if known is None:
            known = dict.fromkeys(self._read_state(self._state_path(session_id)))
            self._captured[session_id] = known
        return known

    def _read_state(self, path: Path) -> list[str]:
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        except (OSError, ValueError) as exc:
            self._warn_corrupt(path, exc)
            return []
        try:
            data = json.loads(raw)
            hashes = data["captured"]
            if not isinstance(hashes, list) or not all(isinstance(h, str) for h in hashes):
                raise ValueError("captured is not a list of strings")
        except (ValueError, KeyError, TypeError, RecursionError) as exc:
            self._warn_corrupt(path, exc)
            return []
        return hashes

    def _warn_corrupt(self, path: Path, exc: Exception) -> None:
        if path in self._warned_corrupt:
            return
        self._warned_corrupt.add(path)
        logger.warning(
            "Kemory capture state %s is unreadable (%s) and is treated as empty", path, exc
        )

    def _mark(self, session_id: str, hashes: list[str]) -> None:
        """Remember prompts as captured, in memory and on disk. Hold the lock."""
        known = self._captured_for(session_id)
        known.update(dict.fromkeys(hashes))
        while len(known) > MAX_CAPTURED_HASHES:
            del known[next(iter(known))]
        self._save(session_id, known)

    def _adopt(self, parent: str, child: str) -> None:
        """Add the parent's captured hashes to the child's."""
        with self._lock:
            inherited = list(self._captured_for(parent))
            if inherited:
                self._mark(child, inherited)

    def _save(self, session_id: str, known: dict[str, None]) -> None:
        try:
            self._state_dir.parent.parent.mkdir(parents=True, exist_ok=True)
            for directory in (self._state_dir.parent, self._state_dir):
                directory.mkdir(mode=0o700, exist_ok=True)
                directory.chmod(0o700)  # mkdir's mode is cut by the umask
            write_private_json(self._state_path(session_id), {"captured": list(known)})
        except OSError as exc:
            # The mark still holds in memory for this run; it only fails to survive a restart.
            logger.warning("Kemory capture state could not be saved: %s", exc)

"""The four memory tools, and the parsing of Kemory's search and context replies.

Tool functions return plain dicts and raise KemoryError when Kemory fails; the provider turns
those into the JSON the agent sees.
"""

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from .client import KemoryClient
from .config import KemoryConfig
from .redact import redact

SEARCH_PATH = "/api/v1/memories/search"
MEMORIES_PATH = "/api/v1/memories"
CONTEXT_PATH = "/api/v1/user/context?depth=l3"

# Kemory rejects a search query longer than this with a 422.
MAX_QUERY_CHARS = 1000
MAX_HIT_CHARS = 1000
MAX_LIMIT = 20

# Memory text is shown inside a <kemory-context> block, so it must not be able to close it.
_CONTEXT_TAG_RE = re.compile(r"</?\s*kemory-context\s*>", re.IGNORECASE)

SUMMARIES_HEADER = "Profile summaries from Kemory, by namespace:"


@dataclass(frozen=True)
class Hit:
    memory_id: str
    namespace: str
    content: str


def cut_query(query: str) -> str:
    """The first MAX_QUERY_CHARS characters (not bytes) of a search query."""
    return query[:MAX_QUERY_CHARS]


def parse_hits(reply: Any, *, max_chars: int | None = None) -> list[Hit]:
    """Hits from a search reply, without empty items and duplicates.

    A hit repeats when its memory_id or its exact content was already seen. A reply of any other
    shape has no hits.
    """
    items = reply.get("items") if isinstance(reply, dict) else None
    if not isinstance(items, list):
        return []
    hits: list[Hit] = []
    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        content = _text(item.get("content"))
        if not content:
            continue
        memory_id = _text(item.get("memory_id"))
        if (memory_id and memory_id in seen_ids) or content in seen_content:
            continue
        if memory_id:
            seen_ids.add(memory_id)
        seen_content.add(content)
        if max_chars is not None:
            content = content[:max_chars]
        hits.append(Hit(memory_id, _text(item.get("namespace")), content))
    return hits


def hit_line(hit: Hit) -> str:
    label = f"[{_one_line(hit.namespace)}] " if hit.namespace else ""
    return f"- {label}{_neutralise(hit.content)}"


def parse_summaries(reply: Any) -> list[tuple[str, str]]:
    """(namespace, summary) pairs from a context reply, without empty ones."""
    rows = reply.get("namespaces") if isinstance(reply, dict) else None
    if not isinstance(rows, list):
        return []
    pairs: list[tuple[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        summary = _text(row.get("summary"))
        if summary:
            pairs.append((_text(row.get("namespace")), summary))
    return pairs


def render_summaries(summaries: list[tuple[str, str]], max_chars: int) -> str:
    """A header and one '- [namespace] summary' line per namespace, within max_chars.

    Lines are added in order until the next would pass max_chars, then a note says how many were
    left out. The note is not counted. No summaries give ''.
    """
    if not summaries:
        return ""
    lines = [SUMMARIES_HEADER]
    used = len(SUMMARIES_HEADER)
    shown = 0
    for namespace, summary in summaries:
        label = f"[{_one_line(namespace)}] " if namespace else ""
        line = f"- {label}{_neutralise(summary)}"
        if used + 1 + len(line) > max_chars:
            break
        lines.append(line)
        used += 1 + len(line)
        shown += 1
    left_out = len(summaries) - shown
    if left_out:
        noun = "namespace" if left_out == 1 else "namespaces"
        lines.append(f"({left_out} more {noun} left out to save space)")
    return "\n".join(lines)


SEARCH_SCHEMA: dict[str, Any] = {
    "name": "kemory_search",
    "description": (
        "Search the user's long-term Kemory memories by meaning. Use it to look up facts, "
        "preferences and earlier decisions that are not in this conversation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for, in plain words."},
            "namespace": {
                "type": "string",
                "description": "Only search this namespace, for example user:profile.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_LIMIT,
                "description": "How many memories to return at most.",
            },
        },
        "required": ["query"],
    },
}

STORE_SCHEMA: dict[str, Any] = {
    "name": "kemory_store",
    "description": (
        "Save something worth remembering to the user's Kemory memory: a fact, a preference or "
        "a decision. Write it as a self-contained sentence. Never save secrets."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The memory to save."},
            "namespace": {
                "type": "string",
                "description": "Where to file it. Leave out to let Kemory choose by topic.",
            },
        },
        "required": ["content"],
    },
}

FORGET_SCHEMA: dict[str, Any] = {
    "name": "kemory_forget",
    "description": (
        "Delete one memory from the user's Kemory memory by its memory_id, as returned by "
        "kemory_search or kemory_store. Only do this when the user asks."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "The id of the memory to delete."},
        },
        "required": ["memory_id"],
    },
}

CONTEXT_SCHEMA: dict[str, Any] = {
    "name": "kemory_context",
    "description": "Show the user's Kemory profile: a short summary of each memory namespace.",
    "parameters": {"type": "object", "properties": {}},
}

TOOL_SCHEMAS: list[dict[str, Any]] = [SEARCH_SCHEMA, STORE_SCHEMA, FORGET_SCHEMA, CONTEXT_SCHEMA]
TOOL_NAMES = frozenset(schema["name"] for schema in TOOL_SCHEMAS)


class ToolInputError(Exception):
    """The agent called a tool with arguments that cannot work; the message tells it why."""


def kemory_search(
    client: KemoryClient, config: KemoryConfig, args: dict[str, Any]
) -> dict[str, Any]:
    query = _required(args, "query")
    body: dict[str, Any] = {
        "query": cut_query(query),
        "limit": _limit(args.get("limit"), config.max_recall_results),
        "search_mode": "hybrid",
    }
    namespace = _text(args.get("namespace"))
    if namespace:
        body["namespace"] = namespace
    reply = client.request("POST", SEARCH_PATH, body=body, deadline=config.recall_timeout)
    return {
        "results": [
            {"memory_id": hit.memory_id, "namespace": hit.namespace, "content": hit.content}
            for hit in parse_hits(reply)
        ]
    }


def kemory_store(
    client: KemoryClient, config: KemoryConfig, args: dict[str, Any]
) -> dict[str, Any]:
    content = _required(args, "content")
    body: dict[str, Any] = {
        "content": redact(content),
        "content_type": "text",
        "metadata": {"source": "hermes", "capture": "tool"},
    }
    namespace = _text(args.get("namespace"))
    if namespace:
        body["namespace"] = namespace
    reply = client.request("POST", MEMORIES_PATH, body=body, deadline=config.write_timeout)
    created = reply if isinstance(reply, dict) else {}
    return {
        "stored": True,
        "memory_id": created.get("memory_id"),
        "namespace": created.get("namespace", namespace or None),
    }


def kemory_forget(
    client: KemoryClient, config: KemoryConfig, args: dict[str, Any]
) -> dict[str, Any]:
    memory_id = _required(args, "memory_id")
    client.request(
        "DELETE", f"{MEMORIES_PATH}/{quote(memory_id, safe='')}", deadline=config.write_timeout
    )
    return {"forgotten": True, "memory_id": memory_id}


def kemory_context(
    client: KemoryClient, config: KemoryConfig, args: dict[str, Any]
) -> dict[str, Any]:
    reply = client.request("GET", CONTEXT_PATH, deadline=config.recall_timeout)
    return {"context": render_summaries(parse_summaries(reply), config.context_max_chars)}


def _required(args: dict[str, Any], name: str) -> str:
    value = args.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ToolInputError(f"{name} is required and must be a non-empty string.")
    return value


def _limit(value: Any, default: int) -> int:
    # bool is an int subclass but never a meaningful count.
    if not isinstance(value, int) or isinstance(value, bool):
        return default
    return max(1, min(MAX_LIMIT, value))


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _one_line(value: str) -> str:
    return " ".join(value.split())


def _neutralise(text: str) -> str:
    return _CONTEXT_TAG_RE.sub("", text)

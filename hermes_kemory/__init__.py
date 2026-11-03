"""Kemory memory provider for Hermes Agent."""

from typing import Any

# Defined before the import below: the client reads it for its User-Agent header.
__version__ = "0.1.0"

from .provider import KemoryMemoryProvider


def register(ctx: Any) -> None:
    """Hermes calls this once per agent; each agent needs its own provider instance."""
    ctx.register_memory_provider(KemoryMemoryProvider())

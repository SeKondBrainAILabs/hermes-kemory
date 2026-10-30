"""Removes credentials from text before it leaves the machine."""

import re

REDACTED = "[REDACTED]"

# ASCII-only \w and \s keep the pattern identical to the one used by Kemory's other agent hooks,
# so a prompt is redacted the same way whichever host captures it.
_CREDENTIAL = re.compile(
    r"(bearer\s+[\w\-\.]{8,}"
    r"|(?:api[_-]?key|api[_-]?token|access[_-]?token|auth[_-]?token|token|secret|password|passwd|pwd)"
    r"\s*[:=]\s*[\"']?[^\s\"',;}]{6,}"
    r"|sk-[A-Za-z0-9]{16,}"
    r"|gh[pousr]_[A-Za-z0-9]{20,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|xox[baprs]-[A-Za-z0-9-]{10,}"
    r"|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"
    r"|-----BEGIN[^-]+PRIVATE KEY-----)",
    re.IGNORECASE | re.ASCII,
)


def redact(text: str) -> str:
    """Return text with every credential-shaped span replaced by [REDACTED]."""
    return _CREDENTIAL.sub(REDACTED, text)

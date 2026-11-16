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


# A private key is secret in its body, not its header, so this takes the whole block, BEGIN line to
# END line (PEM, OpenSSH and PGP). It runs as its own pass before _CREDENTIAL: inside _CREDENTIAL, a
# keyword such as `password=` starts earlier in the text than the BEGIN line, wins the leftmost
# match and eats only "-----BEGIN". The pattern is the one Kemory's Claude Code plugin uses.
# The lazy scan is capped at 16 KB per BEGIN (a 4096-bit RSA key is about 3.3 KB) so a prompt full
# of unterminated headers cannot make redaction quadratic. With no END line (a truncated paste) it
# takes the base64 runs and PEM/PGP header lines that follow, separated by whitespace or a literal
# backslash-n from a JSON-escaped key, and stops at anything else. The header names are listed on
# purpose: a general `Name:` would eat ordinary lines.
_PRIVATE_KEY = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----"
    r"(?:[\s\S]{0,16384}?-----END [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----"
    r"|(?:(?:\s|\\[rn])+(?:[A-Za-z0-9+/=]{16,}"
    r"|(?:Proc-Type|DEK-Info|Version|Comment|Hash|Charset):[^\n]*))*)",
    re.IGNORECASE,
)


def redact(text: str) -> str:
    """Return text with every credential-shaped span replaced by [REDACTED]."""
    return _CREDENTIAL.sub(REDACTED, _PRIVATE_KEY.sub(REDACTED, text))

import time

import pytest

from hermes_kemory.redact import REDACTED, redact

# Placeholder key material, not a real key.
B1 = "FAKEKEYMATERIALline1" + "A" * 44
B2 = "FAKEKEYMATERIALline2" + "B" * 44


def pem(label: str) -> str:
    return f"-----BEGIN {label}-----\n{B1}\n{B2}\n-----END {label}-----"


PEM = pem("RSA PRIVATE KEY")
LINES = ["-----BEGIN RSA PRIVATE KEY-----", B1, B2, "-----END RSA PRIVATE KEY-----"]


def test_private_key_body_is_redacted() -> None:
    out = redact(f"here is my key:\n{PEM}\nthanks")
    assert _body_is_gone(out)
    assert out == "here is my key:\n[REDACTED]\nthanks"


def _body_is_gone(out: str) -> bool:
    return B1 not in out and B2 not in out


ENCRYPTED_RSA = (
    "-----BEGIN RSA PRIVATE KEY-----\n"
    "Proc-Type: 4,ENCRYPTED\nDEK-Info: AES-128-CBC," + "0123456789ABCDEF" * 2 + "\n\n"
    f"{B1}\n{B2}\n-----END RSA PRIVATE KEY-----"
)
PGP = (
    "-----BEGIN PGP PRIVATE KEY BLOCK-----\nVersion: GnuPG v2\n\n"
    f"{B1}\n{B2}\n=abcd\n-----END PGP PRIVATE KEY BLOCK-----"
)
ONE_LINE = f"-----BEGIN RSA PRIVATE KEY----- {B1} {B2} -----END RSA PRIVATE KEY-----"


@pytest.mark.parametrize(
    "block",
    [
        pem("RSA PRIVATE KEY"),
        pem("PRIVATE KEY"),
        pem("EC PRIVATE KEY"),
        pem("OPENSSH PRIVATE KEY"),
        pem("ENCRYPTED PRIVATE KEY"),
        ENCRYPTED_RSA,
        PGP,
        ONE_LINE,
    ],
    ids=["rsa", "pkcs8", "ec", "openssh", "encrypted-pkcs8", "encrypted-rsa", "pgp", "one-line"],
)
def test_every_private_key_format_is_redacted(block: str) -> None:
    assert redact(f"why: {block} thanks") == "why: [REDACTED] thanks"


def test_whole_block_becomes_one_placeholder() -> None:
    out = redact(PEM)
    assert out == REDACTED
    assert "BEGIN" not in out and "END" not in out


@pytest.mark.parametrize(
    ("block", "expected"),
    [
        ("\n".join("  " + line for line in LINES), "why:\n  [REDACTED]\nthanks"),
        ("\n".join("> " + line for line in LINES), "why:\n> [REDACTED]\nthanks"),
        ("\r\n".join(LINES), "why:\n[REDACTED]\nthanks"),
    ],
    ids=["indented", "quoted", "crlf"],
)
def test_indented_quoted_and_crlf_blocks_are_redacted(block: str, expected: str) -> None:
    assert redact(f"why:\n{block}\nthanks") == expected


def test_header_case_does_not_matter() -> None:
    text = f"-----begin rsa private key-----\n{B1}\n-----end rsa private key-----"
    assert redact(text) == REDACTED


def test_key_after_a_keyword_is_redacted() -> None:
    assert redact(f"set password={PEM} done") == "set [REDACTED] done"
    for prefix in ("token: ", "JWT_SECRET=", 'client_secret: "'):
        out = redact(f"set {prefix}{PEM} done")
        assert _body_is_gone(out)
        assert out.endswith(" done")


def test_json_escaped_key_is_redacted() -> None:
    escaped = (
        '{"private_key": "-----BEGIN PRIVATE KEY-----\\n'
        + B1
        + "\\n"
        + B2
        + '\\n-----END PRIVATE KEY-----\\n", "client_email": "a@b.c"}'
    )
    out = redact(escaped)
    assert _body_is_gone(out)
    assert "client_email" in out

    cut_off = '{"private_key": "-----BEGIN PRIVATE KEY-----\\n' + B1 + "\\n" + B2
    assert _body_is_gone(redact(cut_off))


def test_unterminated_key_is_redacted() -> None:
    text = f"-----BEGIN RSA PRIVATE KEY-----\n{B1}\n{B2}\nwhat is wrong?"
    assert redact(text) == "[REDACTED]\nwhat is wrong?"


def test_two_blocks_stay_separate() -> None:
    assert redact(f"a {PEM} b {PEM} c") == "a [REDACTED] b [REDACTED] c"


@pytest.mark.parametrize("label", ["PUBLIC KEY", "CERTIFICATE"])
def test_public_keys_and_certificates_are_kept(label: str) -> None:
    text = pem(label)
    assert redact(text) == text


def test_many_lone_headers_stay_fast() -> None:
    text = "-----BEGIN RSA PRIVATE KEY-----\n" * 2000
    start = time.monotonic()
    out = redact(text)
    assert time.monotonic() - start < 5
    assert "BEGIN" not in out

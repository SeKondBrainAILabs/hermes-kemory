import pytest

from hermes_kemory.redact import redact

# Each sample is a full credential in the shape one alternative of the pattern matches.
CREDENTIALS = [
    "Bearer abcdef12345678",
    "bearer   abc.def-ghi_jkl",
    "api_key=abcdef123456",
    "api-key: abcdef123456",
    "apikey = abcdef123456",
    "api_token=abcdef123456",
    "access_token: abcdef123456",
    "auth_token=abcdef123456",
    "token=abcdef123456",
    "secret: abcdef123456",
    "password=hunter2hunter2",
    "passwd: hunter2hunter2",
    "pwd=hunter2hunter2",
    'password: "quoted-secret-value',
    "sk-" + "a1B2c3D4e5F6g7H8i9J0",
    "ghp_" + "A" * 20,
    "gho_" + "b" * 20,
    "ghu_" + "C" * 20,
    "ghs_" + "d" * 20,
    "ghr_" + "E" * 20,
    "AKIA" + "ABCDEFGHIJKLMNOP",
    "AKIA" + "0123456789ABCDEF",
    "xoxb-" + "1234567890",
    "xoxa-" + "1234567890",
    "xoxp-" + "abcdefghij-klm",
    "xoxr-" + "1234567890",
    "xoxs-" + "1234567890",
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkw.SflKxwRJSMeKKF2QT4fwpMeJf36P",
    "-----BEGIN RSA PRIVATE KEY-----",
    "-----BEGIN PRIVATE KEY-----",
    "-----BEGIN OPENSSH PRIVATE KEY-----",
]


@pytest.mark.parametrize("credential", CREDENTIALS)
def test_redacts_each_credential_shape(credential: str) -> None:
    assert redact(f"before {credential} after") == "before [REDACTED] after"


def test_redaction_is_case_insensitive() -> None:
    assert redact("BEARER ABCDEF12345678") == "[REDACTED]"
    assert redact("API_KEY=ABCDEF123456") == "[REDACTED]"
    assert redact("akia" + "abcdefghijklmnop") == "[REDACTED]"


def test_short_values_are_not_credentials() -> None:
    assert redact("password=short") == "password=short"
    assert redact("Bearer short") == "Bearer short"
    assert redact("sk-tooshort") == "sk-tooshort"


@pytest.mark.parametrize(
    "prose",
    [
        "the token expired yesterday",
        "please reset my password",
        "What does the secret ingredient do?",
        "Explain bearer of bad news",
        "",
    ],
)
def test_keeps_ordinary_prose(prose: str) -> None:
    assert redact(prose) == prose

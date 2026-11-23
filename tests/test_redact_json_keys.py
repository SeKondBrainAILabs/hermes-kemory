import pytest

from hermes_kemory.redact import redact

VALUE = "abcdef123456"

KEY_WORDS = [
    "api_key",
    "api-key",
    "apikey",
    "API_KEY",
    "api_token",
    "access_token",
    "auth_token",
    "token",
    "secret",
    "client_secret",
    "password",
    "passwd",
    "pwd",
]

# K stands for the key word and V for the value. Escaped shapes are raw strings so the
# backslashes are the ones written here.
SHAPES = {
    "double-quoted": '{"K": "V"}',
    "single-quoted": "{'K': 'V'}",
    "no-space": '{"K":"V"}',
    "space-before-colon": '{"K" : "V"}',
    "assign-double": 'K="V"',
    "assign-single": "K='V'",
    "quoted-key-bare-value": '"K"=V',
    "escaped-json": r"{\"K\": \"V\"}",
    "json-inside-json": r"{\\\"K\\\": \\\"V\\\"}",
    "with-neighbour": '{"K": "V", "user": "bob"}',
}


@pytest.mark.parametrize(
    "text",
    [
        '{"api_key": "abcdef123456"}',
        '{"password": "hunter2hunter2"}',
        "{'token': 'abcdef123456'}",
        'AWS_SECRET="abcdef123456"',
    ],
)
def test_quoted_key_value_pairs_are_redacted(text: str) -> None:
    out = redact(text)
    assert "abcdef123456" not in out
    assert "hunter2hunter2" not in out


@pytest.mark.parametrize("shape", SHAPES.values(), ids=SHAPES.keys())
@pytest.mark.parametrize("key", KEY_WORDS)
def test_every_key_word_in_every_quoted_shape_is_redacted(key: str, shape: str) -> None:
    text = shape.replace("K", key).replace("V", VALUE)
    assert VALUE not in redact(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"api_key": "abcdef123456"}', '{"[REDACTED]"}'),
        ("{'token': 'abcdef123456'}", "{'[REDACTED]'}"),
        ('{"password":"hunter2hunter2"}', '{"[REDACTED]"}'),
        ('{"password" : "hunter2hunter2"}', '{"[REDACTED]"}'),
        ('"token"=abcdef123456', '"[REDACTED]'),
        ('AWS_SECRET="abcdef123456"', 'AWS_[REDACTED]"'),
    ],
)
def test_quoted_key_exact_outputs(text: str, expected: str) -> None:
    assert redact(text) == expected


def test_quoted_key_redaction_keeps_neighbouring_keys() -> None:
    assert redact('{"token": "abcdef123456", "user": "bob"}') == '{"[REDACTED]", "user": "bob"}'


def test_quoted_key_with_short_value_is_kept() -> None:
    text = '{"password": "short"}'
    assert redact(text) == text


def test_quoted_key_word_in_prose_is_kept() -> None:
    text = 'the "token" field is empty'
    assert redact(text) == text


@pytest.mark.parametrize(
    "text",
    [
        '{"access_token": "Bearer 8f2c1a9b7d6e5f4a3b2c1d0e"}',
        '{"token": "Bearer 8f2c1a9b7d6e5f4a3b2c1d0e"}',
        "{'auth_token': 'Bearer 8f2c1a9b7d6e5f4a3b2c1d0e'}",
        r"{\"access_token\": \"Bearer 8f2c1a9b7d6e5f4a3b2c1d0e\"}",
        "token: Bearer 8f2c1a9b7d6e5f4a3b2c1d0e",
    ],
    ids=["access_token", "token", "single-quoted", "escaped", "unquoted"],
)
def test_quoted_key_with_bearer_value_redacts_the_token(text: str) -> None:
    assert "8f2c1a9b7d6e5f4a3b2c1d0e" not in redact(text)


def test_quoted_key_with_bearer_value_exact_output() -> None:
    out = redact('{"access_token": "Bearer 8f2c1a9b7d6e5f4a3b2c1d0e"}')
    assert out == '{"[REDACTED]"}'


# '=>' assigns in PHP arrays, Ruby hashes (symbol and string keys, and Hash#inspect output with
# no spaces) and Perl's fat comma. The last line is JSON-escaped text.
FAT_ARROW_SECRETS = [
    ("$db = ['password' => 'Tr0ub4dor-and-3'];", "Tr0ub4dor-and-3"),
    ('{ :api_key => "kmy_live_9f8e7d6c5b4a3210" }', "kmy_live_9f8e7d6c5b4a3210"),
    ('{"access_token"=>"q8Zr2LmN0pX4vB7cW1"}', "q8Zr2LmN0pX4vB7cW1"),
    ("password => hunter2supersecret", "hunter2supersecret"),
    (r"{\"secret\" => \"q8Zr2LmN0pX4vB7cW1\"}", "q8Zr2LmN0pX4vB7cW1"),
]


@pytest.mark.parametrize(("text", "value"), FAT_ARROW_SECRETS)
def test_values_after_fat_arrow_keys_are_redacted(text: str, value: str) -> None:
    out = redact(text)
    assert value not in out
    assert "[REDACTED]" in out

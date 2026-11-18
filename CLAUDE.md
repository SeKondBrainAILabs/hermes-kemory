# hermes-kemory

A Hermes memory-provider plugin backed by Kemory.

## Layout

- The Hermes plugin is `hermes_kemory/` (a catalog subdirectory), not the repo root. The repo root also holds the `.hermes/` checkout of Hermes.
- Never add `pyproject.toml` or other packaging files inside `hermes_kemory/`: Hermes copies a plugin's `pyproject.toml` into the user's plugin uv workspace.
- `hermes_kemory/__init__.py` must define `__version__` before importing `.provider`: `client.py` imports it for the User-Agent, and the other order is a circular import.

## Hermes as a dependency

- `hermes-agent` on PyPI is 0.19.0 (2026-07-20) and lacks `spawn_context_thread`. Use Hermes from a git tag (v0.22.x) for tests.
- Hermes cannot be a `git+` dependency: its `setup.py` raises "Building wheels or sdists for hermes-agent is not supported". Use an editable checkout in `.hermes/` as a uv path source.
- Hermes ships no `py.typed`. Point mypy's `mypy_path` at `.hermes`, set `follow_imports = "silent"` globally and set it back to `"normal"` for `hermes_kemory.*`. A per-module override misses the top-level modules Hermes imports (on v0.22.0 `agent/secret_scope.py` imports `utils`, `hermes_constants`, `pm` and others), and mypy then reports about 330 errors inside `.hermes/`.
- Hermes v0.22.3 declares every runtime dependency (python-dotenv, ruamel.yaml, openai, rich and others) only for `python_version >= '3.14'`. On Python 3.12 `uv sync` installs none of them and about 113 tests fail with `ModuleNotFoundError`, so `make check` fails. Create the dev venv with `uv sync -p 3.14` or run checks with `UV_PYTHON=3.14`.
- `uv.lock` is resolved against the default Hermes pin (v0.22.3). Running `uv sync` after fetching v0.22.0 rewrites `uv.lock`; run `git checkout -- uv.lock` before committing.
- PyYAML is not installed with Hermes v0.22.x (it uses ruamel.yaml). In tests, load YAML with `hermes_yaml.safe_load`.
- Validate a catalog entry with `python .hermes/scripts/validate_plugin_catalog.py <entry.yaml>` (from a test, run it with `sys.executable`). It needs a real 40-hex sha.

## Hermes behaviour

- Under a multiplexed gateway, Hermes `get_secret` raises `UnscopedSecretError` when no secret scope is active. Config loading must catch it and report a `ConfigError`.
- `MemoryManager.add_provider` calls `get_tool_schemas()` before `initialize()` to build tool routing. A provider must return its tools in that state, and may return `[]` only after `initialize()`.
- On a multimodal (list) user turn, Hermes v0.22.3 sets `original_user_message` to the same list it later appends turn context to (`agent/turn_context.py:1131`, `:987-1003`). So `sync_turn`'s `user_content` can carry other plugins' unfenced `pre_llm_call` text and Hermes notes, not only what the user typed.

## Tests

- Every test, even a pure unit test like `tests/test_redact.py`, needs the `.hermes` checkout: importing any `hermes_kemory` module runs `hermes_kemory/__init__.py`, which imports Hermes, and `tests/conftest.py` imports `agent.memory_manager`. Run `scripts/fetch-hermes.sh` before any test.
- `MemoryManager.handle_tool_call` redacts tool arguments before the provider sees them, so a test of the provider's own redaction must call `provider.handle_tool_call` directly.
- In `tests/fake_kemory.py`, `wait_quiet` must count from the call time, not only from the last request, or a "nothing was posted" assertion passes before the writer thread has sent anything.

## Plugin registration

- Hermes discovers a memory provider directory only if its `__init__.py` contains the text `register_memory_provider` or `MemoryProvider`.
- `register(ctx)` must call `ctx.register_memory_provider(<new instance>)`. A returned value is ignored.
- Validate the plugin with `hermes plugins validate hermes_kemory`, which also checks the declared capabilities. Never validate the repo root.

## Plugin sign-in (OAuth)

- Hermes plugin OAuth (`hermes_cli/plugin_oauth.py`) first ships in v0.22.3. `hermes_cli/plugin_provider_requests.py` already exists in v0.22.2 but raises `ValueError` for any non-core provider id.
- Detect plugin sign-in support by importing `hermes_cli.plugin_oauth`, not by whether `plugin_provider_requests` exists.
- `hermes auth logout <plugin>:<id>` only deletes the local grant; it revokes nothing at the identity provider.

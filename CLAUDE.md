# hermes-kemory

A Hermes memory-provider plugin backed by Kemory.

## Layout

- The Hermes plugin is `hermes_kemory/` (a catalog subdirectory), not the repo root. The repo root also holds the `.hermes/` checkout of Hermes.
- Never add `pyproject.toml` or other packaging files inside `hermes_kemory/`: Hermes copies a plugin's `pyproject.toml` into the user's plugin uv workspace.

## Hermes as a dependency

- `hermes-agent` on PyPI is 0.19.0 (2026-07-20) and lacks `spawn_context_thread`. Use Hermes from a git tag (v0.22.x) for tests.
- Hermes cannot be a `git+` dependency: its `setup.py` raises "Building wheels or sdists for hermes-agent is not supported". Use an editable checkout in `.hermes/` as a uv path source.
- Hermes ships no `py.typed`. Point mypy's `mypy_path` at `.hermes` and set `follow_imports = silent` for the Hermes packages.

## Plugin registration

- Hermes discovers a memory provider directory only if its `__init__.py` contains the text `register_memory_provider` or `MemoryProvider`.
- `register(ctx)` must call `ctx.register_memory_provider(<new instance>)`. A returned value is ignored.
- Validate the plugin with `hermes plugins validate hermes_kemory`, which also checks the declared capabilities. Never validate the repo root.

## Plugin sign-in (OAuth)

- Hermes plugin OAuth (`hermes_cli/plugin_oauth.py`) first ships in v0.22.3. `hermes_cli/plugin_provider_requests.py` already exists in v0.22.2 but raises `ValueError` for any non-core provider id.
- Detect plugin sign-in support by importing `hermes_cli.plugin_oauth`, not by whether `plugin_provider_requests` exists.
- `hermes auth logout <plugin>:<id>` only deletes the local grant; it revokes nothing at the identity provider.

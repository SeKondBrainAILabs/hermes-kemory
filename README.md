# hermes-kemory

Kemory long-term memory for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

## What it does

hermes-kemory is a Hermes memory provider backed by Kemory. Once you select it, Hermes:

- **recalls** a short profile summary at the start of a session and the memories that match each
  message you send,
- gives the agent four **memory tools** to search, save, forget and show your Kemory memories,
- **captures** what you type, with secrets redacted, into a private session namespace in Kemory.

It signs in with a Kemory API key. A Kemory error or timeout is logged and skipped. Every call has
a hard deadline (3 seconds for recall by default), so Kemory cannot stall or break the agent.

## Install

Hermes 0.22.0 or newer is required.

1. Install the plugin. Once it is listed in the Hermes plugin catalog:

   ```sh
   hermes plugins install kemory
   ```

   Until then, install it straight from GitHub:

   ```sh
   hermes plugins install SeKondBrainAILabs/hermes-kemory#hermes_kemory
   ```

   Or install it by hand: copy the `hermes_kemory/` directory of this repository to
   `$HERMES_HOME/plugins/kemory`. The directory must be named `kemory`, because Hermes uses the
   directory name as the provider name.

2. Select Kemory as the memory provider, either by setting `memory.provider: kemory` in
   `config.yaml` or by running `hermes memory setup` and choosing `kemory`.

3. Create a Kemory API key and set it as `KEMORY_API_KEY`, in the environment or in Hermes's
   `.env` file. `hermes memory setup` asks for it and saves it to `.env` for you.

4. Check the install:

   ```sh
   hermes memory status
   ```

   It should show `Provider: kemory`, `Plugin: installed` and `Status: available`. If the key is
   missing it shows `Status: not available` with `KEMORY_API_KEY` listed as missing.

   `hermes memory status` only checks that the key is set. It does not contact Kemory, so a wrong
   key still shows `available`. The `kemory config` lines it prints come from a provider that has not connected: `org_id` always
   reads `not looked up yet`, `last_error` always reads `none`, and `base_url` ignores
   `kemory.json` (it shows `KEMORY_URL`, else the default).

## Configuration

### Environment variables

Hermes reads these through its secret scope: from the environment or from Hermes's `.env` file.
Nothing is read from another tool's login files.

| Variable | Meaning |
| --- | --- |
| `KEMORY_API_KEY` | Your Kemory API key (required). |
| `KEMORY_URL` | The Kemory API URL. Overrides `base_url`. |
| `KEMORY_ORG_ID` | Pins the plugin to one Kemory organisation. Overrides `org_id`. |

The API URL is `KEMORY_URL` if set, else `base_url` from `kemory.json`, else
`https://api.kemory.s9n.ai`. It must start with `https://`; plain `http://` is accepted only for
`http://127.0.0.1` and `http://localhost`.

### kemory.json

Every other setting lives in `$HERMES_HOME/kemory.json`. All keys are optional.

| Key | Default | Range | Meaning |
| --- | --- | --- | --- |
| `base_url` | `https://api.kemory.s9n.ai` | https URL (http only for 127.0.0.1 and localhost) | The Kemory API URL, unless `KEMORY_URL` is set. |
| `org_id` | `none` | An organisation id | Pins the plugin to one organisation, unless `KEMORY_ORG_ID` is set. |
| `auto_recall` | `true` | `true` or `false` | Recall a profile summary and matching memories. `false` stops the search and profile requests; the tools still work. |
| `auto_capture` | `true` | `true` or `false` | Capture what you type. `false` turns capture off. |
| `gateways` | `false` | `true` or `false` | Also run on messaging gateways and other non-local surfaces. See Where it runs below. |
| `max_recall_results` | `6` | 1-20 | How many memories to recall per message. |
| `recall_timeout` | `3.0` | 0.5-7.5 (seconds) | Deadline for recall and for the search and context tools. |
| `write_timeout` | `20.0` | 1-60 (seconds) | Deadline for capture posts and for the store and forget tools. |
| `capture_min_new_prompts` | `3` | 1-12 | How many new prompts to collect before a capture is posted. |
| `context_max_chars` | `4000` | 500 or more | Size budget, in characters, of the profile summary. |

A value of the wrong type falls back to the default. A number outside its range is clamped to the
nearest end of it: `recall_timeout` set to `30` becomes `7.5`. A `kemory.json` that is missing or
is not a JSON object gives all defaults.

`hermes memory setup` asks for the API key (saved to Hermes's `.env`), `base_url` and `gateways`
(saved to `$HERMES_HOME/kemory.json`). Set every other key by editing `kemory.json`. Re-running
`hermes memory setup` offers the defaults for `base_url` and `gateways` again, not the values
saved in `kemory.json`, because Hermes pre-fills the questions from `config.yaml`. Check
`kemory.json` after a re-run if you use a custom URL or `gateways: true`.

### Organisations

A Kemory API key belongs to one organisation, and memories are read from and written to that
organisation. The plugin looks the organisation up once, with the key, and sends it on every
request.

If `KEMORY_ORG_ID` or `org_id` is set and differs from the key's organisation, the plugin stops
instead of writing to the other organisation: no recall and no writes, and each tool returns an
error naming both ids. To work in another organisation, use a key from that organisation.

### When something goes wrong

The plugin never raises into Hermes. What you see depends on the failure:

- **A bad or not-yet-approved key (HTTP 401)** turns the plugin off for the rest of that session.
  It logs one warning that names `KEMORY_API_KEY`, in Hermes's logs under `$HERMES_HOME/logs/`,
  and every `kemory_*` tool returns that message as its error.
- **A key from another region** turns the plugin off the same way. Its message names the hosts to
  put in `KEMORY_URL`.
- **A missing key** leaves the plugin off. It logs one warning, and the tools return an error
  naming `KEMORY_API_KEY`.
- **Timeouts, HTTP 429 and HTTP 5xx** skip that one call only. The first of each kind is logged as
  a warning; later ones are logged quietly. The next call is tried as usual.

## Where it runs

| Surface | Recall | Tools | Capture |
| --- | --- | --- | --- |
| CLI (`cli`), terminal UI (`tui`), desktop app (`desktop`), ACP editors (`acp`) and `local` | yes | yes | yes |
| Cron jobs | yes | yes, `kemory_store` and `kemory_forget` included | no |
| Messaging gateways, the API server, webhooks, and any other or unknown surface | no | no | no |

Gateways are off by default because one conversation there can mix people, while the memories
belong to the owner of the API key. Set `"gateways": true` in `kemory.json` to turn recall, tools
and capture on there. Cron jobs never capture, with or without `gateways`.

Subagents never reach memory providers, so they never use Kemory.

## What is captured and where it goes

### What is captured

Only the text you typed for each turn. Never assistant replies, tool output, system messages or the
conversation history.

- Slash commands, empty prompts and turns written by bots are skipped.
- Before anything else, Kemory's own recalled block, Hermes's `<memory-context>` block and
  Hermes's recall system note are removed from your text.
- That is all that is removed. Text that another plugin adds to an image turn can still be
  included in what is captured.
- Secrets are replaced with `[REDACTED]`. This is a pattern match on common credential shapes:
  bearer tokens, `key=value` secrets (`api_key`, `token`, `secret`, `password` and similar), `sk-`
  keys, GitHub, AWS and Slack tokens, JWTs and private key headers. It is pattern-based and will
  not catch every secret, so do not type secrets you cannot afford to have stored.

### When it is posted, and where it goes

A capture is posted after 3 new prompts (`capture_min_new_prompts`), and for anything still
pending before Hermes compresses the conversation, at session end, on a session switch and at
shutdown. Each capture holds at most the last 12 new prompts and 8000 characters. Prompts that do
not fit are not sent later: they count as captured and are never posted.

It is stored as one text memory, visible only to you, in the `user:sessions` namespace of the
API key's organisation. The memory carries `source: hermes`, the platform, and the Hermes session
id. Set `auto_capture` to `false` to turn capture off.

### Each prompt is sent at most once

A prompt is marked as captured only after Kemory accepts the post. A failed post is retried at the
next capture point.

The marks are SHA-256 hashes of captured prompts, with no prompt text, kept in
`$HERMES_HOME/kemory/captured/`, one file per session. The directory is `0700`, the files are
`0600`, and each file holds at most 1000 hashes. Prompts that are not yet posted live only in
memory, so they are lost if Hermes stops before the next capture point.

### What else goes to Kemory

- For each non-trivial prompt, its first 1000 characters as a search query. Greetings,
  acknowledgements and slash commands send nothing.
- A profile summary request at the start of a session. `auto_recall: false` stops this and the
  search requests.
- The workspace path (Hermes's working directory) in the header of each capture.
- Whatever the agent passes to a memory tool, such as a search query or a memory to store.
- On every request: your API key, the plugin version, the Hermes version, and your OS name and
  machine type (the `User-Agent` and `X-Kemory-Client` headers).

Assistant replies and tool output are never sent.

## Tools

Each tool returns JSON. An error comes back as `{"error": ...}` instead of breaking the turn.

- **`kemory_search(query, namespace?, limit?)`** searches your memories by meaning. `limit` is 1-20
  and defaults to `max_recall_results`. Returns `{"results": [{"memory_id", "namespace",
  "content"}]}`.
- **`kemory_store(content, namespace?)`** saves a memory. The content is redacted first. When no
  namespace is given, Kemory picks one by topic. Returns `{"stored": true, "memory_id",
  "namespace"}`.
- **`kemory_forget(memory_id)`** deletes one memory. Returns `{"forgotten": true, "memory_id"}`.
- **`kemory_context()`** shows your profile summaries, one line per namespace. Returns
  `{"context": ...}`.

## Development

The Hermes plugin is the `hermes_kemory/` directory. The rest of the repository is only for
development. The tests run against a checkout of Hermes in `.hermes/`, because Hermes cannot be
installed as a git dependency.

```sh
scripts/fetch-hermes.sh
uv sync --python 3.14
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -n 2
uv run hermes plugins validate hermes_kemory
```

`scripts/fetch-hermes.sh` checks out Hermes v0.22.3 by default. To test against the oldest
supported release, v0.22.0, pass its commit:

```sh
scripts/fetch-hermes.sh e06df9c2eae6901f4c88f464e0ae6e753c0428b8
```

Syncing the v0.22.0 checkout rewrites `uv.lock`. Do not commit that change.

## Catalog entry

The entry for Hermes's plugin catalog, `plugin-catalog/kemory.yaml`. Replace the `sha` with the
commit of the release before submitting it.

```yaml
name: kemory
repo: https://github.com/SeKondBrainAILabs/hermes-kemory
sha: "<40-hex release commit sha>"
subdir: hermes_kemory
version: "0.1.0"
description: Kemory long-term memory provider — profile recall, semantic search and explicit memory tools. Sends your prompts (secrets redacted, at most 12 per capture and 8000 characters), recall queries and the workspace path to the configured Kemory API; assistant replies and tool output are never sent.
maintainer: SeKondBrainAILabs
tier: community
category: memory
requires_hermes: ">=0.22.0"
docs_url: https://github.com/SeKondBrainAILabs/hermes-kemory#readme
capabilities:
  provides_tools: []
  provides_hooks: []
  provides_middleware: []
  requires_env:
    - KEMORY_API_KEY
```

"""The README and its catalog entry must agree with the code and with Hermes's catalog rules."""

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import hermes_yaml

from hermes_kemory import __version__
from hermes_kemory.config import CONFIG_KEYS, DEFAULT_BASE_URL, KemoryConfig

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
PLUGIN_MANIFEST = ROOT / "hermes_kemory" / "plugin.yaml"
CATALOG_VALIDATOR = ROOT / ".hermes" / "scripts" / "validate_plugin_catalog.py"

SECTIONS = [
    "## What it does",
    "## Install",
    "## Configuration",
    "## Where it runs",
    "## What is captured and where it goes",
    "## Tools",
    "## Development",
    "## Catalog entry",
]

SHA_PLACEHOLDER = "<40-hex release commit sha>"

DESCRIPTION = (
    "Kemory long-term memory provider — profile recall, semantic search and explicit memory "
    "tools. Sends your prompts (secrets redacted, at most 12 per capture and 8000 characters), "
    "recall queries and the workspace path to the configured Kemory API; assistant replies and "
    "tool output are never sent."
)

EXPECTED_ENTRY = {
    "name": "kemory",
    "repo": "https://github.com/SeKondBrainAILabs/hermes-kemory",
    "sha": SHA_PLACEHOLDER,
    "subdir": "hermes_kemory",
    "version": "0.1.0",
    "description": DESCRIPTION,
    "maintainer": "SeKondBrainAILabs",
    "tier": "community",
    "category": "memory",
    "requires_hermes": ">=0.22.0",
    "docs_url": "https://github.com/SeKondBrainAILabs/hermes-kemory#readme",
    "capabilities": {
        "provides_tools": [],
        "provides_hooks": [],
        "provides_middleware": [],
        "requires_env": ["KEMORY_API_KEY"],
    },
}


def readme_lines() -> list[str]:
    return README.read_text(encoding="utf-8").splitlines()


def section_lines(heading: str) -> list[str]:
    """The lines of one '## ' section, without its heading."""
    lines = readme_lines()
    start = lines.index(heading) + 1
    end = next((i for i in range(start, len(lines)) if lines[i].startswith("## ")), len(lines))
    return lines[start:end]


def catalog_entry_text() -> str:
    """The text of the first yaml fence after the '## Catalog entry' heading."""
    lines = section_lines("## Catalog entry")
    start = lines.index("```yaml") + 1
    end = lines.index("```", start)
    return "\n".join(lines[start:end]) + "\n"


def table_rows(lines: list[str]) -> list[list[str]]:
    """Every markdown table row as its stripped cells, separator rows left out."""
    rows = []
    for line in lines:
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(set(cell) <= set("-: ") for cell in cells):
            continue
        rows.append(cells)
    return rows


def expected_default(key: str) -> str:
    """The default of a kemory.json key as the README must print it, taken from the code."""
    if key == "base_url":
        return DEFAULT_BASE_URL
    if key == "org_id":
        return "none"  # the pin is optional: KemoryConfig.org_pin is None when it is not set
    field = next(f for f in dataclasses.fields(KemoryConfig) if f.name == key)
    return json.dumps(field.default)


def test_readme_has_the_sections_in_order() -> None:
    lines = readme_lines()

    assert lines[0].startswith("# ")
    assert [line for line in lines if line.startswith("## ")] == SECTIONS


def test_catalog_entry_parses_and_matches() -> None:
    entry = hermes_yaml.safe_load(catalog_entry_text())

    assert list(entry) == list(EXPECTED_ENTRY)
    assert dict(entry) == EXPECTED_ENTRY


def test_catalog_entry_passes_hermes_catalog_validator(tmp_path: Path) -> None:
    text = catalog_entry_text()
    assert SHA_PLACEHOLDER in text
    entry_file = tmp_path / "kemory.yaml"
    entry_file.write_text(text.replace(SHA_PLACEHOLDER, "0" * 40), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(CATALOG_VALIDATOR), str(entry_file)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "OK: 1 file(s) valid" in output
    assert "warning" not in output.lower()


def test_catalog_entry_matches_plugin_manifest() -> None:
    entry = hermes_yaml.safe_load(catalog_entry_text())
    manifest = hermes_yaml.safe_load(PLUGIN_MANIFEST.read_text(encoding="utf-8"))

    assert entry["name"] == manifest["name"]
    assert entry["version"] == manifest["version"] == __version__
    assert entry["capabilities"]["requires_env"] == manifest["requires_env"]


def test_config_table_matches_config_keys() -> None:
    documented: dict[str, str] = {}
    for cells in table_rows(section_lines("## Configuration")):
        key = cells[0].strip("`")
        if cells[0] == f"`{key}`" and key.islower() and key.replace("_", "").isalpha():
            assert key not in documented, f"{key} is listed twice"
            documented[key] = cells[1].replace("`", "")

    assert set(documented) == set(CONFIG_KEYS)
    for key in CONFIG_KEYS:
        assert documented[key] == expected_default(key), key


def test_env_vars_documented() -> None:
    rows = table_rows(section_lines("## Configuration"))

    for name in ("KEMORY_API_KEY", "KEMORY_URL", "KEMORY_ORG_ID"):
        matching = [cells for cells in rows if cells[0] == f"`{name}`"]
        assert len(matching) == 1, f"{name} needs exactly one table row in Configuration"
        if name == "KEMORY_API_KEY":
            assert "required" in " ".join(matching[0])

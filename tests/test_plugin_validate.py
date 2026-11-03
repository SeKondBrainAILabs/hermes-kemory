"""Hermes's own plugin validator accepts the plugin directory."""

import os
import subprocess
import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent.parent / "hermes_kemory"


def test_hermes_plugins_validate_passes(tmp_path: Path) -> None:
    # Only the plugin directory is validated: the repo root holds a full Hermes checkout.
    env = {**os.environ, "HERMES_HOME": str(tmp_path)}

    result = subprocess.run(
        [str(Path(sys.executable).parent / "hermes"), "plugins", "validate", str(PLUGIN_DIR)],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        stdin=subprocess.DEVNULL,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Validation passed." in result.stdout

#!/usr/bin/env bash
# Check out Hermes Agent at a pinned commit into .hermes/ at the repo root.
#
# Usage: scripts/fetch-hermes.sh [sha]
#
# The default is v0.22.3. Pass the v0.22.0 sha (e06df9c2eae6901f4c88f464e0ae6e753c0428b8)
# to test against the oldest supported release. An existing .hermes/ is replaced.
set -euo pipefail

DEFAULT_SHA="5b147ab036d6d19131bff8d20701b5001eab844a" # v0.22.3
REPO_URL="https://github.com/NousResearch/hermes-agent.git"

sha="${1:-$DEFAULT_SHA}"
if [[ ! "$sha" =~ ^[0-9a-f]{40}$ ]]; then
  echo "error: expected a full 40-character commit sha, got '$sha'" >&2
  exit 2
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="$root/.hermes"

rm -rf "$dest"
mkdir -p "$dest"
git -C "$dest" init --quiet
git -C "$dest" remote add origin "$REPO_URL"
git -C "$dest" fetch --quiet --depth 1 origin "$sha"
git -C "$dest" checkout --quiet --detach FETCH_HEAD

echo "Hermes Agent $(git -C "$dest" rev-parse HEAD) is checked out in $dest"

#!/bin/bash
# Restore the model weights and large data files that are stored as GitHub
# release assets instead of git files (they are over GitHub's file size limit).
#
#     ./fetch_artifacts.sh              restore everything (about 12 GB on disk)
#     ./fetch_artifacts.sh phase3_lora  restore only paths containing this text
#
# Needs the GitHub CLI (gh), logged in to an account that can read the repo.
# Files that already exist are skipped. Each restored file is checked against
# the SHA-256 in artifacts/MANIFEST.tsv.

set -euo pipefail
cd "$(dirname "$0")"
REPO="ponyta3/Nulliglot"
TAG="artifacts-2026-11-23"
want="${1:-}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

tail -n +2 artifacts/MANIFEST.tsv | while IFS=$'\t' read -r path asset bytes sha; do
  if [[ -n "$want" && "$path" != *"$want"* ]]; then continue; fi
  if [[ -f "$path" ]]; then echo "have   $path"; continue; fi
  if [[ ! -f "$tmp/$asset" ]]; then
    gh release download "$TAG" -R "$REPO" -p "$asset" -D "$tmp" </dev/null
  fi
  mkdir -p "$(dirname "$path")"
  if [[ "$asset" == *.gz ]]; then gunzip -c "$tmp/$asset" > "$path"; else cp "$tmp/$asset" "$path"; fi
  echo "$sha  $path" | shasum -a 256 -c - >/dev/null || { echo "CHECKSUM FAILED: $path" >&2; exit 1; }
  echo "wrote  $path"
done

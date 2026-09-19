#!/usr/bin/env bash
# Tagging starts the container build. Let the robots earn their co-author line.
set -euo pipefail
cd "$(dirname "$0")"
if [[ $# != 2 || ! "$1" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo 'Usage: ./make_release.sh vX.Y.Z "release message"' >&2
  exit 1
fi
if [[ -n "$(git status --porcelain)" || "$(git branch --show-current)" != main ]]; then
  echo 'Release from a clean main checkout.' >&2
  exit 1
fi
git tag -a "$1" -m "$2"
git push origin "$1"

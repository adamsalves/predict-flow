#!/usr/bin/env bash
# Guard checks ensuring local ML artifacts and datasets can never be committed
# to git nor baked into the runtime image (ADR-0017).
#
# Two layers are verified:
#   1. .gitignore covers a representative set of artifact/dataset paths
#      (git check-ignore works on paths that do not exist yet);
#   2. backend/.dockerignore excludes the artifact base directories and
#      binary artifact extensions from the Docker build context.
#
# Exits 0 when both layers hold, 1 listing the gaps otherwise.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Representative artifact/dataset paths that must stay out of version control.
GITIGNORED_PATHS=(
  "models/v1/model.joblib"
  "backend/models/v1/model.joblib"
  "backend/models/v1/model_meta.json"
  "data/raw/train.csv"
  "datasets/telco.csv"
  "backend/artifacts/run/model.joblib"
)

not_ignored=()
for path in "${GITIGNORED_PATHS[@]}"; do
  if ! git check-ignore -q "$path"; then
    not_ignored+=("$path")
  fi
done

# Base directories and binary extensions that must stay out of the image.
DOCKERIGNORE="backend/.dockerignore"
DOCKERIGNORE_ENTRIES=(
  "models/"
  "data/"
  "datasets/"
  "*.joblib"
  "*.parquet"
)

missing_entries=()
for entry in "${DOCKERIGNORE_ENTRIES[@]}"; do
  if ! grep -qxF "$entry" "$DOCKERIGNORE"; then
    missing_entries+=("$entry")
  fi
done

if ((${#not_ignored[@]} > 0 || ${#missing_entries[@]} > 0)); then
  if ((${#not_ignored[@]} > 0)); then
    printf 'Not ignored by .gitignore:\n' >&2
    printf '  %s\n' "${not_ignored[@]}" >&2
  fi
  if ((${#missing_entries[@]} > 0)); then
    printf 'Missing from %s:\n' "$DOCKERIGNORE" >&2
    printf '  %s\n' "${missing_entries[@]}" >&2
  fi
  exit 1
fi

echo "Artifact protection OK: git and Docker ignore all artifact/dataset paths."

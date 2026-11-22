#!/usr/bin/env python3
"""Verify version parity of shared packages between the serving runtime lock
(backend/requirements.txt) and the training lock (backend/requirements-train.txt).

The training environment reuses the runtime lock for the shared pipeline
libraries (scikit-learn, joblib): training serializes model.joblib and serving
loads it via joblib.load, so any version drift between the two environments
breaks the contract. The locks are compiled from requirements.in /
requirements-train.in with pip-compile (ADR-0017); this check fails the build
if a package present in both locks resolves to different versions.

Exits 0 when the shared packages are consistent, 1 otherwise.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVING_LOCK = REPO_ROOT / "backend" / "requirements.txt"
TRAINING_LOCK = REPO_ROOT / "backend" / "requirements-train.txt"

# Matches a pinned requirement line such as "psycopg[binary]==3.3.9".
# Extras and environment markers are tolerated; name and version are captured.
PIN_RE = re.compile(r"^([A-Za-z0-9._-]+)(?:\[[^\]]*\])?\s*==\s*([^;\s]+)")


def read_lock(path: Path) -> dict[str, str]:
    """Return a mapping of normalized package name to pinned version."""
    if not path.is_file():
        raise FileNotFoundError(f"lock file not found: {path}")
    pins: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = PIN_RE.match(stripped)
        if match is None:
            continue
        name = match.group(1).lower()
        pins[name] = match.group(2)
    return pins


def main() -> int:
    serving = read_lock(SERVING_LOCK)
    training = read_lock(TRAINING_LOCK)

    shared = sorted(set(serving) & set(training))
    mismatched = [
        (name, serving[name], training[name])
        for name in shared
        if serving[name] != training[name]
    ]

    if mismatched:
        print(
            "Dependency parity check FAILED: version mismatch between "
            "backend/requirements.txt and backend/requirements-train.txt:"
        )
        for name, serving_version, training_version in mismatched:
            print(f"  {name}: serving=={serving_version} training=={training_version}")
        return 1

    print(
        f"Dependency parity OK: {len(shared)} shared packages checked between "
        "serving (backend/requirements.txt) and training "
        "(backend/requirements-train.txt)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Compatibility wrapper for paper-grade report figures.

Legacy docs still reference this entrypoint. It now delegates to
scripts/generate_paper_artifacts.py, which rebuilds the paper figure
whitelist from real logs and archives non-paper images.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "generate_paper_artifacts.py"),
        "--archive-legacy",
    ]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
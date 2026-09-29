#!/usr/bin/env python3
"""Freeze the RF-1 baseline state into an immutable artifact directory.

This script captures everything needed to verify that future upgrades do not
break the verified RF-1.x behavior:
  - git commit hash
  - python version
  - dependency lock (pip freeze)
  - test summary (pytest --co -q count)
  - environment fingerprint (sha256 of sorted pip freeze)
  - RESEARCHFORGE_STATE.yaml hash

Output: artifacts/baseline/ with immutable JSON + text artifacts.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIR = ROOT / "artifacts" / "baseline"


def _run(cmd: list[str], cwd: str | None = None) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd or str(ROOT))
    return result.stdout.strip()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Git commit
    commit = _run(["git", "rev-parse", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    (BASELINE_DIR / "baseline_commit.txt").write_text(
        f"commit: {commit}\nbranch: {branch}\ntimestamp: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n"
    )

    # 2. Python version
    (BASELINE_DIR / "python_version.txt").write_text(
        f"python: {sys.version}\nplatform: {platform.platform()}\n"
    )

    # 3. Dependency lock
    pip_freeze = _run([sys.executable, "-m", "pip", "freeze"])
    (BASELINE_DIR / "dependency_lock.txt").write_text(pip_freeze + "\n")

    # 4. Environment fingerprint
    env_hash = _sha256("\n".join(sorted(pip_freeze.splitlines())))
    (BASELINE_DIR / "environment_fingerprint.txt").write_text(f"sha256: {env_hash}\n")

    # 5. State file hash
    state_path = ROOT / "RESEARCHFORGE_STATE.yaml"
    if state_path.exists():
        state_hash = _sha256(state_path.read_text())
    else:
        state_hash = "MISSING"
    
    # 6. Test count (collected, not run)
    test_count_output = _run(
        [sys.executable, "-m", "pytest", "tests/test_basic.py", "tests/test_regression.py",
         "--co", "-q"], cwd=str(ROOT)
    )
    test_lines = [l for l in test_count_output.splitlines() if "::" in l]
    
    # 7. Build summary
    summary = {
        "freeze_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": commit,
        "git_branch": branch,
        "python_version": sys.version,
        "platform": platform.platform(),
        "environment_fingerprint": env_hash,
        "state_yaml_fingerprint": state_hash,
        "baseline_test_count": len(test_lines),
        "baseline_tests": test_lines,
        "researchforge_version": "RF-1.0.0-alpha.2.1",
        "freeze_purpose": (
            "Immutable baseline snapshot before RF-2 research-policy upgrade. "
            "All future regression tests compare against this state."
        ),
    }
    (BASELINE_DIR / "baseline_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print(f"Baseline frozen to {BASELINE_DIR}")
    print(f"  commit:    {commit[:12]}")
    print(f"  tests:     {len(test_lines)}")
    print(f"  env hash:  {env_hash[:16]}...")
    print(f"  state hash: {state_hash[:16]}...")


if __name__ == "__main__":
    main()

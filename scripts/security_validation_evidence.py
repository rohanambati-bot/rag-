"""Create-only workspace/check evidence without copying source, secrets or raw diffs."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ANCHORS = [
    "outputs/eval_results.csv", "outputs/significance_tests.csv", "outputs/evaluation_report.md",
    "outputs/live-lifecycle-20261019-v3.json",
    "outputs/retrieval-ablation-local-20261019-v2/per_query.jsonl",
    "outputs/retrieval-supplement-20261019-v3/summary.json",
    "outputs/answer-boundaries-v1-20261019/summary.json",
    "docs/VALIDATION_2026-10-19_SECURITY_V3.md",
    "requirements.txt", "requirements.lock",
]


def capture(output: Path, phase: str):
    if output.exists():
        raise FileExistsError("Refusing to replace evidence")
    commands = [["git", "status", "--short"], ["git", "branch", "--show-current"],
                ["git", "log", "--oneline", "-5"], ["git", "diff", "--stat"],
                ["git", "diff", "--cached", "--stat"]]
    receipt = {"schema": "security-validation-state/v1", "phase": phase,
               "utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
               "platform": platform.platform(), "git": {}, "anchor_sha256": {}}
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
        receipt["git"][" ".join(command)] = result.stdout
    for name in ANCHORS:
        path = ROOT / name
        receipt["anchor_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"phase": phase, "output": str(output), "anchors": len(ANCHORS)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    args = parser.parse_args()
    capture(args.output, args.phase)

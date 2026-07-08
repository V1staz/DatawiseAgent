"""Gate qwen3-vl follow-up runs on 8B DataModeling performance.

The intended policy is: run qwen3-vl-8b-thinking first; only launch 32B if the
8B result reaches the chosen paper/baseline normalized-performance threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether a qwen3-vl 8B DataModeling run justifies launching 32B.")
    parser.add_argument("--compare-json", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--threshold", type=float, default=0.4290, help="Normalized performance threshold. Default is the recorded qwen2.5-72B baseline.")
    parser.add_argument("--min-quality-passed", type=int, default=0, help="Optional minimum quality-gate-passed task count.")
    parser.add_argument("--min-score-ok", type=int, default=70, help="Minimum official-score-ok tasks for a usable full run.")
    args = parser.parse_args()

    payload = json.loads(Path(args.compare_json).read_text(encoding="utf-8"))
    summary = (payload.get("summary") or {}).get(args.model)
    if not summary:
        print(json.dumps({"passed": False, "reason": "model_missing", "model": args.model}, ensure_ascii=False, indent=2))
        sys.exit(2)

    performance = float(summary.get("performance") or 0.0)
    quality_passed = int(summary.get("quality_passed") or 0)
    score_ok = int(summary.get("score_ok") or 0)
    checks = {
        "performance_reaches_threshold": performance >= args.threshold,
        "quality_passed_reaches_minimum": quality_passed >= args.min_quality_passed,
        "score_ok_reaches_minimum": score_ok >= args.min_score_ok,
    }
    passed = all(checks.values())
    result = {
        "passed": passed,
        "model": args.model,
        "threshold": args.threshold,
        "performance": performance,
        "quality_passed": quality_passed,
        "score_ok": score_ok,
        "checks": checks,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(0 if passed else 2)


if __name__ == "__main__":
    main()

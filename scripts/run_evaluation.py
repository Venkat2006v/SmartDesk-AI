#!/usr/bin/env python
"""Run the SmartDesk AI evaluation suite.

Prerequisites:
  1. Knowledge base must be built:  python scripts/build_knowledge_base.py
  2. .env must be configured with LLM_API_KEY and EMBEDDING_PROVIDER

Usage:
    # Full suite with LLM judges (faithfulness + relevance) — uses API credits
    python scripts/run_evaluation.py

    # Fast structural check — routing + escalation only, zero LLM calls
    python scripts/run_evaluation.py --skip-llm-judges

    # Six-case smoke test (fast)
    python scripts/run_evaluation.py --suite minimal --skip-llm-judges

    # Save results to JSON
    python scripts/run_evaluation.py --output eval_results.json

    # Verbose — prints per-case progress during the run
    python scripts/run_evaluation.py --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure src/ is importable when running directly from the scripts/ directory
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SmartDesk AI evaluation suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--suite",
        choices=["full", "minimal"],
        default="full",
        help="Test suite to run: 'full' (20 cases) or 'minimal' (6 smoke-test cases)",
    )
    parser.add_argument(
        "--skip-llm-judges",
        action="store_true",
        help="Skip faithfulness + relevance LLM calls — routing & escalation only",
    )
    parser.add_argument(
        "--output", "-o",
        metavar="FILE",
        help="Save JSON report to FILE",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-case progress during the run",
    )
    args = parser.parse_args()

    # ── Import pipeline ───────────────────────────────────────────────────────
    try:
        from smartdesk.evaluation.eval_pipeline import (
            DEFAULT_TEST_SUITE,
            MINIMAL_TEST_SUITE,
            print_report,
            run_evaluation,
        )
    except ImportError as exc:
        print(f"[error] Import failed: {exc}")
        print("Make sure you ran:  pip install -e .")
        sys.exit(1)

    suite = MINIMAL_TEST_SUITE if args.suite == "minimal" else DEFAULT_TEST_SUITE

    print(f"[eval] Running '{args.suite}' suite — {len(suite)} cases")
    if args.skip_llm_judges:
        print("[eval] LLM judges skipped (routing + escalation metrics only)")
    print()

    # ── Run ───────────────────────────────────────────────────────────────────
    try:
        report = run_evaluation(
            test_cases=suite,
            skip_llm_judges=args.skip_llm_judges,
            verbose=args.verbose,
        )
    except Exception as exc:
        print(f"[error] Evaluation failed: {exc}")
        print("Check that the knowledge base is built and .env is configured.")
        sys.exit(1)

    # ── Print ─────────────────────────────────────────────────────────────────
    print_report(report)

    # ── Save ─────────────────────────────────────────────────────────────────
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved → {output_path.resolve()}")


if __name__ == "__main__":
    main()

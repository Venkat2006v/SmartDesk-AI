#!/usr/bin/env python
"""Run SmartDesk AI evaluation with LangSmith tracing and dataset logging.

Prerequisites
-------------
1. Knowledge base built:  python scripts/build_knowledge_base.py
2. .env configured with:
       LANGCHAIN_API_KEY=<ls__... key from smith.langchain.com>
       LANGCHAIN_TRACING_V2=true
       LANGCHAIN_PROJECT=smartdesk-ai
       LLM_API_KEY=<your OpenAI key>

Usage
-----
    # Full suite — pushes dataset + runs routing/escalation evaluators
    python scripts/run_langsmith_eval.py

    # Minimal 6-case smoke test
    python scripts/run_langsmith_eval.py --suite minimal

    # Skip LLM-judged faithfulness/relevance (faster, cheaper)
    python scripts/run_langsmith_eval.py --skip-llm-judges

    # Custom experiment name visible in LangSmith UI
    python scripts/run_langsmith_eval.py --experiment-name "post-kb-rebuild"

What this does
--------------
1. Creates (or reuses) a LangSmith dataset named "SmartDesk-AI Eval Suite".
2. Pushes each test case as a LangSmith Example (query → expected outputs).
3. Runs `langsmith.evaluate()` which calls the SmartDesk agent on every example
   and records the full LangGraph trace to your LangSmith project.
4. Three built-in evaluators score each run:
     - routing_correct      : 1 if supervisor chose the right agent, else 0
     - escalation_correct   : 1 if escalation decision matches expected, else 0
     - confidence_score     : raw top-chunk cosine similarity (0–1)
5. Optional LLM judges (faithfulness + answer relevance) run on KB cases.
6. Prints a summary table; the full experiment link is printed at the end.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure src/ is importable when running from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def _check_env() -> None:
    """Abort early with a helpful message if LangSmith is not configured."""
    api_key = os.environ.get("LANGCHAIN_API_KEY", "")
    if not api_key:
        print(
            "[error] LANGCHAIN_API_KEY is not set.\n"
            "  1. Sign up free at https://smith.langchain.com\n"
            "  2. Settings → API Keys → Create API Key\n"
            "  3. Add to your .env:\n"
            "       LANGCHAIN_API_KEY=<your key>\n"
            "       LANGCHAIN_TRACING_V2=true\n"
            "       LANGCHAIN_PROJECT=smartdesk-ai"
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Target function (called by langsmith.evaluate for each example)
# ---------------------------------------------------------------------------

def _build_target(graph, routing_only_ids: set):
    """Return a closure that runs the SmartDesk agent and returns structured output."""
    from smartdesk.agents.supervisor import supervisor_node
    from smartdesk.orchestrator.graph import run_once

    def target(inputs: Dict[str, Any]) -> Dict[str, Any]:
        query = inputs["query"]
        is_routing_only = inputs.get("routing_only", False)

        if is_routing_only:
            sup_state = supervisor_node({"query": query})
            return {
                "route": sup_state.get("route", "off_topic"),
                "should_escalate": False,
                "confidence_score": 0.0,
                "response": "[routing-only]",
                "retrieved_chunks": [],
            }

        result = run_once(graph, {"query": query})
        return {
            "route": result.get("route", "off_topic"),
            "should_escalate": result.get("should_escalate", False),
            "confidence_score": float(result.get("confidence_score", 0.0)),
            "response": result.get("response", ""),
            "retrieved_chunks": result.get("retrieved_chunks", []),
        }

    return target


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

def routing_correct(run, example) -> Dict[str, Any]:
    """Score 1 if the actual route matches the expected route, else 0."""
    expected = example.outputs.get("expected_route", "")
    actual = (run.outputs or {}).get("route", "off_topic")
    return {
        "key": "routing_correct",
        "score": int(actual == expected),
        "comment": f"expected={expected}, got={actual}",
    }


def escalation_correct(run, example) -> Dict[str, Any]:
    """Score 1 if escalation decision matches expected (skip routing-only cases)."""
    if example.outputs.get("routing_only"):
        return {"key": "escalation_correct", "score": 1, "comment": "routing-only (skipped)"}

    expected = bool(example.outputs.get("should_escalate", False))
    actual = bool((run.outputs or {}).get("should_escalate", False))
    return {
        "key": "escalation_correct",
        "score": int(actual == expected),
        "comment": f"expected={expected}, got={actual}",
    }


def confidence_score_evaluator(run, example) -> Dict[str, Any]:
    """Log the raw confidence score as a numeric metric."""
    score = float((run.outputs or {}).get("confidence_score", 0.0))
    return {"key": "confidence_score", "score": score}


def faithfulness_judge(run, example) -> Dict[str, Any]:
    """LLM-judged faithfulness (0–1). Only runs on KB cases with a real answer."""
    from smartdesk.evaluation.eval_pipeline import _judge_faithfulness

    outputs = run.outputs or {}
    route = outputs.get("route", "")
    routing_only = example.outputs.get("routing_only", False)
    if routing_only or route not in ("it_kb", "hr_kb", "combined_kb"):
        return {"key": "faithfulness", "score": None, "comment": "skipped"}

    query = example.inputs.get("query", "")
    answer = outputs.get("response", "")
    chunks = outputs.get("retrieved_chunks", [])
    score = _judge_faithfulness(query, answer, chunks)
    return {"key": "faithfulness", "score": score}


def relevance_judge(run, example) -> Dict[str, Any]:
    """LLM-judged answer relevance (0–1). Only runs on KB cases."""
    from smartdesk.evaluation.eval_pipeline import _judge_relevance

    outputs = run.outputs or {}
    route = outputs.get("route", "")
    routing_only = example.outputs.get("routing_only", False)
    if routing_only or route not in ("it_kb", "hr_kb", "combined_kb"):
        return {"key": "relevance", "score": None, "comment": "skipped"}

    query = example.inputs.get("query", "")
    answer = outputs.get("response", "")
    score = _judge_relevance(query, answer)
    return {"key": "relevance", "score": score}


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _upsert_dataset(client, suite, dataset_name: str):
    """Create or reuse the LangSmith dataset; upsert examples from the suite."""
    # Get or create dataset
    datasets = list(client.list_datasets(dataset_name=dataset_name))
    if datasets:
        dataset = datasets[0]
        print(f"[langsmith] Reusing existing dataset: {dataset.name!r} (id={dataset.id})")
    else:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="SmartDesk AI evaluation suite — routing, escalation, RAG quality",
        )
        print(f"[langsmith] Created dataset: {dataset.name!r} (id={dataset.id})")

    # Upsert examples (create_examples is idempotent when source_run_id is not set;
    # simplest approach: delete + recreate on every run so cases stay fresh)
    existing = list(client.list_examples(dataset_id=dataset.id))
    if existing:
        print(f"[langsmith] Deleting {len(existing)} stale examples...")
        for ex in existing:
            client.delete_example(ex.id)

    examples = [
        {
            "inputs": {
                "query": case["query"],
                "routing_only": case.get("routing_only", False),
            },
            "outputs": {
                "expected_route": case["expected_route"],
                "should_escalate": case["should_escalate"],
                "routing_only": case.get("routing_only", False),
                "category": case["category"],
                "notes": case.get("notes", ""),
            },
        }
        for case in suite
    ]

    client.create_examples(inputs=[e["inputs"] for e in examples],
                           outputs=[e["outputs"] for e in examples],
                           dataset_id=dataset.id)
    print(f"[langsmith] Pushed {len(examples)} examples to dataset.")
    return dataset


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(results) -> None:
    """Print a concise local summary from the LangSmith evaluate() results."""
    sep = "─" * 62
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║      SmartDesk AI — LangSmith Evaluation Summary        ║")
    print("╚══════════════════════════════════════════════════════════╝")

    rows = list(results)
    if not rows:
        print("  No results returned.")
        return

    routing_scores    = []
    escalation_scores = []
    confidence_scores = []
    faith_scores      = []
    rel_scores        = []

    for row in rows:
        # LangSmith SDK: results are dicts with "evaluation_results" key
        # evaluation_results is an object with a "results" list of EvaluationResult
        eval_results = row.get("evaluation_results") or {}
        feedback_list = []

        if hasattr(eval_results, "results"):
            feedback_list = eval_results.results or []
        elif isinstance(eval_results, dict):
            feedback_list = eval_results.get("results") or []

        for fb in feedback_list:
            key   = getattr(fb, "key",   None) or (fb.get("key")   if isinstance(fb, dict) else None)
            score = getattr(fb, "score", None)
            if score is None and isinstance(fb, dict):
                score = fb.get("score")

            if score is None:
                continue
            if key == "routing_correct":    routing_scores.append(score)
            if key == "escalation_correct": escalation_scores.append(score)
            if key == "confidence_score":   confidence_scores.append(score)
            if key == "faithfulness":       faith_scores.append(score)
            if key == "relevance":          rel_scores.append(score)

    def avg(lst): return sum(lst) / len(lst) if lst else None
    def pct(v):   return f"{v:.1%}" if v is not None else "n/a"
    def fmt(v):   return f"{v:.2f}" if v is not None else "n/a"
    def frac(lst): return f"{sum(lst):.0f}/{len(lst)}" if lst else "0/0"

    print(f"\n  Total cases        : {len(rows)}")
    print(sep)
    print(f"  Routing accuracy   : {pct(avg(routing_scores))}  ({frac(routing_scores)} correct)")
    print(f"  Escalation accuracy: {pct(avg(escalation_scores))}  ({frac(escalation_scores)} correct)")
    print(f"  Avg confidence     : {fmt(avg(confidence_scores))}")
    if faith_scores:
        print(f"  Avg faithfulness   : {fmt(avg(faith_scores))}")
    if rel_scores:
        print(f"  Avg relevance      : {fmt(avg(rel_scores))}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SmartDesk AI evaluation via LangSmith",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--suite", choices=["full", "minimal"], default="full")
    parser.add_argument("--skip-llm-judges", action="store_true",
                        help="Skip faithfulness + relevance LLM calls")
    parser.add_argument("--experiment-name", default=None,
                        help="Prefix for the experiment name in LangSmith UI")
    parser.add_argument("--dataset-name", default="SmartDesk-AI Eval Suite",
                        help="LangSmith dataset name")
    args = parser.parse_args()

    # Load .env first so LANGCHAIN_* vars are set
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env", override=True)

    _check_env()

    # Ensure tracing is on
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

    try:
        from langsmith import Client, evaluate
    except ImportError:
        print("[error] langsmith not installed. Run:  pip install langsmith")
        sys.exit(1)

    from smartdesk.evaluation.eval_pipeline import DEFAULT_TEST_SUITE, MINIMAL_TEST_SUITE

    suite = MINIMAL_TEST_SUITE if args.suite == "minimal" else DEFAULT_TEST_SUITE
    print(f"[eval] Suite: '{args.suite}' — {len(suite)} cases")
    if args.skip_llm_judges:
        print("[eval] LLM judges skipped")
    print()

    # Build graph
    from smartdesk.orchestrator.graph import build_orchestrator
    print("[eval] Building agent graph...")
    graph = build_orchestrator()
    print("[eval] Graph ready.\n")

    # LangSmith client
    client = Client()

    # Upsert dataset
    _upsert_dataset(client, suite, args.dataset_name)

    # Evaluators
    evaluators = [routing_correct, escalation_correct, confidence_score_evaluator]
    if not args.skip_llm_judges:
        evaluators += [faithfulness_judge, relevance_judge]

    # Experiment name
    experiment_prefix = args.experiment_name or f"smartdesk-{args.suite}"

    print(f"\n[eval] Starting LangSmith experiment: {experiment_prefix!r}")
    print("[eval] Traces will appear at https://smith.langchain.com\n")

    target = _build_target(graph, routing_only_ids=set())

    results = evaluate(
        target,
        data=args.dataset_name,
        evaluators=evaluators,
        experiment_prefix=experiment_prefix,
        client=client,
        max_concurrency=1,   # sequential — avoids Qdrant lock contention
    )

    _print_summary(results)

    project = os.environ.get("LANGCHAIN_PROJECT", "smartdesk-ai")
    print(f"Full traces → https://smith.langchain.com/projects/{project}")
    print()


if __name__ == "__main__":
    main()

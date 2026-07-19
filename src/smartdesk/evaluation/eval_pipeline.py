"""Evaluation harness for SmartDesk AI.

Three metric categories
-----------------------
1. Routing accuracy    — did the supervisor send each query to the right agent?
2. Escalation accuracy — precision / recall / F1 on escalation decisions
3. RAG quality         — faithfulness + answer relevance (LLM-judged, optional)

Design decisions
----------------
- No extra dependencies (no Ragas / DeepEval required)
- Uses the existing call_llm() adapter for LLM-judged metrics — same provider
  and model configured in .env
- LLM judges are OPTIONAL: pass skip_llm_judges=True (or --skip-llm-judges on
  the CLI) for a fast structural smoke test that costs zero API calls
- create_ticket / ticket_status cases run routing-only through the supervisor
  to avoid HITL blocking and email-required flows during automated eval
- off_topic cases run the full graph (no external calls needed)

Usage
-----
    python scripts/run_evaluation.py               # full suite with LLM judges
    python scripts/run_evaluation.py --skip-llm-judges   # routing + escalation only
    python scripts/run_evaluation.py --suite minimal      # 6 smoke-test cases
    python scripts/run_evaluation.py --output results.json
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from typing import NotRequired, TypedDict
except ImportError:
    from typing_extensions import NotRequired, TypedDict  # type: ignore


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class EvalCase(TypedDict):
    """Single evaluation test case."""
    query: str
    expected_route: str          # one of the Route literals
    should_escalate: bool        # True = KB cannot answer confidently
    category: str                # "it_covered", "hr_escalation", "combined", etc.
    routing_only: NotRequired[bool]  # True = only check supervisor; skip downstream
    notes: NotRequired[str]


class EvalResult(TypedDict):
    """Result for a single test case."""
    query: str
    category: str
    expected_route: str
    actual_route: str
    route_correct: bool
    expected_escalate: bool
    actual_escalate: bool
    escalation_correct: bool
    confidence_score: float
    response_preview: str        # first 120 chars
    faithfulness_score: Optional[float]   # 0–1, None if skipped
    relevance_score: Optional[float]      # 0–1, None if skipped
    latency_seconds: float
    routing_only: bool
    error: Optional[str]


class EvalReport(TypedDict):
    """Aggregated metrics across all test cases."""
    timestamp: str
    total_cases: int
    routing_accuracy: float
    routing_correct: int
    escalation_precision: Optional[float]
    escalation_recall: Optional[float]
    escalation_f1: Optional[float]
    avg_faithfulness: Optional[float]
    avg_relevance: Optional[float]
    avg_confidence_answered: Optional[float]
    avg_confidence_escalated: Optional[float]
    avg_latency_seconds: float
    results: List[EvalResult]


# ---------------------------------------------------------------------------
# Test suites
# ---------------------------------------------------------------------------

DEFAULT_TEST_SUITE: List[EvalCase] = [
    # ── IT covered topics (should answer, not escalate) ──────────────────────
    {
        "query": "How do I connect to the VPN from my laptop?",
        "expected_route": "it_kb",
        "should_escalate": False,
        "category": "it_covered",
        "notes": "Core IT topic — VPN setup is in the knowledge base",
    },
    {
        "query": "Walk me through setting up MFA and TOTP on my phone.",
        "expected_route": "it_kb",
        "should_escalate": False,
        "category": "it_covered",
        "notes": "Tests acronym handling (MFA, TOTP) + step-by-step retrieval",
    },
    {
        "query": "How do I reset my SSO password?",
        "expected_route": "it_kb",
        "should_escalate": False,
        "category": "it_covered",
        "notes": "SSO / password reset — covered IT topic",
    },
    {
        "query": "My laptop won't connect to the office Wi-Fi. What should I do?",
        "expected_route": "it_kb",
        "should_escalate": False,
        "category": "it_covered",
        "notes": "Wi-Fi troubleshooting — covered IT topic",
    },
    {
        "query": "SSO login keeps failing — what should I do?",
        "expected_route": "it_kb",
        "should_escalate": False,
        "category": "it_covered",
        "notes": "SSO login failure troubleshooting — covered by SSO failures doc",
    },
    {
        "query": "LDAP directory sync errors — how do I fix them?",
        "expected_route": "it_kb",
        "should_escalate": False,
        "category": "it_covered",
        "notes": "LDAP sync errors — covered by LDAP troubleshooting doc",
    },
    {
        "query": "My MFA TOTP code is rejected at the SSO portal",
        "expected_route": "it_kb",
        "should_escalate": False,
        "category": "it_covered",
        "notes": "TOTP rejection troubleshooting — covered by MFA/TOTP docs",
    },
    {
        "query": "How do I request access to a new software tool?",
        "expected_route": "it_kb",
        "should_escalate": False,
        "category": "it_covered",
        "notes": "Software access request — covered by software access request doc",
    },
    # ── IT escalation (should escalate — topic not in KB) ────────────────────
    {
        "query": "How do I provision EC2 instances for my team?",
        "expected_route": "it_kb",
        "should_escalate": True,
        "category": "it_escalation",
        "notes": "Intentionally uncovered — AWS/EC2 not in IT docs",
    },
    {
        "query": "Walk me through setting up a Kubernetes cluster.",
        "expected_route": "it_kb",
        "should_escalate": True,
        "category": "it_escalation",
        "notes": "Intentionally uncovered — Kubernetes not in IT docs",
    },
    {
        "query": "How do I process a GDPR data deletion request?",
        "expected_route": "it_kb",
        "should_escalate": True,
        "category": "it_escalation",
        "notes": "Intentionally uncovered — GDPR compliance not in IT docs",
    },
    # ── HR covered topics (should answer, not escalate) ──────────────────────
    {
        "query": "How many PTO days do employees get per year?",
        "expected_route": "hr_kb",
        "should_escalate": False,
        "category": "hr_covered",
        "notes": "PTO policy — in HuggingFace HR dataset",
    },
    {
        "query": "When is the benefits open enrollment window?",
        "expected_route": "hr_kb",
        "should_escalate": False,
        "category": "hr_covered",
        "notes": "Benefits enrollment — in HR knowledge base",
    },
    {
        "query": "How do I submit an expense reimbursement?",
        "expected_route": "hr_kb",
        "should_escalate": False,
        "category": "hr_covered",
        "notes": "Expense policy — synthetic HR doc",
    },
    {
        "query": "What is the company remote work policy?",
        "expected_route": "hr_kb",
        "should_escalate": False,
        "category": "hr_covered",
        "notes": "Remote work policy — synthetic HR doc",
    },
    # ── HR escalation (should escalate — topic not in KB) ────────────────────
    {
        "query": "What are the details of the executive equity and vesting plan?",
        "expected_route": "hr_kb",
        "should_escalate": True,
        "category": "hr_escalation",
        "notes": "Intentionally uncovered — executive equity not in HR docs",
    },
    {
        "query": "What HIPAA training is required for my role?",
        "expected_route": "hr_kb",
        "should_escalate": True,
        "category": "hr_escalation",
        "notes": "Intentionally uncovered — HIPAA training not in HR docs",
    },
    # ── Combined IT + HR (synthesizer route) ─────────────────────────────────
    {
        "query": (
            "I am a new hire — what IT equipment setup steps do I need to complete "
            "and what HR benefits enrollment deadlines should I know about?"
        ),
        "expected_route": "combined_kb",
        "should_escalate": False,
        "category": "combined",
        "notes": "Explicit dual-domain query — should trigger combined_knowledge_node",
    },
    {
        "query": (
            "What are the laptop setup steps from IT and the PTO policy from HR "
            "for employees joining next month?"
        ),
        "expected_route": "combined_kb",
        "should_escalate": False,
        "category": "combined",
        "notes": "Second combined-domain query to test routing consistency",
    },
    # ── Ticket operations (routing-only — HITL / email required downstream) ──
    {
        "query": "Create a ticket — my VPN keeps disconnecting every 30 minutes.",
        "expected_route": "create_ticket",
        "should_escalate": False,
        "category": "ticket_create",
        "routing_only": True,
        "notes": "Explicit ticket creation intent — routing-only to avoid HITL block",
    },
    {
        "query": "Open a helpdesk request: I can't install the required software.",
        "expected_route": "create_ticket",
        "should_escalate": False,
        "category": "ticket_create",
        "routing_only": True,
        "notes": "Synonym for ticket creation ('open a helpdesk request')",
    },
    {
        "query": "What open tickets do I have right now?",
        "expected_route": "ticket_status",
        "should_escalate": False,
        "category": "ticket_status",
        "routing_only": True,
        "notes": "Ticket status intent — routing-only (needs email downstream)",
    },
    # ── Off-topic ─────────────────────────────────────────────────────────────
    {
        "query": "What is the weather like in San Francisco today?",
        "expected_route": "off_topic",
        "should_escalate": False,
        "category": "off_topic",
        "notes": "Completely unrelated — should be declined gracefully",
    },
    {
        "query": "Can you write me a Python script to parse CSV files?",
        "expected_route": "off_topic",
        "should_escalate": False,
        "category": "off_topic",
        "notes": "Out-of-scope technical task — not IT helpdesk",
    },
]

# Six-case suite for fast smoke tests (no API key needed for routing checks)
MINIMAL_TEST_SUITE: List[EvalCase] = [
    DEFAULT_TEST_SUITE[0],   # it_covered: VPN
    DEFAULT_TEST_SUITE[4],   # it_escalation: EC2
    DEFAULT_TEST_SUITE[7],   # hr_covered: PTO
    DEFAULT_TEST_SUITE[11],  # hr_escalation: executive equity
    DEFAULT_TEST_SUITE[15],  # ticket_create (routing only)
    DEFAULT_TEST_SUITE[18],  # off_topic
]


# ---------------------------------------------------------------------------
# LLM judges
# ---------------------------------------------------------------------------

_FAITHFULNESS_SYSTEM = (
    "You are an expert evaluator assessing AI answer quality. "
    "You must respond with ONLY a single integer 0–5. No explanation."
)

_FAITHFULNESS_TEMPLATE = """\
Evaluate whether the assistant's answer is FAITHFUL to the provided context.

Scoring scale:
  5 — Answer uses ONLY information found in the context
  4 — Answer mostly uses context; one minor addition from outside
  3 — Answer mixes context with notable outside knowledge
  2 — Answer partially uses context but adds significant outside knowledge
  1 — Answer mostly ignores context
  0 — Answer entirely contradicts or ignores context

Context:
{context}

Question: {query}

Answer: {answer}

Score (0–5):"""

_RELEVANCE_SYSTEM = (
    "You are an expert evaluator assessing AI answer quality. "
    "You must respond with ONLY a single integer 0–5. No explanation."
)

_RELEVANCE_TEMPLATE = """\
Evaluate whether the assistant's answer is RELEVANT to the question asked.

Scoring scale:
  5 — Directly and completely addresses the question
  4 — Mostly addresses the question; minor omissions
  3 — Partially addresses the question
  2 — Tangentially related; misses the main point
  1 — Barely addresses the question
  0 — Completely off-topic

Question: {query}

Answer: {answer}

Score (0–5):"""


def _judge_faithfulness(
    query: str, answer: str, chunks: list
) -> Optional[float]:
    """Return faithfulness score 0–1 using LLM-as-judge. Returns None on error."""
    from smartdesk.agents._llm import call_llm

    context_text = "\n\n".join(
        f"[{i+1}] {c['text'][:400]}" for i, c in enumerate(chunks)
    ) or "(No context retrieved)"

    prompt = _FAITHFULNESS_TEMPLATE.format(
        context=context_text, query=query, answer=answer[:600]
    )
    try:
        raw = call_llm(system=_FAITHFULNESS_SYSTEM, user=prompt, temperature=0.0)
        score = float(raw.strip().split()[0])
        return min(max(score / 5.0, 0.0), 1.0)
    except Exception as exc:
        print(f"  [eval] faithfulness judge error: {exc!r}")
        return None


def _judge_relevance(query: str, answer: str) -> Optional[float]:
    """Return answer-relevance score 0–1 using LLM-as-judge. Returns None on error."""
    from smartdesk.agents._llm import call_llm

    prompt = _RELEVANCE_TEMPLATE.format(query=query, answer=answer[:600])
    try:
        raw = call_llm(system=_RELEVANCE_SYSTEM, user=prompt, temperature=0.0)
        score = float(raw.strip().split()[0])
        return min(max(score / 5.0, 0.0), 1.0)
    except Exception as exc:
        print(f"  [eval] relevance judge error: {exc!r}")
        return None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_evaluation(
    test_cases: Optional[List[EvalCase]] = None,
    *,
    graph: Optional[Any] = None,
    skip_llm_judges: bool = False,
    verbose: bool = False,
) -> EvalReport:
    """Run the evaluation suite and return an EvalReport.

    Args:
        test_cases:       Cases to evaluate. Defaults to DEFAULT_TEST_SUITE.
        graph:            Compiled LangGraph graph. Built automatically if None.
        skip_llm_judges:  Skip faithfulness + relevance LLM calls (fast, free).
        verbose:          Print per-case detail during the run.

    Returns:
        EvalReport TypedDict with all metrics and per-case results.
    """
    if test_cases is None:
        test_cases = DEFAULT_TEST_SUITE

    # Build graph if not provided
    if graph is None:
        from smartdesk.orchestrator.graph import build_orchestrator
        print("[eval] Building orchestrator graph...")
        graph = build_orchestrator()
        print("[eval] Graph ready.\n")

    from smartdesk.agents.supervisor import supervisor_node
    from smartdesk.orchestrator.graph import run_once

    results: List[EvalResult] = []

    for i, case in enumerate(test_cases, start=1):
        query = case["query"]
        routing_only = case.get("routing_only", False)

        if verbose:
            print(f"[{i:02d}/{len(test_cases)}] {case['category']:15s} | {query[:70]}")

        t0 = time.monotonic()
        error: Optional[str] = None
        actual_route = "off_topic"
        actual_escalate = False
        confidence = 0.0
        response = ""
        chunks: list = []

        try:
            if routing_only:
                # Only run supervisor — don't trigger HITL or email flows
                sup_state = supervisor_node({"query": query})
                actual_route = sup_state.get("route", "off_topic")
                response = f"[routing-only — downstream not run]"
            else:
                result = run_once(graph, {"query": query})
                actual_route = result.get("route", "off_topic")
                actual_escalate = result.get("should_escalate", False)
                confidence = result.get("confidence_score", 0.0)
                response = result.get("response", "")
                chunks = result.get("retrieved_chunks", [])

        except Exception as exc:
            error = str(exc)
            if verbose:
                print(f"         ✗ ERROR: {exc!r}")

        latency = time.monotonic() - t0

        route_correct = actual_route == case["expected_route"]
        escalation_correct = (
            actual_escalate == case["should_escalate"]
            if not routing_only
            else True  # routing-only cases are excluded from escalation metrics
        )

        # LLM judges — only for non-routing-only KB cases with a real answer
        faith: Optional[float] = None
        rel: Optional[float] = None

        if (
            not skip_llm_judges
            and not routing_only
            and not error
            and response
            and actual_route in ("it_kb", "hr_kb", "combined_kb")
        ):
            if verbose:
                print(f"         → judging faithfulness + relevance...")
            faith = _judge_faithfulness(query, response, chunks)
            rel = _judge_relevance(query, response)

        result_entry: EvalResult = {
            "query": query,
            "category": case["category"],
            "expected_route": case["expected_route"],
            "actual_route": actual_route,
            "route_correct": route_correct,
            "expected_escalate": case["should_escalate"],
            "actual_escalate": actual_escalate,
            "escalation_correct": escalation_correct,
            "confidence_score": confidence,
            "response_preview": response[:120].replace("\n", " "),
            "faithfulness_score": faith,
            "relevance_score": rel,
            "latency_seconds": round(latency, 3),
            "routing_only": routing_only,
            "error": error,
        }
        results.append(result_entry)

        if verbose:
            mark = "✓" if route_correct else "✗"
            esc = " ↑escalate" if actual_escalate else ""
            print(f"         {mark} → {actual_route}{esc}  conf={confidence:.2f}  {latency:.2f}s")

    return _compute_metrics(results)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(results: List[EvalResult]) -> EvalReport:
    total = len(results)
    correct_routes = sum(1 for r in results if r["route_correct"])

    # Escalation — only KB cases that ran the full pipeline
    kb_full = [
        r for r in results
        if not r["routing_only"] and r["expected_route"] in ("it_kb", "hr_kb", "combined_kb")
    ]
    tp = sum(1 for r in kb_full if r["expected_escalate"] and r["actual_escalate"])
    fp = sum(1 for r in kb_full if not r["expected_escalate"] and r["actual_escalate"])
    fn = sum(1 for r in kb_full if r["expected_escalate"] and not r["actual_escalate"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )

    # RAG quality
    faith_scores = [r["faithfulness_score"] for r in results if r["faithfulness_score"] is not None]
    rel_scores = [r["relevance_score"] for r in results if r["relevance_score"] is not None]

    avg_faith = sum(faith_scores) / len(faith_scores) if faith_scores else None
    avg_rel = sum(rel_scores) / len(rel_scores) if rel_scores else None

    # Confidence calibration
    answered = [r for r in kb_full if not r["actual_escalate"] and r["confidence_score"] > 0]
    escalated = [r for r in kb_full if r["actual_escalate"]]
    avg_conf_ans = sum(r["confidence_score"] for r in answered) / len(answered) if answered else None
    avg_conf_esc = sum(r["confidence_score"] for r in escalated) / len(escalated) if escalated else None

    avg_latency = sum(r["latency_seconds"] for r in results) / total if total else 0.0

    return EvalReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_cases=total,
        routing_accuracy=round(correct_routes / total, 4) if total else 0.0,
        routing_correct=correct_routes,
        escalation_precision=round(precision, 4) if precision is not None else None,
        escalation_recall=round(recall, 4) if recall is not None else None,
        escalation_f1=round(f1, 4) if f1 is not None else None,
        avg_faithfulness=round(avg_faith, 4) if avg_faith is not None else None,
        avg_relevance=round(avg_rel, 4) if avg_rel is not None else None,
        avg_confidence_answered=round(avg_conf_ans, 4) if avg_conf_ans is not None else None,
        avg_confidence_escalated=round(avg_conf_esc, 4) if avg_conf_esc is not None else None,
        avg_latency_seconds=round(avg_latency, 3),
        results=results,
    )


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(report: EvalReport) -> None:
    """Print a human-readable evaluation report to stdout."""
    sep = "─" * 62
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          SmartDesk AI — Evaluation Report               ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Timestamp : {report['timestamp']}")
    print(f"  Test cases: {report['total_cases']}")
    print()

    # ── Routing ──────────────────────────────────────────────────────────────
    acc = report["routing_accuracy"]
    correct = report["routing_correct"]
    total = report["total_cases"]
    print(f"ROUTING ACCURACY  {correct}/{total}  ({acc:.1%})")
    print(sep)

    # Per-category breakdown
    by_cat: Dict[str, Dict[str, int]] = {}
    for r in report["results"]:
        cat = r["expected_route"]
        by_cat.setdefault(cat, {"correct": 0, "total": 0})
        by_cat[cat]["total"] += 1
        if r["route_correct"]:
            by_cat[cat]["correct"] += 1

    for route, counts in sorted(by_cat.items()):
        mark = "✓" if counts["correct"] == counts["total"] else "✗"
        bar = "█" * counts["correct"] + "░" * (counts["total"] - counts["correct"])
        print(f"  {mark} {route:<15s} {counts['correct']}/{counts['total']}  {bar}")

    # Show failures
    failures = [r for r in report["results"] if not r["route_correct"]]
    if failures:
        print()
        print("  Misrouted cases:")
        for r in failures:
            print(f"    ✗ [{r['category']}] expected={r['expected_route']}  "
                  f"got={r['actual_route']}")
            print(f"      \"{r['query'][:70]}\"")

    print()

    # ── Escalation ────────────────────────────────────────────────────────────
    print("ESCALATION (KB cases — full pipeline only)")
    print(sep)
    p = report["escalation_precision"]
    r_ = report["escalation_recall"]
    f = report["escalation_f1"]
    if p is None:
        print("  No KB cases with full pipeline in this suite.")
    else:
        print(f"  Precision : {p:.2f}  (of escalated → should have escalated)")
        print(f"  Recall    : {r_:.2f}  (of should-escalate → actually escalated)")
        print(f"  F1        : {f:.2f}")
        if report["avg_confidence_answered"] is not None:
            print(f"  Avg conf (answered)  : {report['avg_confidence_answered']:.2f}")
        if report["avg_confidence_escalated"] is not None:
            print(f"  Avg conf (escalated) : {report['avg_confidence_escalated']:.2f}")
    print()

    # ── RAG quality ───────────────────────────────────────────────────────────
    print("RAG QUALITY (LLM-judged)")
    print(sep)
    if report["avg_faithfulness"] is None and report["avg_relevance"] is None:
        print("  Skipped (--skip-llm-judges) or no KB cases ran.")
    else:
        fa = report["avg_faithfulness"]
        ra = report["avg_relevance"]
        print(f"  Faithfulness    : {fa:.2f}" if fa is not None else "  Faithfulness    : n/a")
        print(f"  Answer Relevance: {ra:.2f}" if ra is not None else "  Answer Relevance: n/a")
    print()

    # ── Performance ───────────────────────────────────────────────────────────
    print("PERFORMANCE")
    print(sep)
    print(f"  Avg latency : {report['avg_latency_seconds']:.2f}s per query")
    print()

    # ── Per-case detail ───────────────────────────────────────────────────────
    print("CASE DETAIL")
    print(sep)
    for r in report["results"]:
        mark = "✓" if r["route_correct"] else "✗"
        esc = "↑" if r["actual_escalate"] else " "
        conf = f"conf={r['confidence_score']:.2f}" if r["confidence_score"] else "       "
        faith_str = f" faith={r['faithfulness_score']:.2f}" if r["faithfulness_score"] is not None else ""
        rel_str = f" rel={r['relevance_score']:.2f}" if r["relevance_score"] is not None else ""
        routing_tag = " [routing-only]" if r["routing_only"] else ""
        err = f" ⚠ {r['error'][:40]}" if r["error"] else ""

        print(
            f"  {mark} [{r['category']:<15s}] {esc} {r['actual_route']:<15s} "
            f"{conf}{faith_str}{rel_str}{routing_tag}{err}"
        )
        print(f"      \"{r['query'][:72]}\"")

    print()
    print("═" * 62)


# ---------------------------------------------------------------------------
# LangSmith Datasets & Experiments push
# ---------------------------------------------------------------------------

def push_to_langsmith(
    test_cases: List[EvalCase],
    graph: Any,
    *,
    dataset_name: str = "SmartDesk AI Eval Suite",
    experiment_prefix: str = "smartdesk-eval",
    skip_llm_judges: bool = False,
) -> None:
    """Push eval test cases and results to LangSmith Datasets & Experiments.

    This creates (or resets) a named dataset in LangSmith from the test cases,
    then runs ``langsmith.evaluate()`` against it so results appear in the
    Datasets & Experiments tab — separate from the Tracing/Projects view.

    Requires:
      - LANGCHAIN_API_KEY set in .env
      - LANGCHAIN_TRACING_V2=true set in .env
      - pip install langsmith

    Args:
        test_cases:        The same list passed to run_evaluation().
        graph:             Compiled LangGraph graph from build_orchestrator().
        dataset_name:      LangSmith dataset name (created if it doesn't exist).
        experiment_prefix: Prefix for the experiment name shown in LangSmith UI.
        skip_llm_judges:   Skip faithfulness + relevance LLM evaluators.
    """
    try:
        from langsmith import Client
        from langsmith import evaluate as ls_evaluate
    except ImportError:
        print(
            "[langsmith] langsmith package not found.\n"
            "            Run: pip install langsmith\n"
            "            Skipping Datasets & Experiments push."
        )
        return

    from smartdesk.config import settings
    if not settings.langchain_api_key:
        print(
            "[langsmith] LANGCHAIN_API_KEY not set in .env — "
            "skipping Datasets & Experiments push."
        )
        return

    client = Client()

    # ── Step 1: Create or reset dataset ─────────────────────────────────────
    print(f"\n[langsmith] Preparing dataset '{dataset_name}'...")
    if client.has_dataset(dataset_name=dataset_name):
        dataset = client.read_dataset(dataset_name=dataset_name)
        # Delete existing examples so the dataset stays in sync with test_cases
        existing = list(client.list_examples(dataset_id=dataset.id))
        for ex in existing:
            client.delete_example(ex.id)
        print(f"[langsmith] Cleared {len(existing)} existing example(s).")
    else:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description=(
                "SmartDesk AI evaluation suite — routing accuracy, "
                "escalation precision/recall, RAG faithfulness & relevance."
            ),
        )
        print(f"[langsmith] Created new dataset '{dataset_name}'.")

    # ── Step 2: Upload examples ──────────────────────────────────────────────
    client.create_examples(
        inputs=[
            {
                "query": c["query"],
                "routing_only": c.get("routing_only", False),
            }
            for c in test_cases
        ],
        outputs=[
            {
                "expected_route": c["expected_route"],
                "should_escalate": c["should_escalate"],
            }
            for c in test_cases
        ],
        metadata=[
            {"category": c["category"], "notes": c.get("notes", "")}
            for c in test_cases
        ],
        dataset_id=dataset.id,
    )
    print(f"[langsmith] Uploaded {len(test_cases)} example(s).")

    # ── Step 3: Define target function ───────────────────────────────────────
    from smartdesk.agents.supervisor import supervisor_node
    from smartdesk.orchestrator.graph import run_once

    def _target(inputs: dict) -> dict:
        """Runs one test case through SmartDesk; returns outputs for evaluators."""
        query = inputs["query"]
        routing_only = inputs.get("routing_only", False)

        if routing_only:
            sup_state = supervisor_node({"query": query})
            return {
                "query": query,
                "route": sup_state.get("route", "off_topic"),
                "should_escalate": False,
                "response": "[routing-only — downstream not run]",
                "confidence_score": 0.0,
                "retrieved_chunks": [],
            }

        result = run_once(graph, {"query": query})
        return {
            "query": query,
            "route": result.get("route", "off_topic"),
            "should_escalate": result.get("should_escalate", False),
            "response": result.get("response", ""),
            "confidence_score": result.get("confidence_score", 0.0),
            "retrieved_chunks": result.get("retrieved_chunks", []),
        }

    # ── Step 4: Define evaluators ────────────────────────────────────────────

    def _eval_route(outputs: dict, reference_outputs: dict) -> dict:
        correct = outputs.get("route") == reference_outputs.get("expected_route")
        return {"key": "route_correct", "score": int(correct)}

    def _eval_escalation(outputs: dict, reference_outputs: dict) -> dict:
        # routing_only cases are excluded — always pass
        if outputs.get("route") in ("create_ticket", "ticket_status"):
            return {"key": "escalation_correct", "score": 1}
        correct = (
            outputs.get("should_escalate") == reference_outputs.get("should_escalate")
        )
        return {"key": "escalation_correct", "score": int(correct)}

    def _eval_confidence(outputs: dict, reference_outputs: dict) -> dict:
        return {
            "key": "confidence_score",
            "score": round(float(outputs.get("confidence_score", 0.0)), 4),
        }

    evaluators = [_eval_route, _eval_escalation, _eval_confidence]

    if not skip_llm_judges:
        def _eval_faithfulness(outputs: dict, reference_outputs: dict) -> dict:
            route = outputs.get("route", "")
            if route not in ("it_kb", "hr_kb", "combined_kb"):
                return {"key": "faithfulness", "score": None}
            if outputs.get("should_escalate"):
                return {"key": "faithfulness", "score": None}
            score = _judge_faithfulness(
                outputs.get("query", ""),
                outputs.get("response", ""),
                outputs.get("retrieved_chunks", []),
            )
            return {"key": "faithfulness", "score": score}

        def _eval_relevance(outputs: dict, reference_outputs: dict) -> dict:
            route = outputs.get("route", "")
            if route not in ("it_kb", "hr_kb", "combined_kb"):
                return {"key": "relevance", "score": None}
            if outputs.get("should_escalate"):
                return {"key": "relevance", "score": None}
            score = _judge_relevance(
                outputs.get("query", ""),
                outputs.get("response", ""),
            )
            return {"key": "relevance", "score": score}

        evaluators.extend([_eval_faithfulness, _eval_relevance])

    # ── Step 5: Run experiment ───────────────────────────────────────────────
    from datetime import datetime, timezone
    suffix = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    full_prefix = f"{experiment_prefix}-{suffix}"

    print(f"[langsmith] Running experiment '{full_prefix}' ({len(test_cases)} cases)...")
    print("            This reruns each query through the graph — may take a moment.\n")

    results = ls_evaluate(
        _target,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix=full_prefix,
        metadata={
            "project": "SmartDesk AI",
            "suite": "full" if len(test_cases) > 6 else "minimal",
            "skip_llm_judges": skip_llm_judges,
        },
    )

    # Print URL — langsmith evaluate() returns an ExperimentResults object
    try:
        url = results._results[0].url if hasattr(results, "_results") else None
    except Exception:
        url = None

    print("\n[langsmith] ✓ Experiment pushed to LangSmith.")
    print("            Open: https://smith.langchain.com → Datasets & Experiments")
    print(f"            Dataset : {dataset_name}")
    print(f"            Experiment: {full_prefix}")
    if url:
        print(f"            Direct URL: {url}")

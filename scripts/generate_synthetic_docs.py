"""Generate synthetic IT/HR Q&A docs with an LLM (Option C — synthetic half).

Run with:
    python scripts/generate_synthetic_docs.py

This produces .json files under data/knowledge_base/it_docs/ and
data/knowledge_base/hr_docs/ that load_all_sources() in ingestion.py
will pick up automatically.

The deliberately-uncovered topics at the bottom of this file are the
topics the knowledge base intentionally knows nothing about — they are
what triggers the escalation path (decide_escalation in retriever.py)
and are your primary test cases for that logic.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Make smartdesk importable when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from smartdesk.config import settings  # noqa: E402

# ---------------------------------------------------------------------------
# Topics to COVER (will be generated and indexed)
# ---------------------------------------------------------------------------

IT_TOPICS: list[str] = [
    "How to set up and connect to the company VPN (Windows and macOS)",
    "How to reset your Active Directory / SSO password",
    "How to request a new laptop or peripheral hardware",
    "Wi-Fi troubleshooting: can't connect to corporate network",
    "How to install approved software from the software catalog",
    "How to request access to a new software tool or application",
    "Setting up multi-factor authentication (MFA / TOTP)",
    "How to report a phishing or suspicious email",
    "Remote desktop and screen-sharing setup (Windows and Mac)",
    "Printer setup and network printer troubleshooting",
    "How to access shared network drives and file shares",
    "Outlook / email client setup on a new device",
    "How to submit an IT support ticket and what to include",
]

HR_TOPICS: list[str] = [
    "Annual leave entitlement and how to book vacation",
    "Sick leave policy and how to report an absence",
    "Performance review process, timeline, and self-assessment tips",
    "Employee onboarding checklist and first-week steps",
    "Benefits enrollment: health insurance and dental options",
    "Expense reimbursement process and submission deadlines",
    "Code of conduct and workplace behavior expectations",
    "Remote and hybrid work policy: eligibility and expectations",
    "Training budget and professional development request process",
    "How to submit a grievance, complaint, or HR concern",
    "Payroll schedule and how to access pay stubs",
    "Employee referral program and bonus policy",
]

# ---------------------------------------------------------------------------
# Topics DELIBERATELY left uncovered (no docs generated for these)
# Used to test that decide_escalation() correctly routes to ticket creation.
# ---------------------------------------------------------------------------

IT_GAPS: list[str] = [
    "GDPR personal data deletion request process",
    "Provisioning a new AWS EC2 instance or cloud resource",
    "Kubernetes cluster access and kubectl setup",
]

HR_GAPS: list[str] = [
    "Parental leave and maternity / paternity entitlement details",
    "HIPAA compliance training and certification requirements",
    "Executive compensation, equity, and stock option vesting",
]


# ---------------------------------------------------------------------------
# LLM caller — supports OpenAI, Anthropic, Groq
# ---------------------------------------------------------------------------

def call_llm(prompt: str, retries: int = 3) -> str:
    """Call the configured LLM and return the raw text response."""
    provider = settings.llm_provider.lower()
    last_exc: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            if provider == "openai":
                from openai import OpenAI
                client = OpenAI(api_key=settings.llm_api_key)
                resp = client.chat.completions.create(
                    model=settings.llm_model or "gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                return resp.choices[0].message.content.strip()

            elif provider == "anthropic":
                import anthropic
                client = anthropic.Anthropic(api_key=settings.llm_api_key)
                msg = client.messages.create(
                    model=settings.llm_model or "claude-haiku-4-5-20251001",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                return msg.content[0].text.strip()

            elif provider == "groq":
                from groq import Groq
                client = Groq(api_key=settings.llm_api_key)
                resp = client.chat.completions.create(
                    model=settings.llm_model or "llama3-8b-8192",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                return resp.choices[0].message.content.strip()

            else:
                raise ValueError(f"Unsupported LLM provider: {provider!r}. "
                                 "Set LLM_PROVIDER in .env to openai | anthropic | groq.")

        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                wait = attempt * 2
                print(f"    [retry {attempt}/{retries}] {exc} — waiting {wait}s")
                time.sleep(wait)

    raise RuntimeError(f"LLM call failed after {retries} attempts") from last_exc


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_JSON_INSTRUCTION = (
    "Respond ONLY with a valid JSON object — no markdown, no code fences, "
    "no explanation outside the JSON."
)

def _it_prompt(topic: str) -> str:
    return (
        f"You are writing entries for a company IT helpdesk knowledge base. "
        f"Write a detailed FAQ entry for company employees about: '{topic}'.\n\n"
        f"IMPORTANT: If the topic involves any operating-system-specific steps "
        f"(file paths, menus, settings, dialogs), you MUST cover both Windows and macOS. "
        f"Label each section clearly (e.g. 'Windows:' and 'macOS:') so employees on "
        f"either platform can follow the instructions. Do not write steps for only one OS.\n\n"
        f"Return a JSON object with exactly these keys:\n"
        f"  \"title\": short title for this entry (do not include a specific OS in the title "
        f"unless the topic is genuinely OS-exclusive)\n"
        f"  \"question\": the natural question an employee would ask\n"
        f"  \"answer\": a clear, practical answer with numbered steps where applicable "
        f"(e.g. 'Step 1: Open ... Step 2: Click ...'). Include specific tool names, "
        f"menu paths, or credentials format where relevant. Aim for 8-12 sentences or steps.\n\n"
        f"{_JSON_INSTRUCTION}"
    )

def _hr_prompt(topic: str) -> str:
    return (
        f"You are writing entries for a company HR knowledge base. "
        f"Write a detailed FAQ entry for company employees about: '{topic}'.\n\n"
        f"Return a JSON object with exactly these keys:\n"
        f"  \"title\": short title for this entry\n"
        f"  \"question\": the natural question an employee would ask\n"
        f"  \"answer\": a clear, practical policy answer. Include specific timelines, "
        f"procedures, or contact points where applicable. Aim for 6-10 sentences.\n\n"
        f"{_JSON_INSTRUCTION}"
    )


# ---------------------------------------------------------------------------
# Parse LLM response into a doc dict
# ---------------------------------------------------------------------------

def _parse_response(raw: str, topic: str) -> dict:
    """Strip markdown fences if present and parse JSON."""
    cleaned = raw.strip()
    # Strip ```json ... ``` or ``` ... ```
    for fence in ("```json", "```"):
        if cleaned.startswith(fence):
            cleaned = cleaned[len(fence):]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: treat entire response as the answer text
        print(f"    WARNING: JSON parse failed for '{topic}' — storing raw text.")
        return {
            "title": topic,
            "text": raw.strip(),
            "source": "synthetic-llm",
        }

    question = data.get("question", "").strip()
    answer = data.get("answer", "").strip()
    return {
        "title": data.get("title", topic).strip(),
        "text": f"Q: {question}\nA: {answer}" if question else answer,
        "source": "synthetic-llm",
    }


# ---------------------------------------------------------------------------
# Generation loops
# ---------------------------------------------------------------------------

def generate_docs(topics: list[str], prompt_fn, label: str) -> list[dict]:
    docs = []
    for topic in topics:
        print(f"  → {topic}")
        try:
            raw = call_llm(prompt_fn(topic))
            doc = _parse_response(raw, topic)
            docs.append(doc)
        except Exception as exc:
            print(f"    ERROR: {exc} — skipping this topic.")
    print(f"  Generated {len(docs)}/{len(topics)} {label} docs.")
    return docs


def _title_to_slug(title: str) -> str:
    """Convert a document title to a snake_case filename slug.

    Examples:
        "VPN Setup Guide"          → "vpn_setup_guide"
        "MFA / TOTP Enrollment"    → "mfa_totp_enrollment"
        "Annual Leave & Vacation"  → "annual_leave_vacation"
    """
    import re
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return slug or "doc"


def save_docs(docs: list[dict], out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    seen_slugs: set[str] = set()
    for i, doc in enumerate(docs):
        title = doc.get("title", "")
        slug = _title_to_slug(title) if title else ""

        # Fallback to sequential name if slug is empty or collides
        if not slug or slug in seen_slugs:
            slug = f"{prefix}_{i + 1:03d}"
        seen_slugs.add(slug)

        path = out_dir / f"{slug}.json"
        path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"    saved {path.name}")


def update_data_readme(it_gaps: list[str], hr_gaps: list[str]) -> None:
    """Append deliberately-uncovered topics to data/README.md if not already there."""
    readme = Path(__file__).parent.parent / "data" / "README.md"
    text = readme.read_text(encoding="utf-8")

    if "Deliberately uncovered" in text:
        return  # already written

    it_lines = "\n".join(f"- {t}" for t in it_gaps)
    hr_lines = "\n".join(f"- {t}" for t in hr_gaps)
    section = (
        f"\n## Deliberately uncovered topics (escalation test cases)\n\n"
        f"### IT gaps\n{it_lines}\n\n"
        f"### HR gaps\n{hr_lines}\n"
    )
    readme.write_text(text + section, encoding="utf-8")
    print("  Updated data/README.md with deliberate gap list.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    base = Path(__file__).parent.parent / "data" / "knowledge_base"
    it_out = base / "it_docs"
    hr_out = base / "hr_docs"

    print(f"LLM provider: {settings.llm_provider} / model: {settings.llm_model or '(default)'}")
    print()

    print(f"=== Generating {len(IT_TOPICS)} IT docs ===")
    it_docs = generate_docs(IT_TOPICS, _it_prompt, "IT")
    save_docs(it_docs, it_out, "synthetic_it")
    print()

    print(f"=== Generating {len(HR_TOPICS)} HR docs ===")
    hr_docs = generate_docs(HR_TOPICS, _hr_prompt, "HR")
    save_docs(hr_docs, hr_out, "synthetic_hr")
    print()

    update_data_readme(IT_GAPS, HR_GAPS)

    print("=== Done ===")
    print(f"  IT docs written : {len(it_docs)} → {it_out}")
    print(f"  HR docs written : {len(hr_docs)} → {hr_out}")
    print()
    print("Deliberately uncovered topics (use these to test escalation):")
    print("  IT :", IT_GAPS)
    print("  HR :", HR_GAPS)
    print()
    print("Next step: python scripts/build_knowledge_base.py")


if __name__ == "__main__":
    main()

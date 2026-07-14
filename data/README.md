# Data layout

```
data/
├── raw/                  # Unmodified source files (HF dataset dumps, Kaggle CSVs, etc.)
├── processed/            # Chunked/cleaned docs + the persisted vector index
└── knowledge_base/
    ├── it_docs/          # Final IT knowledge-base documents (one file per topic, or a single corpus file)
    └── hr_docs/          # Final HR knowledge-base documents
```

## Sourcing the knowledge base

TODO: pick a source per domain. Suggestions from the capstone guide:

- Hugging Face datasets (search for IT helpdesk / HR FAQ style datasets)
- Kaggle (IT support ticket datasets, HR policy datasets)
- LLM-generated synthetic Q&A docs — see `scripts/generate_synthetic_docs.py`

## Sizing target

Aim for roughly **30–50+ chunks per domain** (IT and HR) — enough for
retrieval to be meaningful, but not so much that you can't reason about
what's actually in there.

## Deliberate gaps (important)

Don't try to cover every possible IT/HR topic. Intentionally leave a few
topics uncovered in each domain. A knowledge base with no gaps can't
demonstrate the escalation path (`rag/retriever.py::decide_escalation`) —
you need queries that genuinely have no good answer in the KB so you can
show the agent saying "I don't know" and routing to ticket creation,
instead of guessing.

Keep a short list here of which topics you left out on purpose, so you can
reference them when testing:

- TODO: list deliberately-uncovered IT topics
- TODO: list deliberately-uncovered HR topics

## Deliberately uncovered topics (escalation test cases)

### IT gaps
- GDPR personal data deletion request process
- Provisioning a new AWS EC2 instance or cloud resource
- Kubernetes cluster access and kubectl setup

### HR gaps
- Parental leave and maternity / paternity entitlement details
- HIPAA compliance training and certification requirements
- Executive compensation, equity, and stock option vesting

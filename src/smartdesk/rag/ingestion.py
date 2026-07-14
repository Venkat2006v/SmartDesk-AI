"""Load raw IT/HR sources and chunk them for embedding.

Sources (real + synthetic (if available)) combined into a single list of Document dicts, then chunked for embedding.:
  - HR: strova-ai/hr-policies-qa-dataset from Hugging Face
  - IT: Console-AI/IT-helpdesk-synthetic-tickets from Hugging Face
  - IT/HR: local .txt and .json files written by generate_synthetic_docs.py
           or dropped manually into data/knowledge_base/{it,hr}_docs/
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import List, TypedDict

from smartdesk.config import settings




class Document(TypedDict):
    id: str
    domain: str  # "it" | "hr"
    title: str
    text: str
    source: str

# ---------------------------------------------------------------------------
# Load data from Hugging Face dataset or local files, and chunk them for embedding.
# ---------------------------------------------------------------------------

def load_hugging_face_dataset() -> List[Document]:
    """Pull HR Q&A pairs from strova-ai/hr-policies-qa-dataset.

    Column names are detected automatically so this stays robust if the
    dataset schema changes. Run `print(ds["train"][0])` once to inspect.
    """
    try:
        from datasets import load_dataset  # pip install datasets
    except ImportError:
        print("[ingestion] 'datasets' not installed — skipping HF HR source.")
        print("            Run: pip install datasets")
        return []
    print("[ingestion] Loading strova-ai/hr-policies-qa-dataset from HuggingFace...")
    HR_DATASET_NAME = "strova-ai/hr-policies-qa-dataset"
    ds = load_dataset(HR_DATASET_NAME, split="train")
    sample = ds[0]
    question_col = next(
        (c for c in ["question", "Question", "query", "prompt", "input"] if c in sample),
        None,
    )
    answer_col = next(
        (c for c in ["answer", "Answer", "response", "context", "text", "output"] if c in sample),
        None,
    )

    if not answer_col:
        print(f"[ingestion] WARNING: could not detect answer column. Columns: {list(sample.keys())}")
        return []

    docs: List[Document] = []
    for row in ds:
        question = row.get(question_col, "").strip() if question_col else ""
        answer = row.get(answer_col, "").strip()
        if not answer:
            continue
        text = f"Q: {question}\nA: {answer}" if question else answer
        docs.append(
            Document(
                id=str(uuid.uuid4()),
                domain="hr",
                title=question or "HR Policy",
                text=text,
                source="strova-ai/hr-policies-qa-dataset",
            )
        )
    print(f"[ingestion] HF HR dataset: {len(docs)} records loaded.")
    return docs

def load_huggingface_it() -> List[Document]:
    """Pull IT docs from a Hugging Face dataset.
    """
    IT_DATASET_NAME = "Console-AI/IT-helpdesk-synthetic-tickets"   # e.g. "Tagore978/it-support-faq"
    try:
        from datasets import load_dataset
    except ImportError:
        print("[ingestion] 'datasets' not installed — skipping HF IT source.")
        return []
    print(f"[ingestion] Loading {IT_DATASET_NAME} from HuggingFace...")
    ds = load_dataset(IT_DATASET_NAME, split="train")
    sample = ds[0]
    question_col = next(
        (c for c in ["question", "Question", "query", "prompt", "input"] if c in sample),
        None,
    )
    answer_col = next(
        (c for c in ["answer", "Answer", "response", "context", "text", "output"] if c in sample),
        None,
    )

    docs: List[Document] = []
    for row in ds:
        question = row.get(question_col, "").strip() if question_col else ""
        answer = row.get(answer_col, "").strip() if answer_col else ""
        if not answer:
            continue
        text = f"Q: {question}\nA: {answer}" if question else answer
        docs.append(
            Document(
                id=str(uuid.uuid4()),
                domain="it",
                title=question or "IT Helpdesk",
                text=text,
                source=IT_DATASET_NAME,
            )
        )

    print(f"[ingestion] HF IT dataset: {len(docs)} records loaded.")
    return docs


# ---------------------------------------------------------------------------
# Local file loader  (synthetic docs + any CSVs placed in local data/knowledge_base/)
# ---------------------------------------------------------------------------
def load_local_files(domain: str) -> List[Document]:
    """Load .txt and .json files from data/knowledge_base/{domain}_docs/.

    Supports:
      .txt  — entire file becomes one Document
      .json — either a single {title, text/answer/content} dict, or a list of them
    """
    base = Path(settings.knowledge_base_dir) / f"{domain}_docs"
    if not base.exists():
        return []

    docs: List[Document] = []
    for path in sorted(base.glob("**/*")):
        if path.name.startswith("."):
            continue

        if path.suffix == ".txt":
            text = path.read_text(encoding="utf-8").strip()
            if text:
                docs.append(
                    Document(
                        id=str(uuid.uuid4()),
                        domain=domain,
                        title=path.stem.replace("_", " ").title(),
                        text=text,
                        source=str(path),
                    )
                )

        elif path.suffix == ".json":
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                items = raw if isinstance(raw, list) else [raw]
                for item in items:
                    text = (
                        item.get("text")
                        or item.get("answer")
                        or item.get("content")
                        or ""
                    ).strip()
                    title = (
                        item.get("title")
                        or item.get("question")
                        or path.stem
                    )
                    if text:
                        docs.append(
                            Document(
                                id=str(uuid.uuid4()),
                                domain=domain,
                                title=title,
                                text=text,
                                source=str(path),
                            )
                        )
            except (json.JSONDecodeError, KeyError) as exc:
                print(f"[ingestion] skipping {path.name}: {exc}")

    print(f"[ingestion] Local {domain.upper()} files: {len(docs)} documents loaded.")
    return docs



# ---------------------------------------------------------------------------
# Public API: load_all_sources() and chunk_documents()

def load_all_sources() -> List[Document]:
    """Load raw IT + HR documents from all sources (Hugging Face datasets + local files) 
    and return as a list of Document dicts.
    Execution Order:
    1. Load HR docs from strova-ai/hr-policies-qa-dataset (Hugging Face)
    2. Load IT docs from local files in data/knowledge_base/
    """
    docs: List[Document] = []
    docs.extend(load_hugging_face_dataset())
    docs.extend(load_huggingface_it())
    docs.extend(load_local_files("it"))
    docs.extend(load_local_files("hr"))
    return docs

def chunk_documents(
    docs: List[Document],
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Document]:
    """Split long documents into retrieval-sized chunks.

    Short docs (len <= chunk_size) pass through unchanged — most Q&A pairs
    from the HF dataset and the synthetic generator will fall into this
    category. Longer policy texts are split with a sliding window so
    context carries across chunk boundaries.

    Args:
        chunk_size: Maximum character length per chunk.
        overlap:    Character overlap between adjacent chunks to avoid
                    cutting a sentence mid-thought.
    """
    chunks: List[Document] = []

    for doc in docs:
        text = doc["text"]

        if len(text) <= chunk_size:
            chunks.append(
                Document(
                    id=str(uuid.uuid4()),
                    domain=doc["domain"],
                    title=doc["title"],
                    text=text,
                    source=doc["source"],
                )
            )
            continue

        # Sliding-window split
        start = 0
        chunk_num = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    Document(
                        id=str(uuid.uuid4()),
                        domain=doc["domain"],
                        title=f"{doc['title']} (part {chunk_num + 1})",
                        text=chunk_text,
                        source=doc["source"],
                    )
                )
                chunk_num += 1
            if end == len(text):
                break
            start += chunk_size - overlap

    it_chunks = sum(1 for c in chunks if c["domain"] == "it")
    hr_chunks = sum(1 for c in chunks if c["domain"] == "hr")
    print(
        f"[ingestion] Chunked — IT: {it_chunks}, HR: {hr_chunks}, "
        f"total: {len(chunks)} (from {len(docs)} docs)"
    )
    return chunks
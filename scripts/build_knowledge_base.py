"""Build the Qdrant vector index from all knowledge-base sources.

Run once (and again whenever the KB changes) before starting the main app:

    python scripts/build_knowledge_base.py

Prerequisites:
  1. .env populated with LLM_API_KEY, EMBEDDING_*, QDRANT_* settings
  2. Qdrant running: `docker run -p 6333:6333 qdrant/qdrant`
     OR using embedded local mode (set QDRANT_URL="" and configure
     QdrantClient(path=...) in vector_store.py)
  3. Knowledge-base docs present:
       data/knowledge_base/hr_docs/   ← strova-ai HF dataset + synthetic HR docs
       data/knowledge_base/it_docs/   ← synthetic IT docs (+ HF IT dataset if added)
     If empty, run first: python scripts/generate_synthetic_docs.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Make smartdesk importable when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from smartdesk.config import settings                             # noqa: E402
from smartdesk.rag.ingestion import load_all_sources, chunk_documents  # noqa: E402
from smartdesk.rag.vector_store import get_vector_store           # noqa: E402


MIN_RECOMMENDED_CHUNKS = 30   # warn if below this threshold per domain


def _check_counts(chunks: list) -> None:
    """Warn if either domain has fewer chunks than recommended."""
    it_n = sum(1 for c in chunks if c["domain"] == "it")
    hr_n = sum(1 for c in chunks if c["domain"] == "hr")
    for domain, n in [("IT", it_n), ("HR", hr_n)]:
        if n < MIN_RECOMMENDED_CHUNKS:
            print(
                f"  WARNING: only {n} {domain} chunks (recommended ≥ {MIN_RECOMMENDED_CHUNKS}). "
                f"Consider adding more source docs or lowering chunk_size in ingestion.py."
            )


def main() -> None:
    print("=" * 50)
    print("SmartDesk AI — Knowledge Base Builder")
    print("=" * 50)
    print(f"  Vector store : {settings.vector_store_backend}")
    print(f"  Qdrant URL   : {settings.qdrant_url}")
    print(f"  Collection   : {settings.qdrant_collection_name}")
    print(f"  KB directory : {settings.knowledge_base_dir}")
    print()

    # ------------------------------------------------------------------
    # Step 1: Load all sources
    # ------------------------------------------------------------------
    print("Step 1/4  Loading sources...")
    t0 = time.time()
    docs = load_all_sources()

    if not docs:
        print(
            "\nNo documents found. Either:\n"
            "  • Run `python scripts/generate_synthetic_docs.py` first, or\n"
            "  • Drop .txt / .json files into data/knowledge_base/it_docs/ "
            "and data/knowledge_base/hr_docs/\n"
        )
        sys.exit(1)

    print(f"  Done in {time.time() - t0:.1f}s\n")

    # ------------------------------------------------------------------
    # Step 2: Chunk
    # ------------------------------------------------------------------
    print("Step 2/4  Chunking documents...")
    t0 = time.time()
    chunks = chunk_documents(docs)
    _check_counts(chunks)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    # ------------------------------------------------------------------
    # Step 3: Build vector index
    # ------------------------------------------------------------------
    print("Step 3/4  Building vector index (embedding + upsert to Qdrant)...")
    print("          This may take a while depending on your embedding provider.")
    t0 = time.time()
    store = get_vector_store()
    store.add_documents(chunks)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    # ------------------------------------------------------------------
    # Step 4: Persist
    # ------------------------------------------------------------------
    print("Step 4/4  Persisting index...")
    store.persist()
    print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    it_n = sum(1 for c in chunks if c["domain"] == "it")
    hr_n = sum(1 for c in chunks if c["domain"] == "hr")
    print("=" * 50)
    print("Knowledge base ready.")
    print(f"  IT chunks  : {it_n}")
    print(f"  HR chunks  : {hr_n}")
    print(f"  Total      : {len(chunks)}")
    print()
    print("Next step: python -m smartdesk.main")
    print("=" * 50)


if __name__ == "__main__":
    main()

"""Embedding helpers — dense (semantic) + sparse (keyword/hybrid).

Naming convention:
  make_*   → batch input, used at INSERT time (add_documents)
  query_*  → single input, used at QUERY time (similarity_search)

Dense provider is controlled by EMBEDDING_PROVIDER + EMBEDDING_MODEL in .env.
Supported: "openai" | "fastembed"

Sparse always uses fastembed (BM25/SPLADE/miniCOIL).
Model is controlled by SPARSE_EMBEDDING_MODEL (defaults to "Qdrant/bm25").

fastembed models download on first use (~50-200 MB). Subsequent calls use
the cached model — no re-download.
"""

from __future__ import annotations

from typing import Dict, List

from smartdesk.config import settings

# ---------------------------------------------------------------------------
# Module-level model cache — avoids reloading fastembed models on every call
# ---------------------------------------------------------------------------
_dense_model_cache: dict = {}
_sparse_model_cache: dict = {}


# ---------------------------------------------------------------------------
# Dense embeddings
# ---------------------------------------------------------------------------

def make_dense_embedding(texts: List[str]) -> List[List[float]]:
    """Dense-embed a batch of document chunks for indexing.

    Called by vector_store.add_documents() at insert time.
    Returns one float vector per text.
    """
    provider = settings.embedding_provider.lower()

    if provider == "openai":
        return _openai_embed(texts)
    elif provider == "fastembed":
        return _fastembed_dense_embed(texts)
    else:
        raise ValueError(
            f"Unsupported EMBEDDING_PROVIDER: {provider!r}. "
            "Set to 'openai' or 'fastembed' in .env."
        )


def query_dense_embedding(text: str) -> List[float]:
    """Dense-embed a single query string for retrieval.

    Called by vector_store.similarity_search() at query time.
    """
    return make_dense_embedding([text])[0]


# ---------------------------------------------------------------------------
# Sparse embeddings (always fastembed)
# ---------------------------------------------------------------------------

def make_sparse_embedding(texts: List[str]) -> List[Dict[str, list]]:
    """Sparse-embed a batch of document chunks for the keyword channel.

    Called by vector_store.add_documents() alongside make_dense_embedding.
    Return format: [{"indices": [...], "values": [...]}, ...] — one dict per
    text, matching qdrant_client.models.SparseVector(indices, values).
    """
    try:
        from fastembed import SparseTextEmbedding
    except ImportError:
        raise ImportError(
            "fastembed is required for sparse embeddings. "
            "Run: pip install fastembed"
        )

    model_name = settings.sparse_embedding_model or "Qdrant/bm25"
    if model_name not in _sparse_model_cache:
        print(f"[embeddings] Loading sparse model '{model_name}' (first-time download may take a moment)...")
        _sparse_model_cache[model_name] = SparseTextEmbedding(model_name=model_name)

    model = _sparse_model_cache[model_name]
    results = list(model.embed(texts))
    return [
        {"indices": r.indices.tolist(), "values": r.values.tolist()}
        for r in results
    ]


def query_sparse_embedding(text: str) -> Dict[str, list]:
    """Sparse-embed a single query string for the keyword channel at query time."""
    return make_sparse_embedding([text])[0]


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

def _openai_embed(texts: List[str]) -> List[List[float]]:
    """Embed texts using the OpenAI Embeddings API, batched in groups of 100."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")

    client = OpenAI(api_key=settings.llm_api_key)
    model = settings.embedding_model or "text-embedding-3-small"

    vectors: List[List[float]] = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(input=batch, model=model)
        vectors.extend(item.embedding for item in resp.data)

    return vectors


def _fastembed_dense_embed(texts: List[str]) -> List[List[float]]:
    """Embed texts using a local fastembed dense model (no API key needed)."""
    try:
        from fastembed import TextEmbedding
    except ImportError:
        raise ImportError("fastembed is required. Run: pip install fastembed")

    model_name = settings.embedding_model or "BAAI/bge-small-en-v1.5"
    if model_name not in _dense_model_cache:
        print(f"[embeddings] Loading dense model '{model_name}' (first-time download may take a moment)...")
        _dense_model_cache[model_name] = TextEmbedding(model_name=model_name)

    model = _dense_model_cache[model_name]
    return [e.tolist() for e in model.embed(texts)]

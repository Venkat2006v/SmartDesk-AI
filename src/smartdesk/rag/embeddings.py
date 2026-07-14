"""Embedding helpers — dense (semantic search) and sparse (the keyword
half of Qdrant's hybrid search; see docs/DESIGN_DECISIONS.md).

TODO: implement using whichever dense embedding provider you chose
(config.settings.embedding_provider / embedding_model), plus a sparse
model for hybrid search. fastembed bundles ready-to-use sparse models
(BM25, SPLADE++, miniCOIL) so you don't need to train anything:

    from fastembed import SparseTextEmbedding
    sparse_model = SparseTextEmbedding(
        model_name=settings.sparse_embedding_model or "Qdrant/bm25"
    )
    sparse_embeddings = list(sparse_model.embed(texts))  # indices + values per text

Keep dense/sparse functions batch-friendly so vector_store.py and
retriever.py don't need to know which provider/model is behind them.
"""

from __future__ import annotations

from typing import Dict, List


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of documents/chunks (dense) for indexing.

    TODO: implement (OpenAI/Cohere/local sentence-transformers/etc.)
    """
    raise NotImplementedError("TODO: implement document embedding")


def embed_query(text: str) -> List[float]:
    """Embed a single query string (dense) for retrieval.

    Some providers use different embedding modes/instructions for queries
    vs. documents — keep this separate from embed_texts even if today it
    just calls the same underlying API.

    TODO: implement.
    """
    raise NotImplementedError("TODO: implement query embedding")


def embed_sparse(texts: List[str]) -> List[Dict[str, list]]:
    """Compute sparse vectors for a batch of documents/chunks — the
    keyword half of Qdrant's hybrid search. Return shape should match
    whatever vector_store.py expects when building
    qdrant_client.models.SparseVector(indices=..., values=...), e.g.
    [{"indices": [...], "values": [...]}, ...].

    TODO: implement, e.g. via fastembed.SparseTextEmbedding (see module
    docstring).
    """
    raise NotImplementedError("TODO: implement sparse embedding for hybrid search")


def embed_sparse_query(text: str) -> Dict[str, list]:
    """Sparse-embed a single query string. See embed_sparse() for the
    expected return shape.

    TODO: implement.
    """
    raise NotImplementedError("TODO: implement sparse query embedding")

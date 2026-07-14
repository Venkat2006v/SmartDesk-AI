"""Vector store abstraction, backed by Qdrant (see docs/DESIGN_DECISIONS.md).

Qdrant was chosen specifically for native hybrid (sparse + dense) search:
it stores both vector types on the same point and lets you combine them
in one query via the Query API (prefetch with one, rerank/fuse with the
other using RRF/RSF/DBSF). Keeping this interface stable means the rest
of the codebase (retriever.py, the knowledge agents) never talks to
qdrant-client directly.

TODO: implement all methods using qdrant-client. Rough shape:

    from qdrant_client import QdrantClient, models

    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    # or, with no server at all:
    # client = QdrantClient(path="./data/processed/qdrant_local")

    client.create_collection(
        collection_name=settings.qdrant_collection_name,
        vectors_config={"dense": models.VectorParams(size=<dim>, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )

    client.upsert(
        collection_name=settings.qdrant_collection_name,
        points=[
            models.PointStruct(
                id=...,
                vector={
                    "dense": [...],
                    "sparse": models.SparseVector(indices=[...], values=[...]),
                },
                payload={"domain": "it", "text": ..., "source": ...},
            )
        ],
    )

    # hybrid query (prefetch with sparse, rerank with dense — or fuse explicitly):
    client.query_points(
        collection_name=settings.qdrant_collection_name,
        prefetch=[models.Prefetch(query=sparse_query_vector, using="sparse", limit=20)],
        query=dense_query_vector,
        using="dense",
        query_filter=(
            models.Filter(must=[models.FieldCondition(key="domain", match=models.MatchValue(value=domain))])
            if domain else None
        ),
        limit=top_k,
    )

See https://qdrant.tech/articles/hybrid-search/ for the RRF/RSF/DBSF
fusion options if you'd rather fuse explicitly instead of prefetch+rerank.
"""

from __future__ import annotations

from typing import List, Optional

from smartdesk.orchestrator.state import RetrievedChunk
from smartdesk.rag.ingestion import Document


class VectorStore:
    """Thin wrapper around the Qdrant client."""

    def add_documents(self, docs: List[Document]) -> None:
        """Embed (dense + sparse) and upsert a batch of chunked documents
        into the Qdrant collection, tagging each point's payload with its
        domain ("it" | "hr") so similarity_search can filter by it.

        TODO: implement (see module docstring for the upsert shape).
        """
        raise NotImplementedError("TODO: implement add_documents")

    def similarity_search(
        self, query: str, domain: Optional[str] = None, top_k: int = 4
    ) -> List[RetrievedChunk]:
        """Return the top_k most similar chunks to `query`, combining dense
        and sparse search via Qdrant's Query API. If `domain` is given
        ("it" | "hr"), filter to that domain via a payload filter.

        TODO: implement (see module docstring for the query_points shape).
        """
        raise NotImplementedError("TODO: implement similarity_search")

    def persist(self) -> None:
        """Likely a no-op: a Qdrant server persists on write, and local
        mode persists to vector_index_dir automatically. Only needed if
        you add an explicit snapshot/backup step.

        TODO: implement if you want explicit snapshotting
        (client.create_snapshot(...)), otherwise leave as a no-op.
        """
        raise NotImplementedError("TODO: implement or no-op persist")

    def load(self) -> None:
        """Likely a no-op too — Qdrant collections are addressed by
        name/connection, not explicitly "loaded" like a FAISS index file.

        TODO: implement if needed, otherwise leave as a no-op.
        """
        raise NotImplementedError("TODO: implement or no-op load")


def get_vector_store() -> VectorStore:
    """Factory: construct a VectorStore backed by Qdrant, using
    settings.qdrant_url / settings.qdrant_api_key / settings.qdrant_collection_name.

    TODO: implement — construct the QdrantClient, create the collection if
    it doesn't exist yet, and pass it into VectorStore.
    """
    raise NotImplementedError("TODO: implement get_vector_store factory")

"""Vector store implementation backed by Qdrant (local embedded mode).

Local mode: QdrantClient(path=settings.vector_index_dir)
  - No Docker or server needed
  - Data persists on disk at vector_index_dir (default: ./data/processed/vector_index)
  - Suitable for development and demos (supports up to ~1M vectors comfortably)

To switch to server/Docker mode in the future, just swap the client init inside
get_vector_store():
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)

Collection layout:
  Named vectors:
    "dense"  → float vector, COSINE distance (size determined at first add_documents call)
    "sparse" → SparseVector (BM25/SPLADE via fastembed)

  Payload per point:
    {"domain": "it"|"hr", "title": str, "text": str, "source": str}
"""

from __future__ import annotations

import os
import uuid
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from smartdesk.config import settings
from smartdesk.orchestrator.state import RetrievedChunk
from smartdesk.rag.embeddings import (
    make_dense_embedding,
    make_sparse_embedding,
    query_dense_embedding,
    query_sparse_embedding,
)
from smartdesk.rag.ingestion import Document

_DENSE_VECTOR_NAME = "dense"
_SPARSE_VECTOR_NAME = "sparse"
_UPSERT_BATCH_SIZE = 100  # max points per upsert call


class VectorStore:
    """Qdrant-backed vector store with hybrid dense+sparse retrieval.

    Instantiate via get_vector_store() — do not construct directly.
    """

    def __init__(self, client: QdrantClient, collection_name: str) -> None:
        self._client = client
        self._collection = collection_name

    # ------------------------------------------------------------------
    # Collection management (internal)
    # ------------------------------------------------------------------

    def _collection_exists(self) -> bool:
        existing = {c.name for c in self._client.get_collections().collections}
        return self._collection in existing

    def _create_collection(self, dense_dim: int) -> None:
        """Create the Qdrant collection with named dense + sparse vector configs."""
        print(
            f"[vector_store] Creating collection '{self._collection}' "
            f"(dense_dim={dense_dim})"
        )
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config={
                _DENSE_VECTOR_NAME: VectorParams(
                    size=dense_dim,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                _SPARSE_VECTOR_NAME: SparseVectorParams(),
            },
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_documents(self, docs: List[Document]) -> None:
        """Embed (dense + sparse) and upsert docs into Qdrant.

        - Batch-processes embeddings to stay within API/memory limits.
        - Creates the collection on first call (vector size inferred from embedding).
        - Gracefully indexes dense-only if sparse embeddings raise NotImplementedError.
        """
        if not docs:
            print("[vector_store] No documents to add — skipping.")
            return

        texts = [doc["text"] for doc in docs]

        print(f"[vector_store] Generating dense embeddings for {len(texts)} chunks...")
        dense_vecs = make_dense_embedding(texts)

        sparse_vecs_raw: List[Optional[dict]] = []
        try:
            print(f"[vector_store] Generating sparse embeddings for {len(texts)} chunks...")
            sparse_vecs_raw = make_sparse_embedding(texts)
        except NotImplementedError:
            print(
                "[vector_store] Sparse embeddings not implemented yet — "
                "indexing dense-only. Implement make_sparse_embedding() to enable hybrid search."
            )
            sparse_vecs_raw = [None] * len(texts)

        # Create collection on first use (dimension known after first embed)
        if not self._collection_exists():
            self._create_collection(dense_dim=len(dense_vecs[0]))

        # Build PointStructs and upsert in batches
        total = len(docs)
        for batch_start in range(0, total, _UPSERT_BATCH_SIZE):
            batch_end = min(batch_start + _UPSERT_BATCH_SIZE, total)
            print(
                f"[vector_store] Upserting points "
                f"{batch_start + 1}–{batch_end} / {total}..."
            )

            points: List[PointStruct] = []
            for idx in range(batch_start, batch_end):
                doc = docs[idx]

                # Named-vector dict — always has dense; sparse added if available
                vector: dict = {_DENSE_VECTOR_NAME: dense_vecs[idx]}

                sv = sparse_vecs_raw[idx]
                if sv is not None:
                    vector[_SPARSE_VECTOR_NAME] = SparseVector(
                        indices=sv["indices"],
                        values=sv["values"],
                    )

                # Qdrant accepts UUID strings as point IDs
                raw_id = doc.get("id", "")
                try:
                    point_id = str(uuid.UUID(str(raw_id)))
                except ValueError:
                    point_id = str(uuid.uuid4())

                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "domain": doc.get("domain", ""),
                            "title": doc.get("title", ""),
                            "text": doc["text"],
                            "source": doc.get("source", ""),
                        },
                    )
                )

            self._client.upsert(
                collection_name=self._collection,
                points=points,
            )

        print(
            f"[vector_store] Done — {total} points indexed "
            f"into collection '{self._collection}'."
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def similarity_search(
        self,
        query: str,
        domain: Optional[str] = None,
        top_k: int = 4,
    ) -> List[RetrievedChunk]:
        """Hybrid dense+sparse retrieval using Qdrant's Query API.

        Search strategy:
          1. Prefetch top candidates using the sparse (keyword) channel.
             Sparse is especially effective for IT acronyms: MFA, VPN, SSO, TOTP.
          2. Rerank/fuse the candidates using the dense (semantic) channel.
          Falls back to dense-only if sparse raises NotImplementedError.

        Args:
            query: Natural-language query string.
            domain: Optional "it" or "hr" to filter results to one domain.
            top_k: Number of RetrievedChunk results to return.
        """
        dense_vec = query_dense_embedding(query)

        # Build optional domain filter
        search_filter: Optional[Filter] = None
        if domain:
            search_filter = Filter(
                must=[FieldCondition(key="domain", match=MatchValue(value=domain))]
            )

        # Hybrid search (sparse prefetch → dense rerank)
        # Falls back to dense-only if sparse is not yet implemented
        try:
            sparse_vec = query_sparse_embedding(query)
            results = self._client.query_points(
                collection_name=self._collection,
                prefetch=[
                    Prefetch(
                        query=SparseVector(
                            indices=sparse_vec["indices"],
                            values=sparse_vec["values"],
                        ),
                        using=_SPARSE_VECTOR_NAME,
                        # Cast a wider net with sparse; dense will rerank
                        limit=top_k * 3,
                    )
                ],
                query=dense_vec,
                using=_DENSE_VECTOR_NAME,
                query_filter=search_filter,
                with_payload=True,
                limit=top_k,
            )
        except NotImplementedError:
            # Dense-only path — fully functional without sparse
            results = self._client.query_points(
                collection_name=self._collection,
                query=dense_vec,
                using=_DENSE_VECTOR_NAME,
                query_filter=search_filter,
                with_payload=True,
                limit=top_k,
            )

        chunks: List[RetrievedChunk] = []
        for point in results.points:
            payload = point.payload or {}
            chunks.append(
                RetrievedChunk(
                    text=payload.get("text", ""),
                    source=payload.get("source", ""),
                    score=float(point.score),
                )
            )
        return chunks

    # ------------------------------------------------------------------
    # Persistence (no-ops — local mode auto-persists)
    # ------------------------------------------------------------------

    def persist(self) -> None:
        """No-op: Qdrant local mode persists every write to disk automatically."""
        pass

    def load(self) -> None:
        """No-op: Qdrant local mode loads from disk when QdrantClient(path=...) is called."""
        pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_vector_store() -> VectorStore:
    """Create a VectorStore backed by Qdrant local (embedded) mode.

    Data is stored at settings.vector_index_dir.
    Default: ./data/processed/vector_index

    No server or Docker needed — Qdrant manages a local on-disk store.

    Switching to Docker/server mode later:
        Replace `QdrantClient(path=local_path)` with:
        `QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)`
    """
    local_path = settings.vector_index_dir
    os.makedirs(local_path, exist_ok=True)

    client = QdrantClient(path=local_path)
    print(
        f"[vector_store] Connected to Qdrant local store at: "
        f"{os.path.abspath(local_path)}"
    )

    return VectorStore(client=client, collection_name=settings.qdrant_collection_name)

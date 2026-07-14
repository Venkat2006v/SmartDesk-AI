"""Tests for rag/retriever.py — retrieve() and decide_escalation().

All tests run without API keys or disk I/O:
  - Qdrant ":memory:" client (ephemeral, no files written)
  - Patched embeddings return fixed 4-dim float vectors
  - Sparse embeddings intentionally raise NotImplementedError (validates
    the dense-only fallback path in vector_store)
"""

from __future__ import annotations

import uuid
from typing import Dict, List
from unittest.mock import patch

import pytest

from smartdesk.orchestrator.state import RetrievedChunk
from smartdesk.rag.ingestion import Document
from smartdesk.rag.vector_store import VectorStore

# ---------------------------------------------------------------------------
# Tiny fixed-dim embeddings — no model downloads, no API calls
# ---------------------------------------------------------------------------

_DIM = 4  # Small but enough to exercise COSINE distance


def _fixed_dense_batch(texts: List[str]) -> List[List[float]]:
    """Deterministic 4-dim vector per text (keyed on first char)."""
    out = []
    for t in texts:
        seed = ord(t[0]) if t else 65
        out.append([float(seed % 4), 1.0, 0.0, 0.0])
    return out


def _fixed_dense_query(text: str) -> List[float]:
    return _fixed_dense_batch([text])[0]


def _sparse_not_implemented(texts: List[str]) -> List[Dict[str, list]]:
    raise NotImplementedError("sparse disabled in tests")


def _sparse_query_not_implemented(text: str) -> Dict[str, list]:
    raise NotImplementedError("sparse disabled in tests")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mem_store() -> VectorStore:
    """Ephemeral in-memory Qdrant VectorStore — no disk writes."""
    from qdrant_client import QdrantClient
    client = QdrantClient(":memory:")
    return VectorStore(client=client, collection_name="test_kb")


_SAMPLE_DOCS: List[Document] = [
    Document(
        id=str(uuid.uuid4()),
        domain="it",
        title="VPN Setup",
        text="To connect to the corporate VPN, install GlobalProtect and use SSO credentials.",
        source="it_docs/vpn_setup.json",
    ),
    Document(
        id=str(uuid.uuid4()),
        domain="it",
        title="MFA Enrollment",
        text="Enable multi-factor authentication (MFA) via the IT portal using TOTP or a hardware key.",
        source="it_docs/mfa.json",
    ),
    Document(
        id=str(uuid.uuid4()),
        domain="hr",
        title="PTO Policy",
        text="Full-time employees accrue 15 days of paid time off per year, available immediately.",
        source="hr_docs/pto_policy.json",
    ),
    Document(
        id=str(uuid.uuid4()),
        domain="hr",
        title="Benefits Enrollment",
        text="Open enrollment for health benefits runs every November. Visit the HR portal to enroll.",
        source="hr_docs/benefits.json",
    ),
]


@pytest.fixture()
def populated_store(mem_store: VectorStore) -> VectorStore:
    """mem_store pre-loaded with sample IT + HR documents."""
    with (
        patch("smartdesk.rag.vector_store.make_dense_embedding", side_effect=_fixed_dense_batch),
        patch("smartdesk.rag.vector_store.make_sparse_embedding", side_effect=_sparse_not_implemented),
    ):
        mem_store.add_documents(list(_SAMPLE_DOCS))
    return mem_store


# ---------------------------------------------------------------------------
# retrieve() — tests via VectorStore.similarity_search() directly
# (retrieve() is a thin singleton wrapper around similarity_search;
#  we inject the store via retriever._store to avoid disk I/O)
# ---------------------------------------------------------------------------

class TestRetrieve:
    def _search(self, store: VectorStore, query: str, domain: str, top_k: int = 4) -> List[RetrievedChunk]:
        """Helper: run similarity_search with patched query embeddings."""
        with (
            patch("smartdesk.rag.vector_store.query_dense_embedding", side_effect=_fixed_dense_query),
            patch("smartdesk.rag.vector_store.query_sparse_embedding", side_effect=_sparse_query_not_implemented),
        ):
            return store.similarity_search(query=query, domain=domain, top_k=top_k)

    def test_returns_retrieved_chunk_dicts(self, populated_store: VectorStore) -> None:
        result = self._search(populated_store, "VPN setup", domain="it")
        assert isinstance(result, list)
        assert len(result) > 0
        for chunk in result:
            assert "text" in chunk and isinstance(chunk["text"], str)
            assert "source" in chunk and isinstance(chunk["source"], str)
            assert "score" in chunk and isinstance(chunk["score"], float)

    def test_domain_filter_returns_only_it_docs(self, populated_store: VectorStore) -> None:
        result = self._search(populated_store, "VPN MFA SSO", domain="it", top_k=10)
        for chunk in result:
            assert "it_docs" in chunk["source"], (
                f"Got non-IT source '{chunk['source']}' with domain='it' filter"
            )

    def test_domain_filter_returns_only_hr_docs(self, populated_store: VectorStore) -> None:
        result = self._search(populated_store, "benefits PTO", domain="hr", top_k=10)
        for chunk in result:
            assert "hr_docs" in chunk["source"], (
                f"Got non-HR source '{chunk['source']}' with domain='hr' filter"
            )

    def test_top_k_caps_result_count(self, populated_store: VectorStore) -> None:
        result = self._search(populated_store, "anything", domain="it", top_k=1)
        assert len(result) <= 1

    def test_empty_domain_returns_empty_list(self, populated_store: VectorStore) -> None:
        """Query an existing domain that has zero matches for the filter."""
        # "it" domain only has 2 docs; asking for a domain with none in the store
        # Use a non-existent domain string to force zero results
        with (
            patch("smartdesk.rag.vector_store.query_dense_embedding", side_effect=_fixed_dense_query),
            patch("smartdesk.rag.vector_store.query_sparse_embedding", side_effect=_sparse_query_not_implemented),
        ):
            result = populated_store.similarity_search(query="anything", domain="finance", top_k=4)
        assert result == []

    def test_retrieve_module_function_uses_injected_store(self, populated_store: VectorStore) -> None:
        """retrieve() delegates to the singleton store — inject it directly."""
        import smartdesk.rag.retriever as retriever_mod

        original_store = retriever_mod._store
        try:
            retriever_mod._store = populated_store
            with (
                patch("smartdesk.rag.vector_store.query_dense_embedding", side_effect=_fixed_dense_query),
                patch("smartdesk.rag.vector_store.query_sparse_embedding", side_effect=_sparse_query_not_implemented),
            ):
                result = retriever_mod.retrieve("How do I set up VPN?", domain="it", top_k=2)
        finally:
            retriever_mod._store = original_store  # restore for other tests

        assert isinstance(result, list)
        assert len(result) <= 2


# ---------------------------------------------------------------------------
# decide_escalation() tests
# ---------------------------------------------------------------------------

class TestDecideEscalation:
    def test_no_chunks_always_escalates(self) -> None:
        from smartdesk.rag.retriever import decide_escalation

        escalate, score = decide_escalation("What is Kubernetes?", retrieved=[])

        assert escalate is True
        assert score == 0.0

    def test_high_score_does_not_escalate_above_threshold(self) -> None:
        from smartdesk.rag.retriever import decide_escalation

        chunks: List[RetrievedChunk] = [
            RetrievedChunk(text="VPN setup guide", source="it_docs/vpn.json", score=0.9),
            RetrievedChunk(text="SSO details", source="it_docs/sso.json", score=0.7),
        ]
        escalate, score = decide_escalation("VPN setup", retrieved=chunks, confidence_threshold=0.5)

        assert escalate is False
        assert score == pytest.approx(0.9)

    def test_low_score_escalates_below_threshold(self) -> None:
        from smartdesk.rag.retriever import decide_escalation

        chunks: List[RetrievedChunk] = [
            RetrievedChunk(text="Something vaguely related", source="it_docs/misc.json", score=0.3),
        ]
        escalate, score = decide_escalation("AWS EC2 provisioning", retrieved=chunks, confidence_threshold=0.5)

        assert escalate is True
        assert score == pytest.approx(0.3)

    def test_zero_threshold_never_escalates_when_chunks_exist(self) -> None:
        """threshold=0.0 (default) → only escalate when zero chunks returned."""
        from smartdesk.rag.retriever import decide_escalation

        chunks: List[RetrievedChunk] = [
            RetrievedChunk(text="Some result", source="hr_docs/pto.json", score=0.1),
        ]
        escalate, _ = decide_escalation("PTO policy", retrieved=chunks, confidence_threshold=0.0)

        assert escalate is False

    def test_confidence_score_equals_top_chunk_score(self) -> None:
        from smartdesk.rag.retriever import decide_escalation

        chunks: List[RetrievedChunk] = [
            RetrievedChunk(text="Best match", source="it_docs/vpn.json", score=0.88),
            RetrievedChunk(text="Second match", source="it_docs/mfa.json", score=0.65),
        ]
        _, score = decide_escalation("VPN", retrieved=chunks)

        assert score == pytest.approx(0.88)

    def test_exact_threshold_boundary_escalates(self) -> None:
        """score == threshold should NOT escalate (threshold is a minimum floor)."""
        from smartdesk.rag.retriever import decide_escalation

        chunks: List[RetrievedChunk] = [
            RetrievedChunk(text="Exact match", source="it_docs/x.json", score=0.5),
        ]
        # score (0.5) is not < threshold (0.5) → should NOT escalate
        escalate, _ = decide_escalation("query", retrieved=chunks, confidence_threshold=0.5)
        assert escalate is False

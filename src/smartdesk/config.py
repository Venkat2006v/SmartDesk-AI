"""Centralized configuration, loaded from environment variables (.env).

Keeping all config in one place means the rest of the codebase never reads
os.environ directly — swap providers/backends by changing .env, not code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # Orchestration framework (LangGraph — see docs/DESIGN_DECISIONS.md).
    # Informational only; LangGraph itself needs no API key, just whichever
    # LLM provider is configured below.
    orchestration_framework: str = field(
        default_factory=lambda: os.getenv("ORCHESTRATION_FRAMEWORK", "langgraph")
    )

    # LLM
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai"))
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", ""))

    # Vector store (Qdrant — see docs/DESIGN_DECISIONS.md)
    vector_store_backend: str = field(
        default_factory=lambda: os.getenv("VECTOR_STORE_BACKEND", "qdrant")
    )
    vector_index_dir: str = field(
        default_factory=lambda: os.getenv("VECTOR_INDEX_DIR", "./data/processed/vector_index")
    )
    qdrant_url: str = field(
        default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333")
    )
    qdrant_api_key: str = field(default_factory=lambda: os.getenv("QDRANT_API_KEY", ""))
    qdrant_collection_name: str = field(
        default_factory=lambda: os.getenv("QDRANT_COLLECTION_NAME", "smartdesk_kb")
    )

    # Embeddings
    embedding_provider: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "openai")
    )
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", ""))
    sparse_embedding_model: str = field(
        default_factory=lambda: os.getenv("SPARSE_EMBEDDING_MODEL", "")
    )

    # Ticketing
    ticketing_provider: str = field(
        default_factory=lambda: os.getenv("TICKETING_PROVIDER", "mock")
    )
    ticketing_api_key: str = field(default_factory=lambda: os.getenv("TICKETING_API_KEY", ""))
    ticketing_base_url: str = field(default_factory=lambda: os.getenv("TICKETING_BASE_URL", ""))
    ticketing_project_key: str = field(
        default_factory=lambda: os.getenv("TICKETING_PROJECT_KEY", "")
    )

    # Retrieval / escalation
    retrieval_top_k: int = field(default_factory=lambda: int(os.getenv("RETRIEVAL_TOP_K", "4")))
    confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", "0.0"))
    )

    # Knowledge base
    knowledge_base_dir: str = field(
        default_factory=lambda: os.getenv("KNOWLEDGE_BASE_DIR", "./data/knowledge_base")
    )


settings = Settings()

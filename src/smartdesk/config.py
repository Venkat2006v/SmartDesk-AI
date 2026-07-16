"""Centralized configuration loaded from environment variables (.env)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # Orchestration (LangGraph — informational, no API key needed)
    orchestration_framework: str = field(
        default_factory=lambda: os.getenv("ORCHESTRATION_FRAMEWORK", "langgraph")
    )

    # LLM
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai"))
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", ""))

    # Vector store (Qdrant)
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
    # Jira-specific: the Atlassian account email paired with TICKETING_API_KEY
    jira_email: str = field(default_factory=lambda: os.getenv("JIRA_EMAIL", ""))

    # Retrieval / escalation
    retrieval_top_k: int = field(default_factory=lambda: int(os.getenv("RETRIEVAL_TOP_K", "4")))
    confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", "0.0"))
    )

    # Knowledge base
    knowledge_base_dir: str = field(
        default_factory=lambda: os.getenv("KNOWLEDGE_BASE_DIR", "./data/knowledge_base")
    )

    # HITL mode
    hitl_mode: str = field(default_factory=lambda: os.getenv("HITL_MODE", "cli"))

    # Verbose agent logging — set SMARTDESK_VERBOSE=false to suppress agent debug output
    verbose: bool = field(
        default_factory=lambda: os.getenv("SMARTDESK_VERBOSE", "true").lower() != "false"
    )

    # ── LangSmith observability ───────────────────────────────────────────────
    # LangGraph auto-traces every graph.invoke() when tracing is enabled.
    # call_llm() is additionally decorated with @traceable so each LLM call
    # appears as a named child span with token counts and latency.
    #
    # Setup: https://smith.langchain.com → create a project → copy API key
    langchain_tracing_v2: bool = field(
        default_factory=lambda: os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    )
    langchain_api_key: str = field(
        default_factory=lambda: os.getenv("LANGCHAIN_API_KEY", "")
    )
    langchain_project: str = field(
        default_factory=lambda: os.getenv("LANGCHAIN_PROJECT", "smartdesk-ai")
    )
    langchain_endpoint: str = field(
        default_factory=lambda: os.getenv(
            "LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"
        )
    )


settings = Settings()

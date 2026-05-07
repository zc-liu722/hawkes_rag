"""Hawkes-RAG: self-exciting memory for retrieval-augmented generation."""

from hawkes_rag.core import Event, HawkesParams, MultivariateHawkesProcess
from hawkes_rag.embeddings import HashingEmbedding, SentenceTransformerEmbedding, make_embedding_fn
from hawkes_rag.evaluation import heldout_predictive_log_likelihood, temporal_train_test_split
from hawkes_rag.gpu import resolve_torch_device
from hawkes_rag.locomo import (
    AtomicFact,
    ConversationMessage,
    EventizedConversation,
    EventizedCorpus,
    LoCoMoEventizer,
    LoCoMoQAPair,
    OfficialLoCoMoSample,
    load_official_locomo10_json,
    load_official_locomo10_samples,
)
from hawkes_rag.memory import HawkesMemoryStore, MemoryItem, RetrievalResult

__all__ = [
    "Event",
    "HawkesParams",
    "MultivariateHawkesProcess",
    "MemoryItem",
    "RetrievalResult",
    "HawkesMemoryStore",
    "ConversationMessage",
    "AtomicFact",
    "EventizedConversation",
    "EventizedCorpus",
    "HashingEmbedding",
    "SentenceTransformerEmbedding",
    "make_embedding_fn",
    "resolve_torch_device",
    "LoCoMoEventizer",
    "LoCoMoQAPair",
    "OfficialLoCoMoSample",
    "load_official_locomo10_json",
    "load_official_locomo10_samples",
    "temporal_train_test_split",
    "heldout_predictive_log_likelihood",
]

"""Hawkes-RAG: self-exciting memory for retrieval-augmented generation."""

from hawkes_rag.core import Event, HawkesParams, MultivariateHawkesProcess
from hawkes_rag.evaluation import heldout_predictive_log_likelihood, temporal_train_test_split
from hawkes_rag.locomo import (
    AtomicFact,
    ConversationMessage,
    EventizedConversation,
    EventizedCorpus,
    HashingEmbedding,
    LoCoMoEventizer,
    load_official_locomo10_json,
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
    "LoCoMoEventizer",
    "load_official_locomo10_json",
    "temporal_train_test_split",
    "heldout_predictive_log_likelihood",
]

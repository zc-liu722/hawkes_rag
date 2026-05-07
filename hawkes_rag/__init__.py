"""Hawkes-RAG: self-exciting memory for retrieval-augmented generation."""

from hawkes_rag.core import Event, HawkesParams, MultivariateHawkesProcess
from hawkes_rag.memory import HawkesMemoryStore, MemoryItem, RetrievalResult

__all__ = [
    "Event",
    "HawkesParams",
    "MultivariateHawkesProcess",
    "MemoryItem",
    "RetrievalResult",
    "HawkesMemoryStore",
]

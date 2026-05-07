from __future__ import annotations

from hawkes_rag import HawkesMemoryStore
from hawkes_rag.retrieval import diagonal_hawkes_retrieve, naive_retrieve


def test_store_retrieval_uses_intensity() -> None:
    store = HawkesMemoryStore(beta=0.2)
    max_fact = store.add("The user's dog is named Max.", [1.0, 0.0])
    store.add("The user mentioned Python once.", [0.8, 0.2])
    for t in [1.0, 2.0, 3.0]:
        store.record_access(max_fact.id, time=t)

    results = store.retrieve([0.85, 0.05], top_k=2, time=4.0, record_event=False)
    assert results[0].memory.id == max_fact.id
    assert results[0].intensity > results[1].intensity


def test_mention_event_has_weaker_but_real_effect() -> None:
    store = HawkesMemoryStore(beta=0.5)
    item = store.add("The user's dog is named Max.", [1.0, 0.0])
    before = store.intensities(1.0)[item.id]
    store.record_mentions([item.id], time=1.0)
    after = store.intensities(1.1)[item.id]
    assert after > before


def test_baseline_retrievers_do_not_record_events() -> None:
    store = HawkesMemoryStore()
    store.add("A", [1.0, 0.0])
    store.add("B", [0.0, 1.0])
    naive_retrieve(store, [1.0, 0.0])
    diagonal_hawkes_retrieve(store, [1.0, 0.0], time=1.0)
    assert store.events == []

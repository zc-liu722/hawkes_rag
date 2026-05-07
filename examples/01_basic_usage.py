from __future__ import annotations

import numpy as np

import _bootstrap  # noqa: F401
from hawkes_rag import HawkesMemoryStore


def main() -> None:
    store = HawkesMemoryStore(beta=0.4)
    store.add("The user's dog is named Max.", [1.0, 0.0, 0.0])
    store.add("The user takes Max to the park on Saturdays.", [0.9, 0.1, 0.0])
    store.add("The user once mentioned Python packaging.", [0.0, 1.0, 0.0])

    for t in [1, 2, 3, 4, 5]:
        store.record_access(0, time=float(t))

    query = np.array([0.85, 0.05, 0.0])
    results = store.retrieve(query, top_k=3, time=8.0, record_event=False)

    for result in results:
        print(
            f"{result.memory.id}: score={result.score:.3f} "
            f"sim={result.similarity:.3f} lambda={result.intensity:.3f} "
            f"text={result.memory.content}"
        )


if __name__ == "__main__":
    main()

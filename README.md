# hawkes-rag

Self-exciting memory for retrieval-augmented generation.

Current RAG memory is mostly stateless: a fact mentioned five times last week
and a fact mentioned once last month are often ranked only by semantic
similarity. Hawkes-RAG adds memory dynamics. Frequently activated memories
strengthen, unused memories decay, and related memories excite each other.

```text
lambda_i(t) = mu_i + sum_j alpha_ij sum_{events on j} w_e exp(-beta(t - t_e))
score_i(q, t) = cosine(q, e_i) * lambda_i(t)
```

## Why This Repo Exists

The claim is not just "add recency to RAG." The claim is that LLM agent memory
can be modeled as a multivariate Hawkes process:

- self-excitation: a memory gets stronger when it is used
- cross-excitation: related memories activate each other
- decay: unused memories naturally fade toward a baseline
- stability: the interaction matrix is projected to a bounded spectral radius
- estimation: `(alpha, beta, mu)` can be fit from event logs with MLE

The first validation loop is deliberately synthetic: generate event sequences
from known Hawkes parameters with Ogata thinning, recover the parameters with
MLE, then move to LoCoMo eventization.

## Quick Start

```bash
python3 examples/01_basic_usage.py
```

```python
from hawkes_rag import HawkesMemoryStore

store = HawkesMemoryStore(beta=0.4)
store.add("The user's dog is named Max.", [1.0, 0.0, 0.0])
store.add("The user takes Max to the park on Saturdays.", [0.9, 0.1, 0.0])
store.add("The user once mentioned Python packaging.", [0.0, 1.0, 0.0])

for t in [1, 2, 3, 4, 5]:
    store.record_access(0, time=float(t))

results = store.retrieve([0.85, 0.05, 0.0], top_k=3, time=8.0)
```

## MLE Recovery

```bash
python3 examples/02_synthetic_recovery.py
```

This fits a Hawkes model from unlabeled event sequences:

```text
(timestamp, memory_id)
```

No relevance labels are required.

## Mechanism Demo

```bash
python3 examples/04_mechanism_demo.py
```

The demo writes:

- `outputs/naive_vs_hawkes_scores.csv`
- `outputs/demo_alpha_heatmap.png`
- `outputs/demo_lambda_curve.png`

These are the raw materials for the README/demo GIF: naive RAG versus
Hawkes-RAG on the same 50-turn memory stream, with the alpha matrix and
lambda curve visible.

## LoCoMo Eventization Design

```bash
python3 examples/06_locomo_eventization.py
```

The benchmark path is:

1. Extract atomic facts from each message.
2. Emit a source event when a fact first appears.
3. Detect later references to previous facts in the same conversation.
4. Treat each conversation as one Hawkes trajectory.
5. Fit a pooled likelihood across trajectories with per-conversation active
   memory masks.
6. Evaluate held-out predictive log-likelihood on the tail of each trajectory.

The first local implementation uses sentence-level facts and deterministic
hashing embeddings so the pipeline runs without API keys. The research version
should replace those two pieces with a Mem0-style or LLM fact extractor plus
BGE/sentence-transformers embeddings.

## Roadmap

- real LoCoMo dataset schema pinning and artifact caching
- diagonal Hawkes versus low-rank Hawkes benchmark table
- sentence-transformers/BGE embedding example
- demo GIF generation script
- optional SQLite/vector-database backend

## License

MIT

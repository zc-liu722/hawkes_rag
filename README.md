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

## Baseline Evidence

```bash
python3 examples/cross_excitation_demo.py
```

The benchmark runs `naive_retrieve`, `diagonal_hawkes_retrieve`, and the full
alpha Hawkes retriever on the same synthetic cross-excitation corpus. The
association probes intentionally query with a cue memory while grading the
paired target memory, so the test isolates whether off-diagonal alpha matters.

| Retriever | Recall@1 | Recall@3 | MRR | Held-out PLL/event |
| --- | ---: | ---: | ---: | ---: |
| `naive_retrieve` | 0.000 | 1.000 | 0.500 | -1.626 |
| `diagonal_hawkes_retrieve` | 0.000 | 0.000 | 0.250 | -6.717 |
| `full_alpha_hawkes_retrieve` | 1.000 | 1.000 | 1.000 | -1.242 |

The full-alpha model is the only baseline that can promote the target memory
from a related cue event, and it also improves held-out predictive
log-likelihood on the same corpus.

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

The local implementation uses sentence-level facts and MiniLM embeddings
(`sentence-transformers/all-MiniLM-L6-v2`) by default, with deterministic
hashing embeddings kept as a zero-dependency fallback. The research version
should replace the sentence splitter with a Mem0-style or LLM fact extractor.

## Real LoCoMo Run

```bash
python3 benchmarks/locomo/download.py
python3 benchmarks/locomo/run_locomo.py
```

`download.py` fetches the official `snap-research/locomo` `locomo10.json`.
`run_locomo.py` pins the official schema, caches the full eventized corpus in
`outputs/locomo_eventized_<embedding>.json`, and writes the main result to
`outputs/locomo_results.{json,md}`. The default run uses all facts, MiniLM
embeddings, and low-rank MLE; pass `--embedding hashing --no-fit-mle --max-facts
80` for a fast repo benchmark.

Current default run:

| Model | Held-out PLL/event | Held-out PLL total | Held-out events |
| --- | ---: | ---: | ---: |
| `naive_zero_alpha` | -10.552 | -1603.920 | 152 |
| `diagonal_alpha` | -9.759 | -1483.442 | 152 |
| `full_alpha` | -9.574 | -1455.316 | 152 |

## Roadmap

- stronger fact extraction beyond the local sentence splitter
- demo GIF generation script
- optional SQLite/vector-database backend

## License

MIT

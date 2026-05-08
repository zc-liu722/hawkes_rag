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
(`sentence-transformers/all-MiniLM-L6-v2`) by default. Install the embeddings
extra for the default run; deterministic hashing embeddings are available only
when explicitly selected with `--embedding hashing`. LoCoMo benchmark runs
cache downloaded embedding models in `benchmarks/locomo/cache/models` by
default; override this with `--model-cache-dir` if you want a shared cache
elsewhere. The research version
should replace the sentence splitter with a Mem0-style or LLM fact extractor.

## Real LoCoMo Run

```bash
python3 -m pip install -e ".[embeddings]"
python3 benchmarks/locomo/download.py
python3 benchmarks/locomo/run_locomo.py
```

`download.py` fetches the official `snap-research/locomo` `locomo10.json`.
`run_locomo.py` pins the official schema, caches the full eventized corpus in
`outputs/locomo_eventized_<embedding>.json`, and writes the main result to
`outputs/locomo_results.{json,md}`. Retrieval evaluation uses LoCoMo's QA
labels: the question is the query and the annotated evidence messages are the
ground truth. Train QA evidence labels add supervised access events, while
held-out QA labels are used only for retrieval grading. The default run uses
all facts, MiniLM embeddings, and MLE with conversation-local fitting composed
into a sparse global alpha; pass `--embedding hashing --no-fit-mle --max-facts
80` for a fast repo benchmark.
For GPU acceleration, install the torch extra and use Adam MLE:

```bash
python3 -m pip install -e ".[embeddings,torch]"
python3 benchmarks/locomo/run_locomo.py --optimizer adam --device auto
```

`--device auto` selects CUDA first, then Apple MPS, then CPU. The same device
is used for local sentence-transformer embeddings, embedding similarity blocks,
top-k similarity priors, and PyTorch Hawkes optimization.

The LoCoMo benchmark is now retrieval-first. It compares pure semantic
retrieval (`cosine`), a simple temporal baseline (`cosine_recency`), a
self-excitation ablation (`diagonal_alpha`), and the full cross-excitation
model (`full_alpha`). The main success criterion is Recall@5 and MRR on held-out
QA evidence retrieval, with held-out PLL retained only as an event-model
diagnostic. The report also includes two lightweight mechanism slices:
`recurring_evidence` for repeated facts and `linked_evidence` for facts related
to other active memories.

Latest LoCoMo result in `outputs/locomo_results.{json,md}`:

- dataset: `benchmarks/locomo/cache/locomo10.json`
- eventized_cache: `outputs/locomo_eventized_minilm.json` (loaded)
- conversations: 10
- messages: 5882
- facts: 12048
- events: 164873
- embedding: `minilm`
- fit_mode: `low_rank_mle`
- fit_success: True

| Model | Held-out PLL/event | Held-out PLL total | Held-out events |
| --- | ---: | ---: | ---: |
| `naive_zero_alpha` | -5.674 | -428288.406 | 75480 |
| `diagonal_alpha` | -4.744 | -358043.438 | 75480 |
| `full_alpha` | 0.046 | 3476.184 | 75480 |

| Model | Recall@1 | Recall@5 | MRR | Retrieval queries |
| --- | ---: | ---: | ---: | ---: |
| `cosine` | 0.197 | 0.389 | 0.300 | 396 |
| `cosine_recency` | 0.093 | 0.187 | 0.152 | 396 |
| `diagonal_alpha` | 0.051 | 0.126 | 0.101 | 396 |
| `full_alpha` | 0.114 | 0.255 | 0.197 | 396 |

| Subset | Model | Recall@1 | Recall@5 | MRR | Queries |
| --- | --- | ---: | ---: | ---: | ---: |
| `recurring_evidence` | `cosine` | 0.209 | 0.398 | 0.311 | 354 |
| `linked_evidence` | `cosine` | 0.200 | 0.391 | 0.304 | 345 |
| `recurring_evidence` | `cosine_recency` | 0.102 | 0.192 | 0.157 | 354 |
| `linked_evidence` | `cosine_recency` | 0.096 | 0.177 | 0.151 | 345 |
| `recurring_evidence` | `diagonal_alpha` | 0.056 | 0.130 | 0.107 | 354 |
| `linked_evidence` | `diagonal_alpha` | 0.052 | 0.116 | 0.101 | 345 |
| `recurring_evidence` | `full_alpha` | 0.121 | 0.274 | 0.208 | 354 |
| `linked_evidence` | `full_alpha` | 0.116 | 0.261 | 0.201 | 345 |

Paired bootstrap CI:

- comparison: `full_alpha_minus_diagonal_alpha`
- mean_delta_nats_per_event: 4.753
- bootstrap_std: 0.105
- ci95: [4.557, 4.959]
- paired_trajectories: 10
- bootstrap_samples: 1000

## LoCoMo-Plus Run

```bash
python3 benchmarks/locomo/run_locomo_plus.py \
  --data benchmarks/locomo/cache/locomo_plus.json \
  --embedding minilm \
  --device auto
```

For GPU acceleration, use the same pattern as the main LoCoMo runner. The
Plus runner batch-encodes cue/dialogue/eventization text on the selected
sentence-transformer device, uses that device for retrieval similarity and
Hawkes intensity batches, and can run MLE with PyTorch Adam:

```bash
python3 benchmarks/locomo/run_locomo_plus.py \
  --data benchmarks/locomo/cache/locomo_plus.json \
  --embedding minilm \
  --fit-mle \
  --optimizer adam \
  --device auto
```

Latest LoCoMo-Plus result in `outputs/locomo_plus_results.{json,md}`:

- dataset: `benchmarks/locomo/cache/locomo_plus.json`
- probes: 401
- conversations: 401
- facts: 779
- events: 851
- embedding: `minilm`
- fit_mode: `stable_similarity_alpha`
- fusion_gamma: 0.2

| Model | Recall@1 | Recall@5 | MRR | Queries |
| --- | ---: | ---: | ---: | ---: |
| `cosine` | 1.000 | 1.000 | 1.000 | 401 |
| `cosine_recency` | 1.000 | 1.000 | 1.000 | 401 |
| `diagonal_alpha` | 1.000 | 1.000 | 1.000 | 401 |
| `full_alpha` | 1.000 | 1.000 | 1.000 | 401 |
| `zero_alpha` | 1.000 | 1.000 | 1.000 | 401 |

| Bucket | Model | Recall@1 | Recall@5 | MRR | Queries |
| --- | --- | ---: | ---: | ---: | ---: |
| `gap_unknown` | `cosine` | 1.000 | 1.000 | 1.000 | 401 |
| `gap_unknown` | `cosine_recency` | 1.000 | 1.000 | 1.000 | 401 |
| `gap_unknown` | `diagonal_alpha` | 1.000 | 1.000 | 1.000 | 401 |
| `gap_unknown` | `full_alpha` | 1.000 | 1.000 | 1.000 | 401 |
| `gap_unknown` | `zero_alpha` | 1.000 | 1.000 | 1.000 | 401 |

## Roadmap

- stronger fact extraction beyond the local sentence splitter
- demo GIF generation script
- optional SQLite/vector-database backend

## License

MIT

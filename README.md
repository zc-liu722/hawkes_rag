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
ground truth. The default run uses all facts, MiniLM embeddings, and MLE with
conversation-local fitting composed into a sparse global alpha; pass
`--embedding hashing --no-fit-mle --max-facts 80` for a fast repo benchmark.
For GPU acceleration, install the torch extra and use Adam MLE:

```bash
python3 -m pip install -e ".[embeddings,torch]"
python3 benchmarks/locomo/run_locomo.py --optimizer adam --device auto
```

`--device auto` selects CUDA first, then Apple MPS, then CPU. The same device
is used for local sentence-transformer embeddings, embedding similarity blocks,
top-k similarity priors, and PyTorch Hawkes optimization.

Current smoke run (`outputs/locomo_results.{json,md}`) used the official
LoCoMo10 file with deterministic hashing embeddings, `--no-fit-mle`, and
`--max-facts 80`. The full eventized cache contains 10 conversations, 5,882
messages, 12,048 facts, and 82,272 events; the smoke evaluation subset keeps 80
facts and 842 events.

| Model | Held-out PLL/event | Recall@1 | Recall@5 | MRR |
| --- | ---: | ---: | ---: | ---: |
| `naive_zero_alpha` | -5.883 | 0.690 | 0.995 | 0.828 |
| `diagonal_alpha` | -4.525 | 0.690 | 0.995 | 0.827 |
| `full_alpha` | -4.413 | 0.690 | 0.995 | 0.827 |

The smoke test is successful as a pipeline and modeling check: official data
loads, eventization writes a reusable cache, QA retrieval grading runs, and
full-alpha improves held-out predictive log-likelihood over diagonal alpha by
`+0.112` nats/event with a paired bootstrap 95% CI of `[0.057, 0.177]`.
Retrieval quality is effectively tied across the three models on this small
hashing subset, so this run does not yet demonstrate an end-to-end retrieval
gain. The next validation target is the default full-corpus MiniLM + MLE run.

## Roadmap

- stronger fact extraction beyond the local sentence splitter
- demo GIF generation script
- optional SQLite/vector-database backend

## License

MIT

# hawkes-rag

Self-exciting memory for retrieval-augmented generation.

Most RAG memory ranking is stateless: a fact mentioned repeatedly last week and
a fact mentioned once last month are often ranked mainly by semantic similarity.
Hawkes-RAG adds memory dynamics. Frequently activated memories strengthen,
unused memories decay, and related memories can excite each other.

```text
lambda_i(t) = mu_i + sum_j alpha_ij sum_{events on j} w_e exp(-beta(t - t_e))
score_i(q, t) = cosine(q, e_i) * lambda_i(t)
```

## Core Idea

Hawkes-RAG models agent memory as a multivariate Hawkes process:

- self-excitation: a memory becomes easier to retrieve after use
- cross-excitation: related memories can activate each other
- decay: unused memories fade toward a baseline
- stability: the interaction matrix is projected to a bounded spectral radius
- estimation: `(alpha, beta, mu)` can be fit from event logs with MLE

The project is intentionally evidence-first. It starts with controlled synthetic
settings where the right behavior is known, then moves to Chinese memory probes,
LoCoMo eventization, and LoCoMo-Plus retrieval sweeps.

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

## Demo Map

The examples are small by design; each one isolates one part of the mechanism.

| Demo | Command | Purpose | Main output |
| --- | --- | --- | --- |
| Basic usage | `python3 examples/01_basic_usage.py` | Add memories, record access, retrieve with temporal intensity. | Console ranking |
| Synthetic MLE | `python3 examples/02_synthetic_recovery.py` | Generate Hawkes events from known parameters and recover them with MLE. | Estimated `alpha`, `beta`, error |
| Decay visualization | `python3 examples/03_visualize_decay.py` | Show alpha structure and intensity decay. | `outputs/alpha_heatmap.png`, `outputs/lambda_curve.png` |
| Mechanism demo | `python3 examples/04_mechanism_demo.py` | Compare naive cosine vs Hawkes ranking on a 50-turn stream. | `outputs/naive_vs_hawkes_scores.csv`, demo plots |
| Chat memory | `python3 examples/05_chat_with_memory.py` | Minimal chat loop with retrieved memory context. | LLM response or fallback context |
| LoCoMo eventization | `python3 examples/06_locomo_eventization.py` | Convert dialogue into facts and Hawkes trajectories. | Fact/event counts and held-out PLL |
| Cross-excitation | `python3 examples/cross_excitation_demo.py` | Test whether off-diagonal `alpha` retrieves a paired target from a cue. | `outputs/baseline_comparison.md` |

## Key Results

### 1. Cross-excitation works in the controlled setting

The cross-excitation demo queries with a cue memory and grades the paired target
memory. This isolates whether off-diagonal `alpha` is doing useful work.

| Retriever | Recall@1 | Recall@3 | MRR | Held-out PLL/event |
| --- | ---: | ---: | ---: | ---: |
| `naive_retrieve` | 0.000 | 1.000 | 0.500 | -1.626 |
| `diagonal_hawkes_retrieve` | 0.000 | 0.000 | 0.250 | -6.717 |
| `full_alpha_hawkes_retrieve` | 1.000 | 1.000 | 1.000 | -1.242 |

Conclusion: when the task requires retrieving a linked memory from a related cue,
the full-alpha model succeeds and the diagonal ablation cannot.

### 2. LoCoMo improves event modeling, but retrieval needs careful fusion

```bash
python3 -m pip install -e ".[embeddings,torch]"
python3 benchmarks/locomo/download.py
python3 benchmarks/locomo/run_locomo.py --optimizer adam --device auto
```

The LoCoMo run eventizes 10 conversations into 12,048 facts and 164,873 events.
MLE is fit conversation-locally and evaluated on held-out event likelihood plus
held-out QA evidence retrieval.

| Model | Held-out PLL/event | Held-out events |
| --- | ---: | ---: |
| `naive_zero_alpha` | -5.674 | 75480 |
| `diagonal_alpha` | -4.744 | 75480 |
| `full_alpha` | 0.046 | 75480 |

| Model | Recall@1 | Recall@5 | MRR | Queries |
| --- | ---: | ---: | ---: | ---: |
| `cosine` | 0.197 | 0.389 | 0.300 | 396 |
| `cosine_recency` | 0.093 | 0.187 | 0.152 | 396 |
| `diagonal_alpha` | 0.051 | 0.126 | 0.101 | 396 |
| `full_alpha` | 0.114 | 0.255 | 0.197 | 396 |

Paired bootstrap on held-out PLL gives `full_alpha - diagonal_alpha = 4.753`
nats/event, 95% CI `[4.557, 4.959]`.

Conclusion: full-alpha strongly improves predictive event likelihood. At the
current default retrieval fusion, however, pure cosine remains the best LoCoMo
QA retriever, so the Hawkes signal should be treated as a temporal reranking
feature rather than a replacement for semantic retrieval.

### 3. LoCoMo-Plus supports light temporal reranking

```bash
python3 benchmarks/locomo/run_locomo_plus.py \
  --data benchmarks/locomo/cache/locomo_plus.json \
  --embedding minilm \
  --optimizer adam \
  --device auto
```

The LoCoMo-Plus sweep keeps the fitted Hawkes model fixed and varies
`fusion_gamma`, the retrieval weight assigned to temporal intensity.

| Gamma | Model | Recall@1 | Recall@5 | MRR | Queries |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | `cosine` | 0.115 | 0.289 | 0.215 | 401 |
| 0.01 | `diagonal_alpha` | 0.127 | 0.287 | 0.225 | 401 |
| 0.01 | `full_alpha` | 0.125 | 0.289 | 0.224 | 401 |
| 0.02 | `diagonal_alpha` | 0.125 | 0.304 | 0.223 | 401 |
| 0.02 | `full_alpha` | 0.122 | 0.304 | 0.224 | 401 |
| 0.2 | `full_alpha` | 0.030 | 0.060 | 0.065 | 401 |

Conclusion: a small Hawkes weight improves Recall@5 from 0.289 to 0.304 and MRR
from 0.215 to about 0.224-0.225. Larger weights over-dominate semantic
similarity and hurt ranking. This sweep does not yet show a clear full-alpha
retrieval advantage over the diagonal ablation.

## Current Takeaways

- Hawkes dynamics are useful when memory access has structure: repeated use,
  delayed references, and linked facts.
- Full cross-excitation is clearly valuable in controlled association tests.
- On real LoCoMo event streams, full-alpha is a much better event model than
  zero-alpha or diagonal-alpha.
- For QA retrieval, semantic similarity remains the anchor. The Hawkes signal is
  currently best used as a small reranking feature.
- The main open problem is stronger fact/event extraction and better calibration
  of `fusion_gamma` across datasets.

## Benchmark Notes

LoCoMo uses MiniLM embeddings by default and caches models under
`benchmarks/locomo/cache/models`. Use deterministic hashing only for fast smoke
runs:

```bash
python3 benchmarks/locomo/run_locomo.py --embedding hashing --no-fit-mle --max-facts 80
python3 benchmarks/locomo/run_locomo_plus.py --embedding hashing --max-probes 20 --no-fit-mle
```

For GPU acceleration:

```bash
python3 -m pip install -e ".[embeddings,torch]"
python3 benchmarks/locomo/run_locomo.py --optimizer adam --device auto
```

`--device auto` selects CUDA first, then Apple MPS, then CPU. The same device is
used for sentence-transformer embeddings, similarity blocks, top-k priors, and
PyTorch Hawkes optimization.

## Roadmap

- stronger fact extraction beyond the local sentence splitter
- better retrieval fusion calibration
- demo GIF generation script
- optional SQLite/vector-database backend

## License

MIT

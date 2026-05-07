# Benchmarks

## Baseline Ablation

```bash
python3 examples/cross_excitation_demo.py
```

This writes:

- `outputs/baseline_comparison.json`
- `outputs/baseline_comparison.md`

The ablation compares `naive_retrieve`, `diagonal_hawkes_retrieve`, and a
full-alpha Hawkes retriever on the same cross-excitation corpus with Recall@k,
MRR, and held-out predictive log-likelihood.

## LoCoMo

```bash
python3 -m pip install -e ".[embeddings]"
python3 benchmarks/locomo/download.py
python3 benchmarks/locomo/run_locomo.py
```

`download.py` downloads the official `snap-research/locomo` `locomo10.json`
into `benchmarks/locomo/cache/`. The cache directory is gitignored.

`run_locomo.py` uses the pinned official schema loader, caches the full
eventized corpus at `outputs/locomo_eventized_<embedding>.json`, and writes:

- `outputs/locomo_results.json`
- `outputs/locomo_results.md`

The default run uses all eventized facts, local MiniLM embeddings, LoCoMo QA
labels for retrieval grading, and MLE with conversation-local fitting composed
into a sparse global alpha. Use `--embedding hashing --no-fit-mle --max-facts
80` for a fast smoke run.

GPU acceleration is available through PyTorch:

```bash
python3 -m pip install -e ".[embeddings,torch]"
python3 benchmarks/locomo/run_locomo.py --optimizer adam --device auto
```

`--device auto` tries CUDA, then Apple MPS, then CPU. It is used for local
embeddings, similarity-prior construction, and Adam Hawkes MLE.

Latest smoke result in `outputs/locomo_results.{json,md}`:

| Model | Held-out PLL/event | Recall@1 | Recall@5 | MRR |
| --- | ---: | ---: | ---: | ---: |
| `naive_zero_alpha` | -5.883 | 0.690 | 0.995 | 0.828 |
| `diagonal_alpha` | -4.525 | 0.690 | 0.995 | 0.827 |
| `full_alpha` | -4.413 | 0.690 | 0.995 | 0.827 |

This is a successful smoke test for the LoCoMo pipeline and Hawkes likelihood
signal: full alpha beats diagonal alpha by `+0.112` nats/event, with paired
bootstrap 95% CI `[0.057, 0.177]`. It is not yet a final quality benchmark
because it uses hashing embeddings, skips MLE, caps evaluation at 80 facts, and
does not improve QA retrieval metrics over the baselines.

Planned follow-up:

- optional Mem0 comparison if dependency setup is smooth

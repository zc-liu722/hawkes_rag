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
python3 benchmarks/locomo/download.py
python3 benchmarks/locomo/run_locomo.py
```

`download.py` downloads the official `snap-research/locomo` `locomo10.json`
into `benchmarks/locomo/cache/`. The cache directory is gitignored.

`run_locomo.py` uses the pinned official schema loader, caches the full
eventized corpus at `outputs/locomo_eventized_<embedding>.json`, and writes:

- `outputs/locomo_results.json`
- `outputs/locomo_results.md`

The default run uses all eventized facts, local MiniLM embeddings, and low-rank
MLE. Use `--embedding hashing --no-fit-mle --max-facts 80` for a fast smoke run.

Planned follow-up:

- optional Mem0 comparison if dependency setup is smooth

# Benchmarks

## Baseline Ablation

```bash
python3 benchmarks/compare_baselines.py
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
eventized corpus at `outputs/locomo_eventized.json`, and writes:

- `outputs/locomo_results.json`
- `outputs/locomo_results.md`

The default run uses a balanced 80-fact subset across all 10 conversations so
the repository has a fast real-data smoke benchmark. Use `--max-facts 0` for
the full eventized corpus and `--fit-mle` to opt into low-rank MLE.

Planned follow-up:

- optional Mem0 comparison if dependency setup is smooth

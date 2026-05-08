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
into a sparse global alpha. Train QA evidence labels add supervised access
events, while held-out QA evidence labels are used only for retrieval grading.
Use `--embedding hashing --no-fit-mle --max-facts 80` for a fast smoke run.

MiniLM and BGE model files are cached under `benchmarks/locomo/cache/models`
by default, so the first run downloads them once and later runs reuse the local
copy. Use `--model-cache-dir /path/to/models` to place that reusable cache
somewhere else.

GPU acceleration is available through PyTorch:

```bash
python3 -m pip install -e ".[embeddings,torch]"
python3 benchmarks/locomo/run_locomo.py --optimizer adam --device auto
```

`--device auto` tries CUDA, then Apple MPS, then CPU. It is used for local
embeddings, similarity-prior construction, and Adam Hawkes MLE.

The LoCoMo retrieval table now compares the smallest mechanism-focused set of
retrievers:

| Retriever | Purpose |
| --- | --- |
| `cosine` | pure semantic baseline |
| `cosine_recency` | simple temporal decay baseline |
| `diagonal_alpha` | self-excitation ablation |
| `full_alpha` | self- plus cross-excitation model |

The main success criterion is no longer held-out PLL. A run is successful only
if `full_alpha` improves held-out QA evidence retrieval, especially Recall@5
and MRR, over `diagonal_alpha`, `cosine_recency`, and `cosine`. Held-out PLL is
kept as a diagnostic for the event model.

The report also includes two lightweight mechanism diagnostics:

- `recurring_evidence`: evidence that has multiple prior activations, testing
  whether self-excitation and decay help.
- `linked_evidence`: evidence related to other prior active facts, testing
  whether cross-memory excitation helps.

Planned follow-up:

- optional Mem0 comparison if dependency setup is smooth

## LoCoMo-Plus

LoCoMo-Plus adds a Cognitive category where an early cue dialogue must be used
to answer a later trigger query. This repository includes a lightweight
retrieval-first probe for that category:

```bash
python3 benchmarks/locomo/run_locomo_plus.py \
  --data benchmarks/locomo/cache/locomo_plus.json \
  --embedding minilm \
  --device auto
```

If `benchmarks/locomo/cache/locomo_plus.json` is missing, the script downloads
the official JSON from `xjtuleeyf/Locomo-Plus` automatically. Use
`--force-download` to refresh the cache, or `--locomo-plus-url` to point at a
server-local mirror. Raw LoCoMo-Plus records contain only cue/query metadata,
so the runner also loads `benchmarks/locomo/cache/locomo10.json` and samples
LoCoMo dialogue turns as distractor memory candidates. Use
`--max-context-messages` to control how many distractors are included per
probe.

The script accepts either the raw LoCoMo-Plus shape (`cue_dialogue`,
`trigger_query`, `time_gap`) or the unified-input shape (`input_prompt`,
`trigger`, `evidence`, `category`). It eventizes each cue/context, asks whether
the trigger retrieves the cue/evidence facts, and writes:

- `outputs/locomo_plus_results.json`
- `outputs/locomo_plus_results.md`

Use `--embedding hashing --max-probes 20` for a fast smoke run and
`--embedding minilm` or `--embedding bge` for a real semantic retrieval run.
The Plus runner follows the main LoCoMo GPU path: `--device auto` selects CUDA,
then Apple MPS, then CPU for sentence-transformer eventization batches,
retrieval similarity, and Hawkes intensity scoring. Add `--fit-mle --optimizer
adam` to run low-rank Hawkes MLE with PyTorch on the selected device. The
default is intentionally retrieval-only; it does not call an LLM judge for
final answer correctness.

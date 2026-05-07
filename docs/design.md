# Design Notes

Hawkes-RAG treats an atomic memory fact as a mark in a multivariate point
process. Each activation event is a tuple `(timestamp, memory_id, weight)`.

The first public prototype uses two event sources:

- retrieval/access event: weight `1.0`
- mention event: weight `0.3`

The intensity for memory `i` is:

```text
lambda_i(t) = mu_i + sum_j alpha_ij sum_{events on j} w_e exp(-beta (t - t_e))
```

The MLE objective is the standard Hawkes point-process likelihood:

```text
log L = sum_k log lambda_{i_k}(t_k) - sum_i int_0^T lambda_i(s) ds
```

No relevance labels are required. The required data is only an event log.

## First Benchmark Path

1. Generate synthetic trajectories with Ogata thinning from known parameters.
2. Fit parameters and verify recovery.
3. Build LoCoMo eventization: extract atomic facts, detect future mentions, and
   pool trajectories by conversation.
4. Evaluate held-out predictive log-likelihood and retrieval accuracy.

## Ablations

- naive RAG: cosine similarity only
- diagonal Hawkes: self-excitation without cross-memory interaction
- low-rank Hawkes-RAG: full interaction through constrained alpha

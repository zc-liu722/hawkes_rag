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

## LoCoMo Eventization

The eventization unit is an atomic fact, not a raw message chunk. Each
conversation produces:

- a local list of extracted facts
- source events at first mention
- mention events for later references
- a horizon equal to the conversation duration
- an active-memory mask containing only facts in that conversation

The active-memory mask matters. If the corpus has a global catalog of facts, a
single conversation should not pay the point-process integral for facts that
were never observable in that conversation. The likelihood is therefore pooled
over conversations while integrating only each trajectory's active dimensions.

The zero-dependency prototype uses `SentenceFactExtractor`,
`SemanticReferenceDetector`, and `HashingEmbedding`. For the paper path, swap in
an LLM/Mem0-style atomic fact extractor and a local BGE/sentence-transformers
embedding model.

Held-out evaluation uses temporal tail likelihood: fit on the prefix of every
conversation and score the conditional log-likelihood of the held-out tail,
teacher-forcing previous held-out events as they arrive.

## Ablations

- naive RAG: cosine similarity only
- diagonal Hawkes: self-excitation without cross-memory interaction
- low-rank Hawkes-RAG: full interaction through constrained alpha

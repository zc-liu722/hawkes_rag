# Theory Draft

Hawkes-RAG models long-term agent memory as a multivariate Hawkes process.
Each memory item is a process dimension, and each retrieval or mention is an
activation event.

For exponential kernels:

```text
lambda_i(t) = mu_i + sum_j int_0^t alpha_ij exp(-beta(t-s)) dN_j(s)
```

The interaction matrix encodes both self-excitation and associative memory.
`alpha_ii` strengthens a memory after direct access. `alpha_ij` strengthens
memory `i` after related memory `j` fires.

The retrieval score is:

```text
score_i(q, t) = cosine(q, e_i) * lambda_i(t)
```

The stability condition for the branching process is enforced in code with a
spectral-radius projection on `alpha`.

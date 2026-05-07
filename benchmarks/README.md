# Benchmarks

Planned benchmark work:

- synthetic Hawkes recovery
- diagonal versus low-rank Hawkes ablation
- LoCoMo eventization and held-out predictive log-likelihood
- optional Mem0 comparison if dependency setup is smooth

The LoCoMo implementation path is in `hawkes_rag.locomo` and
`examples/06_locomo_eventization.py`. The current adapter is deliberately
permissive about JSON field names, but the research run should pin the exact
dataset schema and save the eventized corpus artifact for reproducibility.

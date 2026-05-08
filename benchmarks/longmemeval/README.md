# LongMemEval Cross-Evidence Probe

Download LongMemEval-S:

```bash
python3 benchmarks/longmemeval/download.py
```

Run the real semantic analysis:

```bash
python3 -m pip install -e ".[embeddings]"
python3 benchmarks/longmemeval/analyze_cross_evidence.py --embedding minilm --device auto
```

For a dependency-light smoke run only:

```bash
python3 benchmarks/longmemeval/analyze_cross_evidence.py --embedding hashing --max-records 20
```

The script writes:

- `outputs/longmemeval_cross_evidence_analysis.json`
- `outputs/longmemeval_cross_evidence_analysis.md`

The go/no-go rule is:

- `go` when at least 30 multi-session questions satisfy the sweet-spot rule.
- `stop` when fewer than 10 questions satisfy it.
- `borderline` otherwise.

Sweet spot means at least one evidence session has query cosine `< 0.4` and
that evidence session has cosine `> 0.6` with another evidence session in the
same question.

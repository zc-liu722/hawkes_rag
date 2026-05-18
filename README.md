# hawkes_rag

Hawkes-style memory dynamics plus retrieval / reranking helpers for agent memory experiments. Core packages:

- **`hawkes_agent`** — λ-dynamics configuration, vector store, recall middleware, rerank routing, LangGraph-facing primitives.
- **`hawkes_rag`** — embeddings utilities used with the agent stack.

Benchmark scripts, datasets, and experiment outputs are intentionally **not** included in this repository (run them from a full checkout if you maintain one locally).

## Setup

Python 3.11+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-agent-harness.txt
```

Optional: configure a local Qwen reranker path via `hawkes_agent.config` / env as in your deployment (heavy model weights live under `models/` locally and stay gitignored).

## Layout

```
hawkes_agent/     # Memory store, dynamics, recall, LLM adapters
hawkes_rag/       # Embedding helpers
scripts/          # Small tooling (e.g. sweep summaries)
requirements-agent-harness.txt
```

## License

Use and distribution terms follow whatever license you attach for your project (add a `LICENSE` file if publishing publicly).

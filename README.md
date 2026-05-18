# hawkes_rag

A research-oriented **long-horizon agent memory** stack that combines **dense retrieval (RAG)** with **Hawkes-inspired λ (intensity) dynamics** in one pipeline. Memories carry not only similarity to the query but also a **learned salience** that grows or decays with **time and feedback**.

This public repository **intentionally omits** benchmarks, datasets, and bulky experiment artifacts; if you need full eval harnesses, keep that tree locally alongside this checkout.

---

## Design rationale (intuition)

- **Plain vector RAG**: high-cosine snippets always dominate ranking, which makes it awkward to express **recent reinforcement, forgetting, or type-dependent stickiness**.
- **This project’s compromise**: each memory holds a **strength λ ∈ [0,1]**. The implementation stores it as `lambda_plus`; at read time an **exponential decay** since `t_last_event` yields **λ⁻** (effective intensity). Retrieval blends cosine similarity with λ through a **batch-level μ**: when λ values look **flat** (“everyone looks hot”), the scorer leans harder on similarity; when a **few memories stand out**, the blend favors **λ-driven Hawkes-style modulation**.
- **Hot / Cold**: memories above a λ⁻ cutoff use the **hot** path (λ-shaped scoring); the rest use the **cold** path (lexical–semantic fallback) so snippets that faded in λ but **still lexical-match** are not permanently invisible.
- **Optional cross-encoder rerank**: hot candidates are **reranked**; if confidence is low or the hot score distribution is too flat, the system triggers **cold hybrid retrieval** (**BM25 + dense**, blended with `alpha`).

Entry points: `hawkes_agent.memory.InMemoryVectorStore`, `hawkes_agent.dynamics`, `hawkes_agent.recall.RecallMiddleware`.

---

## Memory records: `MemoryRecord`

Each entry is addressable text + an embedding + type + namespace:

| Field (concept) | Role |
|-----------------|------|
| `embedding` | L2-normalized dense vector |
| **`lambda_plus` (write-side intensity)** | Upper bound after the last **reinforce / suppress** update |
| **`t_last_event`** | Clock of the last λ update (simulation time `now`) |
| `type_class` | e.g. **`volatile` / `stable` / `identity`**, each with its own decay rate **β** (`DynamicsConfig.beta_by_type`) |
| `namespace` | Logical pool (session / user / shard) |

On insert: `lambda_plus = 1.0`, `t_last_event = t_created = now`.

---

## Temporal decay: λ⁺ → λ⁻ (effective intensity at read time)

For each record at query time `now`:

$$
\lambda^- = \min\left(1,\ \max\left(0,\ \lambda^+ \cdot e^{-\beta \cdot \Delta t}\right)\right),\quad \Delta t = now - t_{\text{last\_event}}
$$

- **Larger β ⇒ faster forgetting** (`volatile` defaults ≫ `stable` / `identity`).
- λ⁻ gates **hot membership** (vs. `DynamicsConfig.hot_lambda_threshold`) and **λ-aware retrieval scores**.

Implementation: `hawkes_agent.dynamics.decayed_lambda`, batched via `InMemoryVectorStore.decayed_lambdas`.

---

## Retrieval scoring: adaptive μ + Hawkes-style blend

Given cosine similarities `cos` and decayed intensities λ⁻, compute a batch **μ ∈ [μ_base, 1]** with `compute_mu`:

- Normalize **λ²** into a probability mass, compute normalized entropy **ĥ** — flatter λ mass ⇒ larger **ĥ**;
- $$
  \mu = \mu_{\text{base}} + (1-\mu_{\text{base}})\sqrt{1-\hat{h}}
  $$
  **Intuition**: when many memories look similarly “hot,” trust geometry more; when mass concentrates on few items, amplify intensity modulation.

Per-candidate score:

$$
\text{score}_i = \cos_i \cdot \bigl(\mu + (1-\mu)\cdot \lambda^-_i\bigr)
$$

- Larger λ⁻ wins more mass under **`(1-μ)`**.
- `cos_i` is clamped below by **`cosine_floor`**.
- The **`use_lambda=False`** branch (cold coarse ranking) **uses cosine only**; in code this collapses to **μ = 1**.

Implementation: `hawkes_agent.dynamics.recall_scores` via `_recall_from_records`. Each `RetrievedSegment` carries **`hawkes_score`** (equals the Hawkes-style term when λ is enabled) for logging and downstream **adoption / suppression**.

---

## Hot / Cold dual path

1. **Split by λ⁻ vs. threshold**:
   - **hot**: $\lambda^- \geq$ `hot_lambda_threshold`
   - **cold**: otherwise
2. **`recall_hot_cold`**:
   - hot: λ-shaped scoring (`use_lambda=True`)
   - cold: cosine coarse rank (`use_lambda=False`)
   - merge by **id**, keep the better score, then sort/truncate.
3. **`recall_hot_cold_reranked` (full path)**:
   - coarse **hot** pool up to **`intermediate_top_k`**;
   - optional **`Reranker`** batched rerank on hot snippets → **`rerank_score`** (e.g. local Qwen cross-encoder);
   - if rerank / Hawkes / **score entropy** look unreliable, **cold** fires: **BM25 + cosine** each min-max normalized, blended with **`DynamicsConfig.alpha`**;
   - merged budget stays **`intermediate_top_k`**; the implementation allocates roughly **~¼ hot + ~¾ cold** when cold activates.
   - **Typical cold triggers** (surfaced in metadata): `low_confidence` (top-1 rerank / Hawkes below `tau_r` / `tau_h`), `flat_hot_distribution` (rerank-score entropy ≥ `hot_entropy_threshold`), `insufficient_hot_coverage` (too few items above cutoff vs. `min_hot_injected`).

---

## Cross-encoder reranking (optional)

`RecallMiddleware` wires a **`Reranker`** via `AgentHarnessConfig.reranker_backend` / `reranker_model`.

- **`rerank_top_k`**: only cross-encode the top-K hot items (0 = no cap).
- Rerankers may consume **priors** from Hawkes scores or cold candidate scores — see `_apply_rerank`.

Default model path hints live in `hawkes_agent.config`; large weights stay under **`models/`** (gitignored).

---

## Write path: adoption, reinforcement, contradiction suppression

`RecallMiddleware` closes the loop:

1. **`write_turn`**: `add_memory` with λ⁺ = 1, `t_last_event = now`.
2. **`score_adoption`**: `adoption_method` chooses
   - `embedding`: cosine between `answer` and each snippet, or
   - `token_overlap`: overlap ratio;
   threshold **`theta_a`** yields adopted memory ids.
3. **`reinforce`** for adopted items:
   $$
   \lambda^+_{\text{new}} = \lambda^- + (1-\lambda^-)\cdot s
   $$
   where **`s`** is the retrieval-stage Hawkes score (or equivalent) — `reinforce_lambda`.
4. **`suppress`** for contradicted ids (upstream supplies the list), using cosine-to-query as **`contradiction_similarity`**:
   $$
   \lambda^+_{\text{new}} = \lambda^- \cdot (1 - \texttt{clamp}(\text{contradiction\_similarity}))
   $$
   (`suppress_lambda`).  
   **`prescreen_contradiction_signal`** flags high-cosine but **non-adopted** neighbors for cheap contradiction checks (`theta_c`, `theta_a`, `contradiction_top_k`).

Together: **retrieve → generate → (optional contradiction handling) → reinforce / suppress**, producing an observable λ trajectory.

---

## Configuration: key `DynamicsConfig` knobs

| Field | Role |
|-------|------|
| `beta_by_type` | Per-type exponential decay rates |
| `mu_base` | Lower bound on μ |
| `hot_lambda_threshold` | Hot vs. cold splitter |
| `intermediate_top_k` / `final_top_k` / `hot_top_k` / `cold_top_k` | Candidate budgets |
| `alpha` | Dense vs. BM25 weight in hybrid cold paths |
| `tau`, `tau_r`, `tau_h`, `hot_entropy_threshold`, `min_hot_injected` | Cutoffs and cold triggers |

See **`hawkes_agent/config.py`** for the full frozen dataclass.

---

## `hawkes_rag` package

Helpers around **sentence-transformers** plus **SQLite embedding caches** so sweeps reuse vectors cheaply. Default cache paths historically pointed at benchmark directories; when using this **minimal checkout without `benchmarks/`**, **pass explicit `cache_dir`** at call sites rather than relying on ignored layout.

---

## Setup and repository layout

**Python 3.11+** recommended.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-agent-harness.txt
```

```
hawkes_agent/           # λ dynamics, vector store, RecallMiddleware, rerank hooks
hawkes_rag/             # embeddings + caches
scripts/                # utilities (e.g. sweep summaries)
requirements-agent-harness.txt
```

Core deps include **LangGraph, LiteLLM, Pydantic AI, sentence-transformers** — see `requirements-agent-harness.txt`.

---

## License

Add a **`LICENSE`** file before publishing broadly; replace this note accordingly.

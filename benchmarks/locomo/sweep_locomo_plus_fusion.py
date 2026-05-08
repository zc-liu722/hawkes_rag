from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.locomo.run_locomo import (  # noqa: E402
    configure_huggingface_defaults,
    diagonal_only,
    estimate_mu,
    stable_similarity_params,
)
from benchmarks.locomo.run_locomo_plus import (  # noqa: E402
    LOCOMO10_URL,
    LOCOMO_PLUS_URL,
    CachedEmbeddingFn,
    ensure_json_data,
    eventize_probes,
    load_locomo_plus_probes,
    needs_locomo_context,
    retrieval_metrics,
)
from hawkes_rag.core import HawkesParams  # noqa: E402
from hawkes_rag.embeddings import make_embedding_fn  # noqa: E402
from hawkes_rag.estimation import LowRankHawkesEstimator  # noqa: E402


OUTPUT_JSON = "locomo_plus_fusion_sweep.json"
OUTPUT_MD = "locomo_plus_fusion_sweep.md"


def _log(message: str) -> None:
    print(f"[locomo-plus-sweep] {message}", flush=True)


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(
        description=(
            "Run LoCoMo-Plus once, then sweep fusion_gamma without refitting "
            "or re-eventizing for each value."
        )
    )
    parser.add_argument("--data", type=Path, default=Path("benchmarks/locomo/cache/locomo_plus.json"))
    parser.add_argument("--locomo-data", type=Path, default=Path("benchmarks/locomo/cache/locomo10.json"))
    parser.add_argument("--locomo-plus-url", default=LOCOMO_PLUS_URL)
    parser.add_argument("--locomo-url", default=LOCOMO10_URL)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--embedding", choices=["hashing", "minilm", "bge"], default="minilm")
    parser.add_argument("--model-cache-dir", type=Path, default=Path("benchmarks/locomo/cache/models"))
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-probes", type=int, default=0, help="First N probes; 0 means all.")
    parser.add_argument("--max-context-messages", type=int, default=120)
    parser.add_argument("--context-seed", type=int, default=0)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--optimizer", choices=["lbfgsb", "adam"], default="adam")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--dense-threshold", type=int, default=4000)
    parser.add_argument(
        "--fit-mle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fit low-rank Hawkes MLE once before sweeping; --no-fit-mle uses the stable fallback.",
    )
    parser.add_argument(
        "--gammas",
        default="0,0.01,0.02,0.05,0.1,0.2,0.4",
        help="Comma-separated fusion_gamma values to evaluate.",
    )
    parser.add_argument("--probe-delay-days", type=float, default=0.0)
    args = parser.parse_args()

    gammas = parse_gammas(args.gammas)
    args.outputs_dir.mkdir(parents=True, exist_ok=True)

    ensure_json_data(args.data, args.locomo_plus_url, "LoCoMo-Plus", force=args.force_download)
    plus_payload = json.loads(args.data.read_text())
    locomo_payload = None
    if needs_locomo_context(plus_payload):
        ensure_json_data(args.locomo_data, args.locomo_url, "LoCoMo10", force=False)
        locomo_payload = json.loads(args.locomo_data.read_text())

    if args.embedding != "hashing":
        args.model_cache_dir.mkdir(parents=True, exist_ok=True)
        configure_huggingface_defaults(args.model_cache_dir)
        _log(f"Using embedding model cache: {args.model_cache_dir}")
        _log(f"Using Hugging Face endpoint: {os.environ['HF_ENDPOINT']}")
        _log(f"Using Hugging Face home: {os.environ['HF_HOME']}")

    _log(f"Loading LoCoMo-Plus data: {args.data}")
    probes = load_locomo_plus_probes(
        args.data,
        locomo_payload=locomo_payload,
        max_context_messages=args.max_context_messages,
        context_seed=args.context_seed,
    )
    if args.max_probes > 0:
        probes = probes[: args.max_probes]
    if not probes:
        raise SystemExit("No LoCoMo-Plus cognitive probes found in the input file.")
    _log(f"Loaded probes={len(probes)}")

    _log(f"Initializing embedding backend: {args.embedding}")
    embedding_fn = make_embedding_fn(
        args.embedding,
        device=args.device,
        batch_size=args.embedding_batch_size,
        cache_dir=args.model_cache_dir if args.embedding != "hashing" else None,
    )
    embedding_fn = CachedEmbeddingFn(
        embedding_fn,
        batch_size=args.embedding_batch_size,
        log_prefix="LoCoMo-Plus sweep embeddings",
    )

    _log(f"Eventizing probes on device={args.device or 'auto'}")
    corpus = eventize_probes(probes, embedding_fn)
    embeddings = corpus.embeddings()
    _log(
        f"Eventized conversations={len(corpus.conversations)} "
        f"facts={corpus.n_memories} events={sum(len(c.events) for c in corpus.conversations)}"
    )

    if args.fit_mle:
        _log(
            f"Fitting Hawkes MLE once rank={args.rank} max_iter={args.max_iter} "
            f"optimizer={args.optimizer} device={args.device or 'auto'}"
        )
        estimator = LowRankHawkesEstimator.from_embeddings(
            embeddings,
            rank=args.rank,
            seed=0,
            learn_beta=True,
            dense_threshold=args.dense_threshold,
            optimizer=args.optimizer,
            learning_rate=args.learning_rate,
            device=args.device,
        )
        fit = estimator.fit(
            corpus.trajectories(),
            corpus.horizons(),
            active_memory_ids=corpus.active_memory_ids(),
            max_iter=args.max_iter,
        )
        params = fit.params
        fit_payload = {
            "mode": "low_rank_mle",
            "success": fit.success,
            "objective": fit.objective,
            "message": fit.message,
            "n_iter": fit.n_iter,
            "rank": args.rank,
            "beta": fit.params.beta,
            "optimizer": args.optimizer,
            "device": args.device or "auto",
        }
    else:
        _log("Building stable similarity-alpha Hawkes parameters once")
        params = stable_similarity_params(corpus, embeddings, dense_threshold=args.dense_threshold)
        fit_payload = {
            "mode": "stable_similarity_alpha",
            "success": True,
            "objective": None,
            "message": "MLE skipped via --no-fit-mle",
            "n_iter": 0,
            "rank": None,
            "beta": params.beta,
            "optimizer": None,
            "device": args.device or "auto",
        }

    models = {
        "cosine": None,
        "cosine_recency": params,
        "diagonal_alpha": diagonal_only(params),
        "full_alpha": params,
        "zero_alpha": HawkesParams(
            mu=estimate_mu(corpus.trajectories(), corpus.horizons(), corpus.n_memories),
            alpha=sparse.csr_matrix((corpus.n_memories, corpus.n_memories), dtype=float),
            beta=params.beta,
        ),
    }

    sweep = []
    for gamma in gammas:
        _log(f"Scoring retrieval for fusion_gamma={gamma:g}")
        retrieval = retrieval_metrics(
            probes,
            corpus,
            embedding_fn,
            models,
            fusion_gamma=gamma,
            probe_delay_days=args.probe_delay_days,
            device=args.device,
        )
        sweep.append({"fusion_gamma": gamma, "retrieval": retrieval})

    result = {
        "dataset": str(args.data),
        "locomo_context_dataset": str(args.locomo_data) if locomo_payload is not None else None,
        "n_probes": len(probes),
        "n_conversations": len(corpus.conversations),
        "n_facts": corpus.n_memories,
        "n_events": sum(len(conversation.events) for conversation in corpus.conversations),
        "embedding": args.embedding,
        "embedding_batch_size": args.embedding_batch_size,
        "probe_delay_days": args.probe_delay_days,
        "max_context_messages": args.max_context_messages,
        "dense_threshold": args.dense_threshold,
        "gammas": gammas,
        "fit": fit_payload,
        "sweep": sweep,
    }
    (args.outputs_dir / OUTPUT_JSON).write_text(json.dumps(result, indent=2) + "\n")
    markdown = format_markdown(result)
    (args.outputs_dir / OUTPUT_MD).write_text(markdown + "\n")
    _log(f"Run complete in {time.perf_counter() - started:.1f}s")
    print(markdown)


def parse_gammas(raw: str) -> list[float]:
    gammas = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            gammas.append(float(item))
    if not gammas:
        raise ValueError("--gammas must contain at least one float")
    return gammas


def format_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# LoCoMo-Plus Fusion Gamma Sweep",
        "",
        f"- dataset: `{result['dataset']}`",
        f"- locomo_context_dataset: `{result['locomo_context_dataset']}`",
        f"- probes: {result['n_probes']}",
        f"- conversations: {result['n_conversations']}",
        f"- facts: {result['n_facts']}",
        f"- events: {result['n_events']}",
        f"- embedding: `{result['embedding']}`",
        f"- embedding_batch_size: {result['embedding_batch_size']}",
        f"- device: `{result['fit']['device']}`",
        f"- fit_mode: `{result['fit']['mode']}`",
        f"- fit_optimizer: `{result['fit']['optimizer']}`",
        f"- max_context_messages: {result['max_context_messages']}",
        "",
        "| Gamma | Model | Recall@1 | Recall@5 | MRR | Queries |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    best_rows: dict[str, dict[str, Any]] = {}
    for sweep_item in result["sweep"]:
        gamma = sweep_item["fusion_gamma"]
        for row in sweep_item["retrieval"]["overall"]:
            row_with_gamma = {**row, "fusion_gamma": gamma}
            current = best_rows.get(row["model"])
            if current is None or row["mrr"] > current["mrr"]:
                best_rows[row["model"]] = row_with_gamma
            lines.append(
                f"| {gamma:g} | `{row['model']}` | {row['recall_at_1']:.3f} | "
                f"{row['recall_at_5']:.3f} | {row['mrr']:.3f} | {row['queries']} |"
            )
    lines.extend(
        [
            "",
            "## Best Gamma By Model",
            "",
            "| Model | Best gamma | Recall@1 | Recall@5 | MRR |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for model, row in sorted(best_rows.items()):
        lines.append(
            f"| `{model}` | {row['fusion_gamma']:g} | {row['recall_at_1']:.3f} | "
            f"{row['recall_at_5']:.3f} | {row['mrr']:.3f} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()

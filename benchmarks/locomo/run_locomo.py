from __future__ import annotations

import argparse
import hashlib
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

from hawkes_rag.core import Event, HawkesParams, MultivariateHawkesProcess, diagonal_only
from hawkes_rag.embeddings import EmbeddingFn, make_embedding_fn
from hawkes_rag.estimation import LowRankHawkesEstimator
from hawkes_rag.evaluation import heldout_predictive_log_likelihood
from hawkes_rag.gpu import adaptive_cuda_chunk_size, best_float_dtype, resolve_torch_device
from hawkes_rag.memory import _query_cosine
from hawkes_rag.locomo import (
    AtomicFact,
    ConversationMessage,
    EventizedConversation,
    EventizedCorpus,
    LoCoMoEventizer,
)
try:
    from hawkes_rag.locomo import LoCoMoQAPair, load_official_locomo10_samples
except ImportError:
    LoCoMoQAPair = Any
    from hawkes_rag.locomo import load_official_locomo10_json

    def load_official_locomo10_samples(path: Path):
        return load_official_locomo10_json(path)

from hawkes_rag.estimation import topk_similarity_prior


EVENTIZED_SCHEMA = "hawkes_rag.eventized_locomo.v3"


def _log(message: str) -> None:
    print(f"[locomo] {message}", flush=True)


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw_value = os.environ.get(name, str(default))
    try:
        value = int(raw_value)
    except ValueError:
        value = default
    return max(minimum, value)


def configure_huggingface_defaults(model_cache_dir: Path) -> None:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HOME", str(model_cache_dir / "hf_home"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(model_cache_dir / "sentence_transformers"))


def _corpus_summary(corpus: EventizedCorpus) -> str:
    n_messages = sum(len(conversation.messages) for conversation in corpus.conversations)
    n_events = sum(len(conversation.events) for conversation in corpus.conversations)
    return (
        f"conversations={len(corpus.conversations)} messages={n_messages} "
        f"facts={corpus.n_memories} events={n_events}"
    )


def _conversation_qa_pairs(conversation: EventizedConversation) -> list[Any]:
    return getattr(conversation, "qa_pairs", None) or []


def _make_eventized_conversation(
    *,
    conversation_id: str,
    messages: list[ConversationMessage],
    facts: list[AtomicFact],
    events: list[Event],
    horizon: float,
    qa_pairs: list[Any] | None,
) -> EventizedConversation:
    kwargs = {
        "conversation_id": conversation_id,
        "messages": messages,
        "facts": facts,
        "events": events,
        "horizon": horizon,
    }
    if "qa_pairs" in getattr(EventizedConversation, "__dataclass_fields__", {}):
        kwargs["qa_pairs"] = qa_pairs
    return EventizedConversation(**kwargs)


def _qa_question(qa_pair: Any) -> str:
    return qa_pair["question"] if isinstance(qa_pair, dict) else qa_pair.question


def _qa_evidence_message_ids(qa_pair: Any) -> list[str]:
    if isinstance(qa_pair, dict):
        return [str(value) for value in qa_pair.get("evidence_message_ids", [])]
    return qa_pair.evidence_message_ids


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description="Run Hawkes-RAG on the official LoCoMo corpus.")
    parser.add_argument("--data", type=Path, default=Path("benchmarks/locomo/cache/locomo10.json"))
    parser.add_argument("--eventized-cache", type=Path, default=None)
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument(
        "--optimizer",
        choices=["lbfgsb", "adam"],
        default="lbfgsb",
        help="Low-rank MLE optimizer. Use adam to run the fit with PyTorch on GPU when available.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device for embeddings/similarity/Adam MLE: auto, cuda, mps, or cpu.",
    )
    parser.add_argument(
        "--dense-threshold",
        type=int,
        default=4000,
        help="Above this fact count, MLE fits conversation-local dense blocks and composes sparse alpha.",
    )
    parser.add_argument(
        "--fit-mle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run low-rank MLE; use --no-fit-mle for the stable similarity-alpha fallback.",
    )
    parser.add_argument(
        "--embedding",
        choices=["hashing", "minilm", "bge"],
        default="minilm",
        help="Embedding backend for eventization and retrieval queries.",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=Path("benchmarks/locomo/cache/models"),
        help="Directory for downloaded sentence-transformers models, reused across runs.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=16,
        help="Sentence-transformers encode batch size. Default is one GPU-memory step below 32.",
    )
    parser.add_argument(
        "--max-facts",
        type=int,
        default=0,
        help="Fit/evaluate the first N eventized facts; 0 means full corpus.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument(
        "--qa-probe-delay-days",
        type=float,
        default=7.0,
        help="Evaluate QA retrieval after the latest evidence time plus this delay.",
    )
    parser.add_argument(
        "--qa-train-fraction",
        type=float,
        default=0.8,
        help="Per-conversation QA fraction used only to add supervised evidence access events.",
    )
    parser.add_argument(
        "--qa-split-seed",
        type=int,
        default=0,
        help="Seed for deterministic per-conversation QA train/test splitting.",
    )
    parser.add_argument(
        "--evidence-event-weight",
        type=float,
        default=1.0,
        help="Weight for access events added from train QA evidence labels.",
    )
    parser.add_argument(
        "--fusion-gamma",
        type=float,
        default=0.2,
        help="Weight for z-scored log temporal intensity in retrieval scoring.",
    )
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()
    if args.eventized_cache is None:
        args.eventized_cache = Path(f"outputs/locomo_eventized_{args.embedding}.json")

    _log(f"Starting LoCoMo run with embedding={args.embedding}")
    _log(f"Checking data file: {args.data}")
    if not args.data.exists():
        raise SystemExit(
            f"Missing LoCoMo data at {args.data}. Run: python3 benchmarks/locomo/download.py"
        )
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    args.eventized_cache.parent.mkdir(parents=True, exist_ok=True)

    if args.embedding != "hashing":
        args.model_cache_dir.mkdir(parents=True, exist_ok=True)
        configure_huggingface_defaults(args.model_cache_dir)
        _log(f"Using embedding model cache: {args.model_cache_dir}")
        _log(f"Using Hugging Face endpoint: {os.environ['HF_ENDPOINT']}")
        _log(f"Using Hugging Face home: {os.environ['HF_HOME']}")
    _log("Initializing embedding backend")
    embedding_fn = make_embedding_fn(
        args.embedding,
        device=args.device,
        batch_size=args.embedding_batch_size,
        cache_dir=args.model_cache_dir if args.embedding != "hashing" else None,
    )

    if args.eventized_cache.exists() and not args.refresh_cache:
        _log(f"Loading eventized cache: {args.eventized_cache}")
        try:
            corpus = read_eventized_corpus(args.eventized_cache)
            cache_status = "loaded"
            _log(f"Loaded eventized cache ({_corpus_summary(corpus)})")
        except ValueError:
            _log("Eventized cache schema is stale; loading source data and rebuilding cache")
            samples = load_official_locomo10_samples(args.data)
            _log(f"Eventizing {len(samples)} conversations")
            corpus = LoCoMoEventizer(embedding_fn=embedding_fn).eventize(samples)
            write_eventized_corpus(corpus, args.eventized_cache)
            cache_status = "refreshed_schema"
            _log(f"Refreshed eventized cache ({_corpus_summary(corpus)})")
    else:
        _log(f"Loading LoCoMo source data: {args.data}")
        samples = load_official_locomo10_samples(args.data)
        _log(f"Eventizing {len(samples)} conversations")
        corpus = LoCoMoEventizer(embedding_fn=embedding_fn).eventize(samples)
        _log(f"Writing eventized cache: {args.eventized_cache}")
        write_eventized_corpus(corpus, args.eventized_cache)
        cache_status = "written"
        _log(f"Wrote eventized cache ({_corpus_summary(corpus)})")

    full_corpus = corpus
    if args.max_facts > 0:
        _log(f"Limiting corpus to max_facts={args.max_facts}")
        corpus = limit_corpus_facts(corpus, args.max_facts)
        _log(f"Limited corpus ready ({_corpus_summary(corpus)})")

    _log("Adding train-QA evidence access events for supervised retrieval calibration")
    corpus = add_train_qa_evidence_events(
        corpus,
        qa_train_fraction=args.qa_train_fraction,
        qa_split_seed=args.qa_split_seed,
        probe_delay_days=args.qa_probe_delay_days,
        event_weight=args.evidence_event_weight,
    )
    _log(f"QA-calibrated corpus ready ({_corpus_summary(corpus)})")

    _log("Preparing embedding matrix")
    embeddings = corpus.embeddings()
    _log(f"Embedding matrix shape={embeddings.shape}")
    if args.fit_mle:
        _log(
            f"Fitting low-rank Hawkes MLE rank={args.rank} max_iter={args.max_iter} "
            f"dense_threshold={args.dense_threshold} optimizer={args.optimizer} device={args.device or 'auto'}"
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
        _log(
            f"MLE finished success={fit.success} n_iter={fit.n_iter} "
            f"objective={fit.objective:.6f}"
        )
        full_params = fit.params
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
        _log("Skipping MLE; building stable similarity-alpha fallback")
        full_params = stable_similarity_params(corpus, embeddings, dense_threshold=args.dense_threshold)
        _log("Stable similarity-alpha parameters ready")
        fit_payload = {
            "mode": "stable_similarity_alpha",
            "success": True,
            "objective": None,
            "message": "MLE skipped via --no-fit-mle",
            "n_iter": 0,
            "rank": None,
            "beta": full_params.beta,
        }

    _log("Building train/test splits")
    splits = corpus.heldout_splits(args.train_fraction)
    _log(f"Created {len(splits)} heldout splits with train_fraction={args.train_fraction}")
    _log("Evaluating heldout PLL for full_alpha")
    full_pll = heldout_predictive_log_likelihood(
        full_params,
        splits,
        device=args.device,
        label="full_alpha_pll",
    )
    _log(
        f"full_alpha heldout PLL/event={full_pll.per_event:.6f} "
        f"events={full_pll.n_events}"
    )
    _log("Evaluating heldout PLL for diagonal_alpha")
    diagonal_params = diagonal_only(full_params)
    diagonal_pll = heldout_predictive_log_likelihood(
        diagonal_params,
        splits,
        device=args.device,
        label="diagonal_alpha_pll",
    )
    _log(
        f"diagonal_alpha heldout PLL/event={diagonal_pll.per_event:.6f} "
        f"events={diagonal_pll.n_events}"
    )
    _log("Evaluating heldout PLL for naive_zero_alpha")
    naive_params = HawkesParams(
        mu=estimate_mu(corpus.trajectories(), corpus.horizons(), corpus.n_memories),
        alpha=zero_alpha_like(full_params.alpha),
        beta=full_params.beta,
    )
    naive_pll = heldout_predictive_log_likelihood(
        naive_params,
        splits,
        device=args.device,
        label="naive_zero_alpha_pll",
    )
    _log(
        f"naive_zero_alpha heldout PLL/event={naive_pll.per_event:.6f} "
        f"events={naive_pll.n_events}"
    )
    _log("Evaluating QA retrieval metrics")
    retrieval = retrieval_metrics(
        corpus,
        embedding_fn,
        {
            "cosine": None,
            "cosine_recency": full_params,
            "diagonal_alpha": diagonal_params,
            "full_alpha": full_params,
        },
        qa_train_fraction=args.qa_train_fraction,
        qa_split_seed=args.qa_split_seed,
        probe_delay_days=args.qa_probe_delay_days,
        fusion_gamma=args.fusion_gamma,
        device=args.device,
    )
    _log("Running paired bootstrap delta CI")
    bootstrap = paired_bootstrap_delta_ci(
        full_params,
        diagonal_params,
        splits,
        samples=args.bootstrap_samples,
        seed=0,
        device=args.device,
    )

    result = {
        "dataset": str(args.data),
        "eventized_cache": str(args.eventized_cache),
        "cache_status": cache_status,
        "n_conversations": len(corpus.conversations),
        "n_conversations_full_cache": len(full_corpus.conversations),
        "n_messages": sum(len(conversation.messages) for conversation in corpus.conversations),
        "n_messages_full_cache": sum(
            len(conversation.messages) for conversation in full_corpus.conversations
        ),
        "n_facts": corpus.n_memories,
        "n_facts_full_cache": full_corpus.n_memories,
        "max_facts": args.max_facts,
        "embedding": args.embedding,
        "qa_probe_delay_days": args.qa_probe_delay_days,
        "qa_train_fraction": args.qa_train_fraction,
        "qa_split_seed": args.qa_split_seed,
        "evidence_event_weight": args.evidence_event_weight,
        "fusion_gamma": args.fusion_gamma,
        "dense_threshold": args.dense_threshold,
        "n_events": sum(len(conversation.events) for conversation in corpus.conversations),
        "n_events_full_cache": sum(
            len(conversation.events) for conversation in full_corpus.conversations
        ),
        "fit": fit_payload,
        "heldout": [
            pll_row("naive_zero_alpha", naive_pll),
            pll_row("diagonal_alpha", diagonal_pll),
            pll_row("full_alpha", full_pll),
        ],
        "retrieval": retrieval,
        "paired_bootstrap": bootstrap,
    }
    _log(f"Writing JSON results: {args.outputs_dir / 'locomo_results.json'}")
    (args.outputs_dir / "locomo_results.json").write_text(json.dumps(result, indent=2) + "\n")
    markdown = format_markdown(result)
    _log(f"Writing Markdown results: {args.outputs_dir / 'locomo_results.md'}")
    (args.outputs_dir / "locomo_results.md").write_text(markdown + "\n")
    _log(f"Run complete in {time.perf_counter() - started:.1f}s")
    print(markdown)


def estimate_mu(trajectories: list[list[Event]], horizons: list[float], n_memories: int) -> np.ndarray:
    counts = np.full(n_memories, 0.5, dtype=float)
    for events in trajectories:
        for event in events:
            counts[event.memory_id] += event.weight
    return np.maximum(counts / max(sum(horizons), 1e-6), 1e-5)


def pll_row(name: str, pll) -> dict[str, Any]:
    return {
        "model": name,
        "heldout_pll_per_event": pll.per_event,
        "heldout_pll_total": pll.total,
        "heldout_events": pll.n_events,
        "trajectories": pll.n_trajectories,
    }


def qa_train_test_split(
    qa_pairs: list[Any],
    train_fraction: float,
    *,
    seed: int | None = None,
    key: str = "",
) -> tuple[list[Any], list[Any]]:
    if not (0.0 <= train_fraction < 1.0):
        raise ValueError("qa_train_fraction must be in [0, 1)")
    if len(qa_pairs) <= 1:
        return [], qa_pairs
    ordered_pairs = list(qa_pairs)
    if seed is not None:
        digest = hashlib.sha1(f"{seed}:{key}".encode("utf-8")).hexdigest()
        rng = np.random.default_rng(int(digest[:16], 16))
        order = rng.permutation(len(ordered_pairs))
        ordered_pairs = [ordered_pairs[int(index)] for index in order]
    train_count = int(len(qa_pairs) * train_fraction)
    train_count = min(max(train_count, 1), len(qa_pairs) - 1)
    return ordered_pairs[:train_count], ordered_pairs[train_count:]


def facts_by_message_id(facts: list[AtomicFact]) -> dict[str, list[AtomicFact]]:
    grouped: dict[str, list[AtomicFact]] = {}
    for fact in facts:
        grouped.setdefault(fact.source_message_id, []).append(fact)
    return grouped


def evidence_facts_for_qa(
    qa_pair: Any,
    grouped_facts: dict[str, list[AtomicFact]],
) -> list[AtomicFact]:
    return [
        fact
        for evidence_id in _qa_evidence_message_ids(qa_pair)
        for fact in grouped_facts.get(evidence_id, [])
    ]


def add_train_qa_evidence_events(
    corpus: EventizedCorpus,
    *,
    qa_train_fraction: float,
    probe_delay_days: float,
    event_weight: float,
    qa_split_seed: int | None = None,
) -> EventizedCorpus:
    conversations = []
    for conversation in corpus.conversations:
        grouped = facts_by_message_id(conversation.facts)
        train_qas, _ = qa_train_test_split(
            _conversation_qa_pairs(conversation),
            qa_train_fraction,
            seed=qa_split_seed,
            key=conversation.conversation_id,
        )
        added_events = []
        for qa_pair in train_qas:
            evidence_facts = evidence_facts_for_qa(qa_pair, grouped)
            if not evidence_facts:
                continue
            access_time = max(fact.source_time for fact in evidence_facts) + probe_delay_days
            for fact in evidence_facts:
                added_events.append(Event(time=access_time, memory_id=fact.id, weight=event_weight))
        events = sorted([*conversation.events, *added_events], key=lambda event: event.time)
        horizon = max(
            [conversation.horizon, *[event.time + 1e-4 for event in added_events]],
            default=conversation.horizon,
        )
        conversations.append(
            _make_eventized_conversation(
                conversation_id=conversation.conversation_id,
                messages=conversation.messages,
                facts=conversation.facts,
                events=events,
                horizon=float(horizon),
                qa_pairs=_conversation_qa_pairs(conversation),
            )
        )
    return EventizedCorpus(conversations=conversations, facts=corpus.facts)


def _combine_semantic_and_temporal(
    similarities: np.ndarray,
    intensities: np.ndarray,
    *,
    gamma: float,
) -> np.ndarray:
    if similarities.size == 0:
        return similarities
    log_intensity = np.log(np.maximum(intensities, 1e-12))
    std = float(np.std(log_intensity))
    if std <= 1e-12:
        return similarities.copy()
    temporal = (log_intensity - float(np.mean(log_intensity))) / std
    return similarities + gamma * temporal


def _metric_row(model: str, subset: str, hits: dict[int, int], reciprocal_ranks: list[float], n_queries: int) -> dict[str, Any]:
    return {
        "model": model,
        "subset": subset,
        "queries": n_queries,
        "recall_at_1": hits[1] / max(n_queries, 1),
        "recall_at_5": hits[5] / max(n_queries, 1),
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
    }


def diagnostic_subsets(
    evidence_facts: list[AtomicFact],
    facts: list[AtomicFact],
    events: list[Event],
    query_time: float,
    fact_embeddings: np.ndarray,
    fact_ids: np.ndarray,
) -> set[str]:
    evidence_ids = {fact.id for fact in evidence_facts}
    past_events = [event for event in events if event.time < query_time]
    counts: dict[int, int] = {}
    for event in past_events:
        counts[event.memory_id] = counts.get(event.memory_id, 0) + 1

    subsets = set()
    if any(counts.get(fact.id, 0) >= 2 for fact in evidence_facts):
        subsets.add("recurring_evidence")

    active_related_ids = {
        event.memory_id
        for event in past_events
        if event.memory_id not in evidence_ids and counts.get(event.memory_id, 0) >= 1
    }
    if active_related_ids and facts:
        id_to_index = {int(fact_id): index for index, fact_id in enumerate(fact_ids)}
        active_indices = [
            id_to_index[memory_id]
            for memory_id in active_related_ids
            if memory_id in id_to_index
        ]
        evidence_indices = [
            id_to_index[fact.id]
            for fact in evidence_facts
            if fact.id in id_to_index
        ]
        if active_indices and evidence_indices:
            active_embeddings = fact_embeddings[active_indices]
            for evidence_index in evidence_indices:
                sims = _query_cosine(fact_embeddings[evidence_index], active_embeddings, device="cpu")
                if bool(np.any(sims >= 0.55)):
                    subsets.add("linked_evidence")
                    break
    return subsets


def retrieval_metrics(
    corpus: EventizedCorpus,
    embedding_fn: EmbeddingFn,
    params_by_model: dict[str, HawkesParams | None],
    *,
    qa_train_fraction: float,
    qa_split_seed: int | None,
    probe_delay_days: float,
    fusion_gamma: float,
    device: str | None,
    top_ks: tuple[int, ...] = (1, 5),
) -> dict[str, list[dict[str, Any]]]:
    rows = []
    diagnostics = []
    for model, params in params_by_model.items():
        model_started = time.perf_counter()
        _log(f"Retrieval model={model} started")
        hits = {k: 0 for k in top_ks}
        reciprocal_ranks = []
        subset_hits = {
            "recurring_evidence": {k: 0 for k in top_ks},
            "linked_evidence": {k: 0 for k in top_ks},
        }
        subset_rr = {"recurring_evidence": [], "linked_evidence": []}
        subset_queries = {"recurring_evidence": 0, "linked_evidence": 0}
        n_queries = 0
        for conversation_index, conversation in enumerate(corpus.conversations, start=1):
            conversation_started = time.perf_counter()
            _, qa_pairs = qa_train_test_split(
                _conversation_qa_pairs(conversation),
                qa_train_fraction,
                seed=qa_split_seed,
                key=conversation.conversation_id,
            )
            if not qa_pairs:
                continue
            facts = sorted(conversation.facts, key=lambda fact: fact.id)
            fact_ids = np.asarray([fact.id for fact in facts], dtype=int)
            fact_times = np.asarray([fact.source_time for fact in facts], dtype=float)
            fact_embeddings = np.vstack([fact.embedding for fact in facts]) if facts else np.empty((0, 0))
            grouped = facts_by_message_id(facts)
            query_payloads = []
            for qa_pair in qa_pairs:
                evidence_facts = evidence_facts_for_qa(qa_pair, grouped)
                if evidence_facts:
                    query_time = max(fact.source_time for fact in evidence_facts) + probe_delay_days
                    query_payloads.append(
                        (
                            qa_pair,
                            evidence_facts,
                            query_time,
                            diagnostic_subsets(
                                evidence_facts,
                                facts,
                                conversation.events,
                                query_time,
                                fact_embeddings,
                                fact_ids,
                            ),
                        )
                    )
            query_times = [query_time for _, _, query_time, _ in query_payloads]
            batch_intensities = None
            if params is not None and model != "cosine":
                batch_intensities = batch_intensities_at_times(
                    params,
                    conversation.events,
                    query_times,
                    device=device,
                )
            conversation_queries = 0
            for query_index, (qa_pair, evidence_facts, query_time, subsets) in enumerate(query_payloads):
                query = embedding_fn(_qa_question(qa_pair))
                visible = fact_times <= query_time
                if not np.any(visible):
                    continue
                sims = _query_cosine(query, fact_embeddings[visible], device=device)
                visible_ids = fact_ids[visible]
                if model == "cosine":
                    scores = sims
                elif model == "cosine_recency":
                    if params is None:
                        raise ValueError("cosine_recency requires params for beta")
                    recency = np.exp(-params.beta * np.maximum(query_time - fact_times[visible], 0.0))
                    scores = _combine_semantic_and_temporal(sims, recency, gamma=fusion_gamma)
                else:
                    if batch_intensities is None:
                        raise ValueError(f"{model} requires Hawkes intensities")
                    intensities = batch_intensities[query_index]
                    scores = _combine_semantic_and_temporal(
                        sims,
                        intensities[visible_ids],
                        gamma=fusion_gamma,
                    )
                order = np.argsort(-scores)
                ranked_ids = [int(fact_id) for fact_id in visible_ids[order]]
                ground_truth_ids = {fact.id for fact in evidence_facts}
                n_queries += 1
                conversation_queries += 1
                for k in top_ks:
                    if ground_truth_ids.intersection(ranked_ids[:k]):
                        hits[k] += 1
                ranks = [
                    ranked_ids.index(fact_id) + 1
                    for fact_id in ground_truth_ids
                    if fact_id in ranked_ids
                ]
                rank = min(ranks) if ranks else 0
                reciprocal_rank = 0.0 if rank == 0 else 1.0 / rank
                reciprocal_ranks.append(reciprocal_rank)
                for subset in subsets:
                    subset_queries[subset] += 1
                    subset_rr[subset].append(reciprocal_rank)
                    for k in top_ks:
                        if ground_truth_ids.intersection(ranked_ids[:k]):
                            subset_hits[subset][k] += 1
            _log(
                f"Retrieval model={model} conversation={conversation_index}/{len(corpus.conversations)} "
                f"queries={conversation_queries} cumulative_queries={n_queries} "
                f"elapsed={time.perf_counter() - conversation_started:.1f}s"
            )
        rows.append(_metric_row(model, "all_test_qa", hits, reciprocal_ranks, n_queries))
        for subset in ("recurring_evidence", "linked_evidence"):
            diagnostics.append(
                _metric_row(
                    model,
                    subset,
                    subset_hits[subset],
                    subset_rr[subset],
                    subset_queries[subset],
                )
            )
        _log(f"Retrieval model={model} done queries={n_queries} elapsed={time.perf_counter() - model_started:.1f}s")
    return {"overall": rows, "diagnostics": diagnostics}


def batch_intensities_at_times(
    params: HawkesParams,
    events: list[Event],
    query_times: list[float],
    *,
    device: str | None,
) -> np.ndarray:
    if not query_times:
        return np.empty((0, params.n_memories), dtype=float)
    if device is not None and device.lower() != "cpu":
        try:
            import torch
        except ImportError:
            pass
        else:
            torch_device = resolve_torch_device(device)
            dtype = best_float_dtype(torch, torch_device)
            alpha_np = params.alpha.toarray() if sparse.issparse(params.alpha) else params.alpha
            alpha_t = torch.as_tensor(alpha_np, dtype=dtype, device=torch_device)
            mu_t = torch.as_tensor(params.mu, dtype=dtype, device=torch_device)
            query_chunk_size = adaptive_cuda_chunk_size(
                torch,
                torch_device,
                dtype,
                len(events),
                preferred=_env_int("HAWKES_RAG_RETRIEVAL_QUERY_CHUNK_SIZE", 64),
                minimum=8,
            )
            batches = []
            if events:
                event_times = torch.as_tensor(
                    [event.time for event in events],
                    dtype=dtype,
                    device=torch_device,
                )
                event_ids = torch.as_tensor(
                    [event.memory_id for event in events],
                    dtype=torch.long,
                    device=torch_device,
                )
                event_weights = torch.as_tensor(
                    [event.weight for event in events],
                    dtype=dtype,
                    device=torch_device,
                )
                beta_t = torch.as_tensor(params.beta, dtype=dtype, device=torch_device)
                with torch.inference_mode():
                    for start in range(0, len(query_times), query_chunk_size):
                        query_times_t = torch.as_tensor(
                            query_times[start : start + query_chunk_size],
                            dtype=dtype,
                            device=torch_device,
                        )
                        dt = query_times_t[:, None] - event_times[None, :]
                        past = dt > 0.0
                        decayed = torch.exp(-beta_t * torch.clamp(dt, min=0.0))
                        decayed = decayed * past.to(dtype) * event_weights[None, :]
                        excitation_by_memory = torch.zeros(
                            (len(query_times_t), params.n_memories),
                            dtype=dtype,
                            device=torch_device,
                        )
                        excitation_by_memory.scatter_add_(
                            1,
                            event_ids[None, :].expand(len(query_times_t), -1),
                            decayed,
                        )
                        intensities = mu_t[None, :] + excitation_by_memory @ alpha_t.T
                        batches.append(torch.clamp(intensities, min=1e-12).detach().cpu())
            else:
                with torch.inference_mode():
                    for start in range(0, len(query_times), query_chunk_size):
                        size = len(query_times[start : start + query_chunk_size])
                        batches.append(mu_t[None, :].expand(size, -1).detach().cpu())
            return torch.cat(batches, dim=0).numpy().astype(float)
    process = MultivariateHawkesProcess(params)
    return np.vstack(
        [
            process.intensities(
                query_time,
                [event for event in events if event.time < query_time],
            )
            for query_time in query_times
        ]
    )


def paired_bootstrap_delta_ci(
    left: HawkesParams,
    right: HawkesParams,
    splits,
    *,
    samples: int,
    seed: int,
    device: str | None,
) -> dict[str, Any]:
    deltas = []
    started = time.perf_counter()
    for index, split in enumerate(splits, start=1):
        left_value = trajectory_conditional_pll(left, split, device=device)
        right_value = trajectory_conditional_pll(right, split, device=device)
        denom = max(len(split.test_events), 1)
        deltas.append((left_value - right_value) / denom)
        _log(
            f"Bootstrap PLL split={index}/{len(splits)} events={len(split.test_events)} "
            f"delta_per_event={deltas[-1]:.6f} elapsed={time.perf_counter() - started:.1f}s"
        )
    values = np.asarray(deltas, dtype=float)
    rng = np.random.default_rng(seed)
    if values.size and samples > 0:
        indices = rng.integers(0, values.size, size=(samples, values.size))
        boot_values = np.mean(values[indices], axis=1)
    else:
        boot_values = np.asarray([], dtype=float)
    return {
        "comparison": "full_alpha_minus_diagonal_alpha",
        "unit": "nats_per_heldout_event",
        "n_paired_trajectories": int(values.size),
        "samples": int(samples),
        "mean": float(np.mean(values)) if values.size else 0.0,
        "std": float(np.std(boot_values, ddof=1)) if boot_values.size > 1 else 0.0,
        "ci95_low": float(np.quantile(boot_values, 0.025)) if boot_values.size else 0.0,
        "ci95_high": float(np.quantile(boot_values, 0.975)) if boot_values.size else 0.0,
    }


def trajectory_conditional_pll(params: HawkesParams, split, *, device: str | None) -> float:
    pll = heldout_predictive_log_likelihood(
        params,
        [split],
        device=device,
        label="bootstrap_pll",
    )
    return pll.total


def format_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# LoCoMo Hawkes-RAG Run",
        "",
        f"- dataset: `{result['dataset']}`",
        f"- eventized_cache: `{result['eventized_cache']}` ({result['cache_status']})",
        f"- conversations: {result['n_conversations']}",
        f"- conversations_full_cache: {result['n_conversations_full_cache']}",
        f"- messages: {result['n_messages']}",
        f"- messages_full_cache: {result['n_messages_full_cache']}",
        f"- facts: {result['n_facts']}",
        f"- facts_full_cache: {result['n_facts_full_cache']}",
        f"- max_facts: {result['max_facts']} (`0` means full corpus)",
        f"- embedding: `{result['embedding']}`",
        f"- qa_probe_delay_days: {result['qa_probe_delay_days']}",
        f"- qa_train_fraction: {result['qa_train_fraction']}",
        f"- qa_split_seed: {result['qa_split_seed']}",
        f"- evidence_event_weight: {result['evidence_event_weight']}",
        f"- fusion_gamma: {result['fusion_gamma']}",
        f"- dense_threshold: {result['dense_threshold']}",
        f"- events: {result['n_events']}",
        f"- events_full_cache: {result['n_events_full_cache']}",
        f"- fit_mode: {result['fit']['mode']}",
        f"- fit_success: {result['fit']['success']} ({result['fit']['message']})",
        "",
        "| Model | Held-out PLL/event | Held-out PLL total | Held-out events |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in result["heldout"]:
        lines.append(
            f"| `{row['model']}` | {row['heldout_pll_per_event']:.3f} | "
            f"{row['heldout_pll_total']:.3f} | {row['heldout_events']} |"
        )
    lines.extend(
        [
            "",
            "| Model | Recall@1 | Recall@5 | MRR | Retrieval queries |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result["retrieval"]["overall"]:
        lines.append(
            f"| `{row['model']}` | {row['recall_at_1']:.3f} | {row['recall_at_5']:.3f} | "
            f"{row['mrr']:.3f} | {row['queries']} |"
        )
    lines.extend(
        [
            "",
            "## Mechanism Diagnostics",
            "",
            "| Subset | Model | Recall@1 | Recall@5 | MRR | Queries |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result["retrieval"]["diagnostics"]:
        lines.append(
            f"| `{row['subset']}` | `{row['model']}` | {row['recall_at_1']:.3f} | "
            f"{row['recall_at_5']:.3f} | {row['mrr']:.3f} | {row['queries']} |"
        )
    boot = result["paired_bootstrap"]
    lines.extend(
        [
            "",
            "## Paired Bootstrap CI",
            "",
            f"- comparison: `{boot['comparison']}`",
            f"- mean_delta_nats_per_event: {boot['mean']:.3f}",
            f"- bootstrap_std: {boot['std']:.3f}",
            f"- ci95: [{boot['ci95_low']:.3f}, {boot['ci95_high']:.3f}]",
            f"- paired_trajectories: {boot['n_paired_trajectories']}",
            f"- bootstrap_samples: {boot['samples']}",
        ]
    )
    return "\n".join(lines)


def limit_corpus_facts(corpus: EventizedCorpus, max_facts: int) -> EventizedCorpus:
    keep_ids = balanced_fact_ids(corpus, max_facts)
    conversations = []
    facts_by_id: dict[int, AtomicFact] = {}
    for conversation in corpus.conversations:
        facts = [fact for fact in conversation.facts if fact.id in keep_ids]
        events = [event for event in conversation.events if event.memory_id in keep_ids]
        if not facts or not events:
            continue
        for fact in facts:
            facts_by_id[fact.id] = fact
        conversations.append(
            _make_eventized_conversation(
                conversation_id=conversation.conversation_id,
                messages=conversation.messages,
                facts=facts,
                events=events,
                horizon=conversation.horizon,
                qa_pairs=_conversation_qa_pairs(conversation),
            )
        )
    remap = {old_id: new_id for new_id, old_id in enumerate(sorted(facts_by_id))}
    remapped_conversations = []
    remapped_facts_by_id: dict[int, AtomicFact] = {}
    for conversation in conversations:
        facts = []
        for fact in conversation.facts:
            remapped = AtomicFact(
                id=remap[fact.id],
                conversation_id=fact.conversation_id,
                text=fact.text,
                source_message_id=fact.source_message_id,
                source_time=fact.source_time,
                embedding=fact.embedding,
            )
            facts.append(remapped)
            remapped_facts_by_id[remapped.id] = remapped
        events = [
            Event(time=event.time, memory_id=remap[event.memory_id], weight=event.weight)
            for event in conversation.events
            if event.memory_id in remap
        ]
        remapped_conversations.append(
            _make_eventized_conversation(
                conversation_id=conversation.conversation_id,
                messages=conversation.messages,
                facts=facts,
                events=events,
                horizon=conversation.horizon,
                qa_pairs=_conversation_qa_pairs(conversation),
            )
        )
    remapped_facts = [remapped_facts_by_id[key] for key in sorted(remapped_facts_by_id)]
    return EventizedCorpus(conversations=remapped_conversations, facts=remapped_facts)


def balanced_fact_ids(corpus: EventizedCorpus, max_facts: int) -> set[int]:
    per_conversation = [sorted(conversation.facts, key=lambda fact: fact.id) for conversation in corpus.conversations]
    keep_ids: set[int] = set()
    offset = 0
    while len(keep_ids) < max_facts:
        added = False
        for facts in per_conversation:
            if offset < len(facts):
                keep_ids.add(facts[offset].id)
                added = True
                if len(keep_ids) >= max_facts:
                    break
        if not added:
            break
        offset += 1
    return keep_ids


def stable_similarity_params(
    corpus: EventizedCorpus,
    embeddings: np.ndarray,
    *,
    dense_threshold: int,
) -> HawkesParams:
    mu = estimate_mu(corpus.trajectories(), corpus.horizons(), corpus.n_memories)
    alpha = topk_similarity_prior(
        embeddings,
        threshold=0.32,
        top_k=32,
        dense_output=corpus.n_memories <= dense_threshold,
    )
    if sparse.issparse(alpha):
        alpha = (0.45 * alpha).tolil()
        alpha.setdiag(0.6)
        alpha = alpha.tocsr()
    else:
        alpha = 0.45 * alpha
        np.fill_diagonal(alpha, 0.6)
    return HawkesParams(mu=mu, alpha=alpha, beta=1.0).stable(max_radius=0.95)


def zero_alpha_like(alpha: np.ndarray) -> np.ndarray:
    if sparse.issparse(alpha):
        return sparse.csr_matrix(alpha.shape, dtype=float)
    return np.zeros_like(alpha)


def write_eventized_corpus(corpus: EventizedCorpus, path: Path) -> None:
    payload = {
        "schema": EVENTIZED_SCHEMA,
        "conversations": [
            {
                "conversation_id": conversation.conversation_id,
                "messages": [
                    {
                        "conversation_id": message.conversation_id,
                        "message_id": message.message_id,
                        "text": message.text,
                        "timestamp": message.timestamp,
                        "speaker": message.speaker,
                    }
                    for message in conversation.messages
                ],
                "facts": [fact_to_json(fact) for fact in conversation.facts],
                "events": [event_to_json(event) for event in conversation.events],
                "horizon": conversation.horizon,
                "qa_pairs": [qa_to_json(qa_pair) for qa_pair in _conversation_qa_pairs(conversation)],
            }
            for conversation in corpus.conversations
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def read_eventized_corpus(path: Path) -> EventizedCorpus:
    payload = json.loads(path.read_text())
    if payload.get("schema") != EVENTIZED_SCHEMA:
        raise ValueError(f"unsupported eventized cache schema in {path}")
    conversations = []
    facts_by_id: dict[int, AtomicFact] = {}
    for raw in payload["conversations"]:
        messages = [
            ConversationMessage(
                conversation_id=item["conversation_id"],
                message_id=item["message_id"],
                text=item["text"],
                timestamp=float(item["timestamp"]),
                speaker=item.get("speaker", ""),
            )
            for item in raw["messages"]
        ]
        facts = [fact_from_json(item) for item in raw["facts"]]
        for fact in facts:
            facts_by_id[fact.id] = fact
        conversations.append(
            _make_eventized_conversation(
                conversation_id=raw["conversation_id"],
                messages=messages,
                facts=facts,
                events=[event_from_json(item) for item in raw["events"]],
                horizon=float(raw["horizon"]),
                qa_pairs=[qa_from_json(item) for item in raw.get("qa_pairs", [])],
            )
        )
    facts = [facts_by_id[key] for key in sorted(facts_by_id)]
    return EventizedCorpus(conversations=conversations, facts=facts)


def fact_to_json(fact: AtomicFact) -> dict[str, Any]:
    return {
        "id": fact.id,
        "conversation_id": fact.conversation_id,
        "text": fact.text,
        "source_message_id": fact.source_message_id,
        "source_time": fact.source_time,
        "embedding": fact.embedding.tolist(),
    }


def fact_from_json(item: dict[str, Any]) -> AtomicFact:
    return AtomicFact(
        id=int(item["id"]),
        conversation_id=item["conversation_id"],
        text=item["text"],
        source_message_id=item["source_message_id"],
        source_time=float(item["source_time"]),
        embedding=np.asarray(item["embedding"], dtype=float),
    )


def event_to_json(event: Event) -> dict[str, Any]:
    return {"time": event.time, "memory_id": event.memory_id, "weight": event.weight}


def event_from_json(item: dict[str, Any]) -> Event:
    return Event(time=float(item["time"]), memory_id=int(item["memory_id"]), weight=float(item["weight"]))


def qa_to_json(qa_pair: LoCoMoQAPair) -> dict[str, Any]:
    if isinstance(qa_pair, dict):
        return {
            "question": qa_pair["question"],
            "answer": qa_pair.get("answer"),
            "evidence_message_ids": qa_pair.get("evidence_message_ids", []),
            "category": qa_pair.get("category"),
        }
    return {
        "question": qa_pair.question,
        "answer": qa_pair.answer,
        "evidence_message_ids": qa_pair.evidence_message_ids,
        "category": qa_pair.category,
    }


def qa_from_json(item: dict[str, Any]) -> LoCoMoQAPair:
    if LoCoMoQAPair is Any:
        return {
            "question": item["question"],
            "answer": item.get("answer"),
            "evidence_message_ids": [str(value) for value in item.get("evidence_message_ids", [])],
            "category": item.get("category"),
        }
    return LoCoMoQAPair(
        question=item["question"],
        answer=item.get("answer"),
        evidence_message_ids=[str(value) for value in item.get("evidence_message_ids", [])],
        category=item.get("category"),
    )


if __name__ == "__main__":
    main()

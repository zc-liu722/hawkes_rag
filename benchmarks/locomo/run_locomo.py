from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hawkes_rag.core import Event, HawkesParams, diagonal_only
from hawkes_rag.estimation import LowRankHawkesEstimator
from hawkes_rag.evaluation import heldout_predictive_log_likelihood
from hawkes_rag.locomo import (
    AtomicFact,
    ConversationMessage,
    EventizedConversation,
    EventizedCorpus,
    LoCoMoEventizer,
    load_official_locomo10_json,
)
from hawkes_rag.utils import pairwise_cosine


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Hawkes-RAG on the official LoCoMo corpus.")
    parser.add_argument("--data", type=Path, default=Path("benchmarks/locomo/cache/locomo10.json"))
    parser.add_argument("--eventized-cache", type=Path, default=Path("outputs/locomo_eventized.json"))
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--fit-mle", action="store_true", help="Run low-rank MLE instead of the fast stable similarity-alpha default.")
    parser.add_argument(
        "--max-facts",
        type=int,
        default=80,
        help="Fit/evaluate the first N eventized facts for a practical default; use 0 for full corpus.",
    )
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(
            f"Missing LoCoMo data at {args.data}. Run: python3 benchmarks/locomo/download.py"
        )
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    args.eventized_cache.parent.mkdir(parents=True, exist_ok=True)

    if args.eventized_cache.exists() and not args.refresh_cache:
        corpus = read_eventized_corpus(args.eventized_cache)
        cache_status = "loaded"
    else:
        conversations = load_official_locomo10_json(args.data)
        corpus = LoCoMoEventizer().eventize(conversations)
        write_eventized_corpus(corpus, args.eventized_cache)
        cache_status = "written"

    full_corpus = corpus
    if args.max_facts > 0:
        corpus = limit_corpus_facts(corpus, args.max_facts)

    embeddings = corpus.embeddings()
    if args.fit_mle:
        estimator = LowRankHawkesEstimator.from_embeddings(
            embeddings,
            rank=args.rank,
            seed=0,
            learn_beta=True,
        )
        fit = estimator.fit(
            corpus.trajectories(),
            corpus.horizons(),
            active_memory_ids=corpus.active_memory_ids(),
            max_iter=args.max_iter,
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
        }
    else:
        full_params = stable_similarity_params(corpus, embeddings)
        fit_payload = {
            "mode": "stable_similarity_alpha",
            "success": True,
            "objective": None,
            "message": "MLE skipped; pass --fit-mle for optimizer-based fitting",
            "n_iter": 0,
            "rank": None,
            "beta": full_params.beta,
        }

    splits = corpus.heldout_splits(args.train_fraction)
    full_pll = heldout_predictive_log_likelihood(full_params, splits)
    diagonal_pll = heldout_predictive_log_likelihood(diagonal_only(full_params), splits)
    naive_pll = heldout_predictive_log_likelihood(
        HawkesParams(
            mu=estimate_mu(corpus.trajectories(), corpus.horizons(), corpus.n_memories),
            alpha=np.zeros_like(full_params.alpha),
            beta=full_params.beta,
        ),
        splits,
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
    }
    (args.outputs_dir / "locomo_results.json").write_text(json.dumps(result, indent=2) + "\n")
    markdown = format_markdown(result)
    (args.outputs_dir / "locomo_results.md").write_text(markdown + "\n")
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
            EventizedConversation(
                conversation_id=conversation.conversation_id,
                messages=conversation.messages,
                facts=facts,
                events=events,
                horizon=conversation.horizon,
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
            EventizedConversation(
                conversation_id=conversation.conversation_id,
                messages=conversation.messages,
                facts=facts,
                events=events,
                horizon=conversation.horizon,
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


def stable_similarity_params(corpus: EventizedCorpus, embeddings: np.ndarray) -> HawkesParams:
    mu = estimate_mu(corpus.trajectories(), corpus.horizons(), corpus.n_memories)
    alpha = 0.45 * np.maximum(0.0, pairwise_cosine(embeddings) - 0.32)
    np.fill_diagonal(alpha, 0.6)
    return HawkesParams(mu=mu, alpha=alpha, beta=1.0).stable(max_radius=0.95)


def write_eventized_corpus(corpus: EventizedCorpus, path: Path) -> None:
    payload = {
        "schema": "hawkes_rag.eventized_locomo.v1",
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
            }
            for conversation in corpus.conversations
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def read_eventized_corpus(path: Path) -> EventizedCorpus:
    payload = json.loads(path.read_text())
    if payload.get("schema") != "hawkes_rag.eventized_locomo.v1":
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
            EventizedConversation(
                conversation_id=raw["conversation_id"],
                messages=messages,
                facts=facts,
                events=[event_from_json(item) for item in raw["events"]],
                horizon=float(raw["horizon"]),
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


if __name__ == "__main__":
    main()

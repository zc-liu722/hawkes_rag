from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.locomo.run_locomo import (  # noqa: E402
    _combine_semantic_and_temporal,
    batch_intensities_at_times,
    configure_huggingface_defaults,
    diagonal_only,
    estimate_mu,
    stable_similarity_params,
)
from hawkes_rag.core import Event, HawkesParams  # noqa: E402
from hawkes_rag.embeddings import EmbeddingFn, make_embedding_fn  # noqa: E402
from hawkes_rag.estimation import LowRankHawkesEstimator  # noqa: E402
from hawkes_rag.locomo import (  # noqa: E402
    AtomicFact,
    ConversationMessage,
    EventizedConversation,
    EventizedCorpus,
    LoCoMoEventizer,
)
from hawkes_rag.memory import _query_cosine  # noqa: E402


OUTPUT_JSON = "locomo_plus_results.json"
OUTPUT_MD = "locomo_plus_results.md"
LOCOMO_PLUS_URL = (
    "https://raw.githubusercontent.com/xjtuleeyf/Locomo-Plus/main/data/locomo_plus.json"
)
LOCOMO10_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"


@dataclass(frozen=True)
class LoCoMoPlusProbe:
    probe_id: str
    trigger_query: str
    evidence_text: str
    messages: list[ConversationMessage]
    evidence_message_ids: list[str]
    time_gap_days: float | None
    category: str


class CachedEmbeddingFn:
    """Batch-prewarm sentence-transformer embeddings, then serve eventizer calls from RAM."""

    def __init__(self, embedding_fn: EmbeddingFn, *, batch_size: int, log_prefix: str = "Embedding"):
        self.embedding_fn = embedding_fn
        self.batch_size = int(batch_size)
        self.log_prefix = log_prefix
        self.cache: dict[str, np.ndarray] = {}

    def __call__(self, text: str) -> np.ndarray:
        if text not in self.cache:
            self.prewarm([text])
        return self.cache[text].copy()

    def prewarm(self, texts: list[str]) -> None:
        pending = [text for text in dict.fromkeys(texts) if text and text not in self.cache]
        if not pending:
            return
        model = getattr(self.embedding_fn, "model", None)
        if model is not None and hasattr(model, "encode"):
            _log(f"{self.log_prefix}: batch-encoding {len(pending)} texts")
            vectors = model.encode(
                pending,
                normalize_embeddings=True,
                batch_size=getattr(self.embedding_fn, "batch_size", self.batch_size),
                convert_to_numpy=True,
            )
            for text, vector in zip(pending, vectors):
                self.cache[text] = np.asarray(vector, dtype=float)
            return
        _log(f"{self.log_prefix}: caching {len(pending)} texts")
        for text in pending:
            self.cache[text] = np.asarray(self.embedding_fn(text), dtype=float)


def _log(message: str) -> None:
    print(f"[locomo-plus] {message}", flush=True)


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(
        description=(
            "Run a LoCoMo-Plus cognitive retrieval probe for Hawkes-RAG. "
            "The probe grades whether trigger queries retrieve the cue/evidence memories."
        )
    )
    parser.add_argument("--data", type=Path, default=Path("benchmarks/locomo/cache/locomo_plus.json"))
    parser.add_argument("--locomo-data", type=Path, default=Path("benchmarks/locomo/cache/locomo10.json"))
    parser.add_argument(
        "--locomo-plus-url",
        default=LOCOMO_PLUS_URL,
        help="Source URL used when --data is missing or --force-download is set.",
    )
    parser.add_argument(
        "--locomo-url",
        default=LOCOMO10_URL,
        help="Source URL used when --locomo-data is needed for raw LoCoMo-Plus records.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Refresh --data from --locomo-plus-url before running.",
    )
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--embedding",
        choices=["hashing", "minilm", "bge"],
        default="hashing",
        help="Use hashing for a no-download smoke run; use minilm/bge for a real semantic run.",
    )
    parser.add_argument("--model-cache-dir", type=Path, default=Path("benchmarks/locomo/cache/models"))
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-probes", type=int, default=0, help="First N probes; 0 means all.")
    parser.add_argument(
        "--max-context-messages",
        type=int,
        default=120,
        help="Maximum LoCoMo distractor messages per raw LoCoMo-Plus probe; 0 means all.",
    )
    parser.add_argument("--context-seed", type=int, default=0)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument(
        "--optimizer",
        choices=["lbfgsb", "adam"],
        default="lbfgsb",
        help="Low-rank MLE optimizer. Use adam to run the fit with PyTorch on GPU when available.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--dense-threshold", type=int, default=4000)
    parser.add_argument(
        "--fit-mle",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fit low-rank Hawkes MLE. Default uses a stable similarity-alpha smoke baseline.",
    )
    parser.add_argument("--fusion-gamma", type=float, default=0.2)
    parser.add_argument("--probe-delay-days", type=float, default=0.0)
    args = parser.parse_args()

    ensure_json_data(args.data, args.locomo_plus_url, "LoCoMo-Plus", force=args.force_download)
    plus_payload = json.loads(args.data.read_text())
    locomo_payload = None
    if needs_locomo_context(plus_payload):
        ensure_json_data(args.locomo_data, args.locomo_url, "LoCoMo10", force=False)
        locomo_payload = json.loads(args.locomo_data.read_text())
    args.outputs_dir.mkdir(parents=True, exist_ok=True)

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
        log_prefix="LoCoMo-Plus embeddings",
    )

    _log(f"Eventizing cognitive cue/dialogue probes on device={args.device or 'auto'}")
    corpus = eventize_probes(probes, embedding_fn)
    embeddings = corpus.embeddings()
    _log(
        f"Eventized conversations={len(corpus.conversations)} "
        f"facts={corpus.n_memories} events={sum(len(c.events) for c in corpus.conversations)}"
    )

    if args.fit_mle:
        _log(
            f"Fitting Hawkes MLE rank={args.rank} max_iter={args.max_iter} "
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
        _log("Building stable similarity-alpha Hawkes parameters")
        params = stable_similarity_params(corpus, embeddings, dense_threshold=args.dense_threshold)
        fit_payload = {
            "mode": "stable_similarity_alpha",
            "success": True,
            "objective": None,
            "message": "MLE skipped via --no-fit-mle default",
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
            alpha=np.zeros((corpus.n_memories, corpus.n_memories), dtype=float),
            beta=params.beta,
        ),
    }
    _log("Scoring trigger-query retrieval")
    retrieval = retrieval_metrics(
        probes,
        corpus,
        embedding_fn,
        models,
        fusion_gamma=args.fusion_gamma,
        probe_delay_days=args.probe_delay_days,
        device=args.device,
    )

    result = {
        "dataset": str(args.data),
        "locomo_context_dataset": str(args.locomo_data) if locomo_payload is not None else None,
        "n_probes": len(probes),
        "n_conversations": len(corpus.conversations),
        "n_facts": corpus.n_memories,
        "n_events": sum(len(conversation.events) for conversation in corpus.conversations),
        "embedding": args.embedding,
        "embedding_batch_size": args.embedding_batch_size,
        "fusion_gamma": args.fusion_gamma,
        "probe_delay_days": args.probe_delay_days,
        "max_context_messages": args.max_context_messages,
        "dense_threshold": args.dense_threshold,
        "fit": fit_payload,
        "retrieval": retrieval,
    }
    (args.outputs_dir / OUTPUT_JSON).write_text(json.dumps(result, indent=2) + "\n")
    markdown = format_markdown(result)
    (args.outputs_dir / OUTPUT_MD).write_text(markdown + "\n")
    _log(f"Run complete in {time.perf_counter() - started:.1f}s")
    print(markdown)


def load_locomo_plus_probes(
    path: Path,
    *,
    locomo_payload: Any | None = None,
    max_context_messages: int,
    context_seed: int,
) -> list[LoCoMoPlusProbe]:
    payload = json.loads(path.read_text())
    records = payload if isinstance(payload, list) else payload.get("data") or payload.get("samples") or []
    if not isinstance(records, list):
        raise ValueError("LoCoMo-Plus input must be a JSON list or object with data/samples list")
    probes = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        locomo_record = locomo_record_for_plus_record(record, index, locomo_payload)
        probe = probe_from_record(
            record,
            index,
            locomo_record=locomo_record,
            max_context_messages=max_context_messages,
            context_seed=context_seed,
        )
        if probe is not None:
            probes.append(probe)
    return probes


def needs_locomo_context(payload: Any) -> bool:
    records = payload if isinstance(payload, list) else payload.get("data") or payload.get("samples") or []
    if not isinstance(records, list):
        return False
    for record in records[:20]:
        if isinstance(record, dict) and first_text(record, ["input_prompt", "prompt", "context", "conversation"]):
            return False
    return True


def ensure_json_data(path: Path, url: str, label: str, *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        _log(f"Using cached {label} data: {path} sha256={sha256(path)}")
        return
    action = "Refreshing" if path.exists() else "Downloading"
    _log(f"{action} {label} data from {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
    except Exception as exc:
        raise SystemExit(
            f"Could not download {label} data from {url}: {exc}. "
            f"Place the JSON at {path} or pass a reachable mirror URL."
        ) from exc
    path.write_bytes(payload)
    _log(f"Downloaded {len(payload)} bytes to {path} sha256={sha256(path)}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_from_record(
    record: dict[str, Any],
    index: int,
    *,
    locomo_record: dict[str, Any] | None,
    max_context_messages: int,
    context_seed: int,
) -> LoCoMoPlusProbe | None:
    category = str(record.get("category") or record.get("task") or record.get("relation_type") or "cognitive")
    if category not in {"6", "cognitive", "Cognitive"} and not any(
        key in record for key in ("cue_dialogue", "trigger_query", "trigger")
    ):
        return None
    trigger = first_text(record, ["trigger_query", "trigger", "question", "query"])
    evidence = first_text(record, ["cue_dialogue", "evidence", "evidence_text", "cue", "answer"])
    if not trigger or not evidence:
        return None
    probe_id = str(record.get("id") or record.get("sample_id") or record.get("qid") or f"plus_{index}")
    context = first_text(record, ["input_prompt", "prompt", "context", "conversation"]) or ""
    time_gap_days = parse_time_gap_days(
        record.get("time_gap") or record.get("timegap") or record.get("delay")
    )
    if context:
        messages, evidence_ids = build_probe_messages(
            probe_id,
            evidence_text=evidence,
            context_text=context,
            trigger_query=trigger,
            time_gap_days=time_gap_days,
        )
    else:
        if locomo_record is None:
            return None
        messages, evidence_ids = build_probe_messages_from_locomo(
            probe_id,
            record=record,
            locomo_record=locomo_record,
            evidence_text=evidence,
            trigger_query=trigger,
            time_gap_days=time_gap_days,
            max_context_messages=max_context_messages,
            context_seed=context_seed,
        )
    if not messages or not evidence_ids:
        return None
    return LoCoMoPlusProbe(
        probe_id=probe_id,
        trigger_query=trigger,
        evidence_text=evidence,
        messages=messages,
        evidence_message_ids=evidence_ids,
        time_gap_days=time_gap_days,
        category=category,
    )


def first_text(record: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = record.get(key)
        text = value_to_text(value)
        if text:
            return text
    return ""


def value_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        speaker = value.get("speaker") or value.get("role") or value.get("name")
        text = value.get("text") or value.get("content") or value.get("message")
        if text is not None:
            prefix = f"{speaker}: " if speaker else ""
            return f"{prefix}{str(text).strip()}".strip()
        lines = []
        for key, item in value.items():
            item_text = value_to_text(item)
            if item_text:
                lines.append(f"{key}: {item_text}")
        return "\n".join(lines).strip()
    if isinstance(value, list):
        lines = [value_to_text(item) for item in value]
        return "\n".join(line for line in lines if line).strip()
    return ""


def locomo_record_for_plus_record(
    record: dict[str, Any],
    index: int,
    locomo_payload: Any | None,
) -> dict[str, Any] | None:
    if locomo_payload is None:
        return None
    samples = locomo_payload if isinstance(locomo_payload, list) else locomo_payload.get("data") or []
    if not isinstance(samples, list) or not samples:
        return None
    sample_hint = (
        record.get("sample_id")
        or record.get("locomo_sample_id")
        or record.get("conversation_id")
        or record.get("dialogue_id")
    )
    if sample_hint is not None:
        for sample in samples:
            if isinstance(sample, dict) and str(sample.get("sample_id")) == str(sample_hint):
                return sample
    return samples[index % len(samples)] if isinstance(samples[index % len(samples)], dict) else None


def build_probe_messages_from_locomo(
    probe_id: str,
    *,
    record: dict[str, Any],
    locomo_record: dict[str, Any],
    evidence_text: str,
    trigger_query: str,
    time_gap_days: float | None,
    max_context_messages: int,
    context_seed: int,
) -> tuple[list[ConversationMessage], list[str]]:
    context_messages = locomo_context_messages(
        probe_id,
        record=record,
        locomo_record=locomo_record,
        max_context_messages=max_context_messages,
        seed=context_seed,
    )
    cue_turns = text_to_turns(evidence_text)
    if not cue_turns:
        return context_messages, []
    query_time = max((message.timestamp for message in context_messages), default=1.0) + 1.0
    cue_time = max(0.0, query_time - float(time_gap_days or 30.0))
    cue_speaker_map = plus_speaker_map(record, locomo_record)

    evidence_ids = []
    cue_messages = []
    for turn, (speaker, text) in enumerate(cue_turns):
        mapped_speaker = cue_speaker_map.get(speaker.strip(), speaker)
        message_id = f"{probe_id}:evidence:{turn}"
        evidence_ids.append(message_id)
        cue_messages.append(
            ConversationMessage(
                conversation_id=probe_id,
                message_id=message_id,
                text=text,
                timestamp=cue_time + turn * 1e-4,
                speaker=mapped_speaker,
            )
        )
    trigger_norm = normalize_text(trigger_query)
    evidence_norm = {normalize_text(message.text) for message in cue_messages}
    messages = [
        message
        for message in context_messages
        if normalize_text(message.text) not in evidence_norm
        and normalize_text(message.text) != trigger_norm
    ]
    messages.extend(cue_messages)
    return sorted(messages, key=lambda message: message.timestamp), evidence_ids


def locomo_context_messages(
    probe_id: str,
    *,
    record: dict[str, Any],
    locomo_record: dict[str, Any],
    max_context_messages: int,
    seed: int,
) -> list[ConversationMessage]:
    conversation = locomo_record.get("conversation")
    if not isinstance(conversation, dict):
        return []
    speaker_map = plus_speaker_map(record, locomo_record)
    messages = []
    for session_number in official_session_numbers(conversation):
        timestamp_base = float(session_number)
        turns = conversation.get(f"session_{session_number}", [])
        if not isinstance(turns, list):
            continue
        for turn_index, turn in enumerate(turns):
            if not isinstance(turn, dict):
                continue
            text = str(turn.get("text") or "").strip()
            if not text:
                continue
            raw_speaker = str(turn.get("speaker") or "")
            speaker = speaker_map.get(raw_speaker, raw_speaker)
            messages.append(
                ConversationMessage(
                    conversation_id=probe_id,
                    message_id=f"{probe_id}:locomo:{session_number}:{turn_index}",
                    text=text,
                    timestamp=timestamp_base + turn_index * 1e-4,
                    speaker=speaker,
                )
            )
    if max_context_messages <= 0 or len(messages) <= max_context_messages:
        return messages
    return sample_context_messages(messages, max_context_messages, seed=seed, key=probe_id)


def sample_context_messages(
    messages: list[ConversationMessage],
    max_context_messages: int,
    *,
    seed: int,
    key: str,
) -> list[ConversationMessage]:
    digest = hashlib.sha1(f"{seed}:{key}".encode("utf-8")).hexdigest()
    rng = np.random.default_rng(int(digest[:16], 16))
    session_buckets: dict[int, list[ConversationMessage]] = {}
    for message in messages:
        session_buckets.setdefault(int(message.timestamp), []).append(message)
    selected = []
    per_session = max(1, max_context_messages // max(len(session_buckets), 1))
    for bucket in session_buckets.values():
        if len(bucket) <= per_session:
            selected.extend(bucket)
        else:
            indices = rng.choice(len(bucket), size=per_session, replace=False)
            selected.extend(bucket[int(index)] for index in indices)
    remaining = max_context_messages - len(selected)
    if remaining > 0:
        selected_ids = {message.message_id for message in selected}
        rest = [message for message in messages if message.message_id not in selected_ids]
        if rest:
            indices = rng.choice(len(rest), size=min(remaining, len(rest)), replace=False)
            selected.extend(rest[int(index)] for index in indices)
    return sorted(selected[:max_context_messages], key=lambda message: message.timestamp)


def plus_speaker_map(record: dict[str, Any], locomo_record: dict[str, Any]) -> dict[str, str]:
    conversation = locomo_record.get("conversation", {})
    speaker_a = str(conversation.get("speaker_a") or "speaker_a")
    speaker_b = str(conversation.get("speaker_b") or "speaker_b")
    plus_a = str(record.get("speaker_a") or record.get("name_a") or "speaker_a")
    plus_b = str(record.get("speaker_b") or record.get("name_b") or "speaker_b")
    return {
        plus_a: speaker_a,
        plus_b: speaker_b,
        "speaker_a": speaker_a,
        "speaker_b": speaker_b,
        speaker_a: speaker_a,
        speaker_b: speaker_b,
    }


def official_session_numbers(conversation: dict[str, Any]) -> list[int]:
    numbers = []
    for key, value in conversation.items():
        match = re.fullmatch(r"session_(\d+)", key)
        if match and isinstance(value, list):
            numbers.append(int(match.group(1)))
    return sorted(numbers)


def build_probe_messages(
    probe_id: str,
    *,
    evidence_text: str,
    context_text: str,
    trigger_query: str,
    time_gap_days: float | None,
) -> tuple[list[ConversationMessage], list[str]]:
    evidence_turns = text_to_turns(evidence_text)
    context_turns = text_to_turns(context_text)
    trigger_normalized = normalize_text(trigger_query)
    evidence_normalized = {normalize_text(text) for _, text in evidence_turns}

    messages = []
    evidence_ids = []
    for turn, (speaker, text) in enumerate(evidence_turns):
        message_id = f"{probe_id}:evidence:{turn}"
        evidence_ids.append(message_id)
        messages.append(
            ConversationMessage(
                conversation_id=probe_id,
                message_id=message_id,
                text=text,
                timestamp=float(turn) * 1e-4,
                speaker=speaker,
            )
        )

    distractor_time = max(1.0, float(time_gap_days or 1.0) / 2.0)
    added = 0
    for speaker, text in context_turns:
        normalized = normalize_text(text)
        if not normalized or normalized == trigger_normalized or normalized in evidence_normalized:
            continue
        message_id = f"{probe_id}:context:{added}"
        messages.append(
            ConversationMessage(
                conversation_id=probe_id,
                message_id=message_id,
                text=text,
                timestamp=distractor_time + added * 1e-4,
                speaker=speaker,
            )
        )
        added += 1

    return sorted(messages, key=lambda message: message.timestamp), evidence_ids


def text_to_turns(text: str) -> list[tuple[str, str]]:
    turns = []
    for raw_line in re.split(r"\n+", text):
        line = raw_line.strip()
        if not line:
            continue
        parsed = parse_dialogue_line(line)
        if parsed is not None:
            turns.append(parsed)
            continue
        cleaned = re.sub(r"^(cue dialogue|evidence|dialogue|conversation)\s*:\s*", "", line, flags=re.I)
        if cleaned and not cleaned.lower().startswith(("question:", "trigger:")):
            turns.append(("", cleaned))
    return turns


def parse_dialogue_line(line: str) -> tuple[str, str] | None:
    patterns = [
        r"^\s*([^:]{1,48})\s*:\s*(.+)$",
        r"^\s*([^,]{1,48})\s+said,\s*[\"'](.+?)[\"']\s*$",
        r"^\s*([^,]{1,48})\s+says,\s*[\"'](.+?)[\"']\s*$",
    ]
    for pattern in patterns:
        match = re.match(pattern, line)
        if match:
            speaker, text = match.group(1).strip(), match.group(2).strip()
            if text:
                return speaker, text
    return None


def parse_time_gap_days(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    lowered = value.lower().replace("-", " ")
    amount = first_number_or_word_number(lowered)
    if amount is None:
        return None
    if "year" in lowered:
        return amount * 365.0
    if "month" in lowered:
        return amount * 30.0
    if "week" in lowered:
        return amount * 7.0
    if "hour" in lowered:
        return amount / 24.0
    if "minute" in lowered:
        return amount / 1440.0
    return amount


def first_number_or_word_number(text: str) -> float | None:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
    if match:
        return float(match.group(1))
    word_numbers = {
        "a": 1.0,
        "an": 1.0,
        "one": 1.0,
        "two": 2.0,
        "three": 3.0,
        "four": 4.0,
        "five": 5.0,
        "six": 6.0,
        "seven": 7.0,
        "eight": 8.0,
        "nine": 9.0,
        "ten": 10.0,
        "eleven": 11.0,
        "twelve": 12.0,
        "thirteen": 13.0,
        "fourteen": 14.0,
        "fifteen": 15.0,
        "sixteen": 16.0,
        "seventeen": 17.0,
        "eighteen": 18.0,
        "nineteen": 19.0,
        "twenty": 20.0,
    }
    for token in re.findall(r"[a-z]+", text):
        if token in word_numbers:
            return word_numbers[token]
    return None


def normalize_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def eventize_probes(probes: list[LoCoMoPlusProbe], embedding_fn: EmbeddingFn) -> EventizedCorpus:
    eventized = []
    all_facts: list[AtomicFact] = []
    next_fact_id = 0
    eventizer = LoCoMoEventizer(embedding_fn=embedding_fn)
    prewarm = getattr(embedding_fn, "prewarm", None)
    if callable(prewarm):
        texts = [probe.trigger_query for probe in probes]
        for probe in probes:
            for message in probe.messages:
                texts.append(message.text)
                texts.extend(eventizer.fact_extractor.extract(message))
        prewarm(texts)
    for probe in probes:
        conversation, next_fact_id = eventizer.eventize_conversation(
            probe.messages,
            next_fact_id=next_fact_id,
            qa_pairs=None,
        )
        eventized.append(conversation)
        all_facts.extend(conversation.facts)
    return EventizedCorpus(conversations=eventized, facts=all_facts)


def retrieval_metrics(
    probes: list[LoCoMoPlusProbe],
    corpus: EventizedCorpus,
    embedding_fn: EmbeddingFn,
    params_by_model: dict[str, HawkesParams | None],
    *,
    fusion_gamma: float,
    probe_delay_days: float,
    device: str | None,
    top_ks: tuple[int, ...] = (1, 5),
) -> dict[str, list[dict[str, Any]]]:
    probe_by_id = {probe.probe_id: probe for probe in probes}
    rows = []
    by_gap = []
    for model, params in params_by_model.items():
        hits = {k: 0 for k in top_ks}
        reciprocal_ranks = []
        gap_stats: dict[str, dict[str, Any]] = {}
        n_queries = 0
        for conversation in corpus.conversations:
            probe = probe_by_id.get(conversation.conversation_id)
            if probe is None or not conversation.facts:
                continue
            facts = sorted(conversation.facts, key=lambda fact: fact.id)
            fact_ids = np.asarray([fact.id for fact in facts], dtype=int)
            fact_times = np.asarray([fact.source_time for fact in facts], dtype=float)
            fact_embeddings = np.vstack([fact.embedding for fact in facts])
            evidence_ids = {
                fact.id
                for fact in facts
                if fact.source_message_id in set(probe.evidence_message_ids)
            }
            if not evidence_ids:
                continue
            query_time = (
                max(fact.source_time for fact in facts if fact.id in evidence_ids)
                + float(probe.time_gap_days or 0.0)
                + probe_delay_days
            )
            visible = fact_times <= query_time
            if not np.any(visible):
                continue
            query = embedding_fn(probe.trigger_query)
            sims = _query_cosine(query, fact_embeddings[visible], device=device)
            visible_ids = fact_ids[visible]
            if model == "cosine":
                scores = sims
            elif model == "cosine_recency":
                if params is None:
                    raise ValueError("cosine_recency requires params")
                recency = np.exp(-params.beta * np.maximum(query_time - fact_times[visible], 0.0))
                scores = _combine_semantic_and_temporal(sims, recency, gamma=fusion_gamma)
            else:
                if params is None:
                    raise ValueError(f"{model} requires Hawkes parameters")
                intensities = batch_intensities_at_times(
                    params,
                    conversation.events,
                    [query_time],
                    device=device,
                )[0]
                scores = _combine_semantic_and_temporal(
                    sims,
                    intensities[visible_ids],
                    gamma=fusion_gamma,
                )
            ranked_ids = [int(fact_id) for fact_id in visible_ids[np.argsort(-scores)]]
            n_queries += 1
            reciprocal_rank = reciprocal_rank_for(ranked_ids, evidence_ids)
            reciprocal_ranks.append(reciprocal_rank)
            for k in top_ks:
                if evidence_ids.intersection(ranked_ids[:k]):
                    hits[k] += 1
            bucket = gap_bucket(probe.time_gap_days)
            bucket_row = gap_stats.setdefault(
                bucket,
                {"hits": {k: 0 for k in top_ks}, "rr": [], "queries": 0},
            )
            bucket_row["queries"] += 1
            bucket_row["rr"].append(reciprocal_rank)
            for k in top_ks:
                if evidence_ids.intersection(ranked_ids[:k]):
                    bucket_row["hits"][k] += 1
        rows.append(metric_row(model, "all_cognitive", hits, reciprocal_ranks, n_queries))
        for bucket, stats in sorted(gap_stats.items()):
            by_gap.append(metric_row(model, bucket, stats["hits"], stats["rr"], stats["queries"]))
    return {"overall": rows, "by_time_gap": by_gap}


def reciprocal_rank_for(ranked_ids: list[int], evidence_ids: set[int]) -> float:
    ranks = [rank + 1 for rank, fact_id in enumerate(ranked_ids) if fact_id in evidence_ids]
    return 0.0 if not ranks else 1.0 / min(ranks)


def metric_row(
    model: str,
    subset: str,
    hits: dict[int, int],
    reciprocal_ranks: list[float],
    n_queries: int,
) -> dict[str, Any]:
    return {
        "model": model,
        "subset": subset,
        "queries": int(n_queries),
        "recall_at_1": hits[1] / max(n_queries, 1),
        "recall_at_5": hits[5] / max(n_queries, 1),
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
    }


def gap_bucket(days: float | None) -> str:
    if days is None:
        return "gap_unknown"
    if days <= 7:
        return "gap_<=7d"
    if days <= 30:
        return "gap_8_30d"
    if days <= 180:
        return "gap_31_180d"
    return "gap_>180d"


def format_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# LoCoMo-Plus Cognitive Retrieval Probe",
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
        f"- fusion_gamma: {result['fusion_gamma']}",
        f"- dense_threshold: {result['dense_threshold']}",
        f"- max_context_messages: {result['max_context_messages']}",
        "",
        "| Model | Recall@1 | Recall@5 | MRR | Queries |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in result["retrieval"]["overall"]:
        lines.append(
            f"| `{row['model']}` | {row['recall_at_1']:.3f} | {row['recall_at_5']:.3f} | "
            f"{row['mrr']:.3f} | {row['queries']} |"
        )
    lines.extend(
        [
            "",
            "## Time-Gap Buckets",
            "",
            "| Bucket | Model | Recall@1 | Recall@5 | MRR | Queries |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result["retrieval"]["by_time_gap"]:
        lines.append(
            f"| `{row['subset']}` | `{row['model']}` | {row['recall_at_1']:.3f} | "
            f"{row['recall_at_5']:.3f} | {row['mrr']:.3f} | {row['queries']} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()

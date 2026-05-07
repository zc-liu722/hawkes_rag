from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

import numpy as np

from hawkes_rag.core import Event
from hawkes_rag.evaluation import HeldoutSplit, temporal_train_test_split
from hawkes_rag.utils import cosine_similarity


EmbeddingFn = Callable[[str], np.ndarray]


@dataclass(frozen=True)
class ConversationMessage:
    conversation_id: str
    message_id: str
    text: str
    timestamp: float
    speaker: str = ""


@dataclass(frozen=True)
class AtomicFact:
    id: int
    conversation_id: str
    text: str
    source_message_id: str
    source_time: float
    embedding: np.ndarray


@dataclass(frozen=True)
class EventizedConversation:
    conversation_id: str
    messages: list[ConversationMessage]
    facts: list[AtomicFact]
    events: list[Event]
    horizon: float

    @property
    def active_memory_ids(self) -> list[int]:
        return [fact.id for fact in self.facts]


@dataclass(frozen=True)
class EventizedCorpus:
    conversations: list[EventizedConversation]
    facts: list[AtomicFact]

    @property
    def n_memories(self) -> int:
        return len(self.facts)

    def trajectories(self) -> list[list[Event]]:
        return [conversation.events for conversation in self.conversations]

    def horizons(self) -> list[float]:
        return [conversation.horizon for conversation in self.conversations]

    def active_memory_ids(self) -> list[list[int]]:
        return [conversation.active_memory_ids for conversation in self.conversations]

    def embeddings(self) -> np.ndarray:
        if not self.facts:
            return np.zeros((0, 0), dtype=float)
        return np.vstack([fact.embedding for fact in self.facts])

    def heldout_splits(self, train_fraction: float = 0.8) -> list[HeldoutSplit]:
        return [
            temporal_train_test_split(
                conversation.events,
                conversation.horizon,
                active_memory_ids=conversation.active_memory_ids,
                train_fraction=train_fraction,
            )
            for conversation in self.conversations
        ]


class AtomicFactExtractor(Protocol):
    def extract(self, message: ConversationMessage) -> list[str]:
        ...


class ReferenceDetector(Protocol):
    def detect(self, message: ConversationMessage, candidate_facts: list[AtomicFact]) -> list[int]:
        ...


class SentenceFactExtractor:
    """Zero-dependency fact extractor used as the local design baseline.

    The paper path should replace this with an LLM/Mem0-style extractor. This
    class intentionally keeps the repo runnable without API keys.
    """

    def __init__(
        self,
        *,
        min_chars: int = 24,
        allowed_speakers: set[str] | None = None,
    ):
        self.min_chars = int(min_chars)
        self.allowed_speakers = allowed_speakers

    def extract(self, message: ConversationMessage) -> list[str]:
        if self.allowed_speakers is not None and message.speaker not in self.allowed_speakers:
            return []
        chunks = re.split(r"(?<=[.!?。！？])\s+", message.text.strip())
        facts = []
        for chunk in chunks:
            fact = chunk.strip()
            if len(fact) >= self.min_chars:
                facts.append(fact)
        return facts


class SemanticReferenceDetector:
    """Detect later mentions of known facts with embedding similarity."""

    def __init__(
        self,
        embedding_fn: EmbeddingFn,
        *,
        similarity_threshold: float = 0.42,
        lexical_overlap_threshold: float = 0.18,
    ):
        self.embedding_fn = embedding_fn
        self.similarity_threshold = float(similarity_threshold)
        self.lexical_overlap_threshold = float(lexical_overlap_threshold)

    def detect(self, message: ConversationMessage, candidate_facts: list[AtomicFact]) -> list[int]:
        if not candidate_facts:
            return []
        message_embedding = self.embedding_fn(message.text)
        hits = []
        message_tokens = set(_tokens(message.text))
        for fact in candidate_facts:
            if message.timestamp <= fact.source_time:
                continue
            sim = cosine_similarity(message_embedding, fact.embedding)
            overlap = _jaccard(message_tokens, set(_tokens(fact.text)))
            if sim >= self.similarity_threshold or overlap >= self.lexical_overlap_threshold:
                hits.append(fact.id)
        return hits


class HashingEmbedding:
    """Deterministic bag-of-words hashing embedding for reproducible examples."""

    def __init__(self, dim: int = 128):
        self.dim = int(dim)

    def __call__(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=float)
        for token in _tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector


class LoCoMoEventizer:
    """Convert LoCoMo-style conversations into Hawkes trajectories.

    Pipeline:
    1. extract atomic facts from each message
    2. emit a source activation event when the fact first appears
    3. detect future references to previous facts in the same conversation
    4. pool conversations as separate trajectories with active memory masks
    """

    def __init__(
        self,
        *,
        embedding_fn: EmbeddingFn | None = None,
        fact_extractor: AtomicFactExtractor | None = None,
        reference_detector: ReferenceDetector | None = None,
        source_event_weight: float = 1.0,
        mention_event_weight: float = 0.3,
        time_step: float = 1.0,
    ):
        self.embedding_fn = embedding_fn or HashingEmbedding()
        self.fact_extractor = fact_extractor or SentenceFactExtractor()
        self.reference_detector = reference_detector or SemanticReferenceDetector(self.embedding_fn)
        self.source_event_weight = float(source_event_weight)
        self.mention_event_weight = float(mention_event_weight)
        self.time_step = float(time_step)

    def eventize(self, conversations: Iterable[list[ConversationMessage]]) -> EventizedCorpus:
        global_facts: list[AtomicFact] = []
        eventized_conversations = []
        next_fact_id = 0
        for messages in conversations:
            eventized, next_fact_id = self.eventize_conversation(
                sorted(messages, key=lambda m: m.timestamp),
                next_fact_id=next_fact_id,
            )
            global_facts.extend(eventized.facts)
            eventized_conversations.append(eventized)
        return EventizedCorpus(conversations=eventized_conversations, facts=global_facts)

    def eventize_conversation(
        self,
        messages: list[ConversationMessage],
        *,
        next_fact_id: int = 0,
    ) -> tuple[EventizedConversation, int]:
        if not messages:
            raise ValueError("cannot eventize an empty conversation")
        conversation_id = messages[0].conversation_id
        start = messages[0].timestamp
        shifted = [
            ConversationMessage(
                conversation_id=m.conversation_id,
                message_id=m.message_id,
                text=m.text,
                timestamp=(m.timestamp - start) * self.time_step,
                speaker=m.speaker,
            )
            for m in messages
        ]

        facts: list[AtomicFact] = []
        events: list[Event] = []
        for message in shifted:
            referenced = self.reference_detector.detect(message, facts)
            for fact_id in referenced:
                events.append(
                    Event(
                        time=message.timestamp,
                        memory_id=fact_id,
                        weight=self.mention_event_weight,
                    )
                )

            for fact_text in self.fact_extractor.extract(message):
                fact = AtomicFact(
                    id=next_fact_id,
                    conversation_id=conversation_id,
                    text=fact_text,
                    source_message_id=message.message_id,
                    source_time=message.timestamp,
                    embedding=self.embedding_fn(fact_text),
                )
                next_fact_id += 1
                facts.append(fact)
                events.append(
                    Event(
                        time=message.timestamp,
                        memory_id=fact.id,
                        weight=self.source_event_weight,
                    )
                )

        events.sort(key=lambda e: e.time)
        horizon = max(message.timestamp for message in shifted) + self.time_step
        return (
            EventizedConversation(
                conversation_id=conversation_id,
                messages=shifted,
                facts=facts,
                events=events,
                horizon=float(horizon),
            ),
            next_fact_id,
        )


def load_locomo_json(path: str | Path) -> list[list[ConversationMessage]]:
    """Load common LoCoMo JSON shapes into normalized messages.

    This is intentionally permissive because benchmark mirrors often wrap the
    conversation list with slightly different keys.
    """
    data = json.loads(Path(path).read_text())
    raw_conversations = _find_conversations(data)
    conversations = []
    for index, raw in enumerate(raw_conversations):
        conversation_id = str(
            raw.get("conversation_id")
            or raw.get("sample_id")
            or raw.get("id")
            or f"conversation_{index}"
        )
        raw_messages = raw.get("messages") or raw.get("conversation") or raw.get("dialogue") or []
        messages = []
        for turn, item in enumerate(raw_messages):
            if isinstance(item, str):
                text = item
                speaker = ""
            else:
                text = str(item.get("text") or item.get("content") or item.get("message") or "")
                speaker = str(item.get("speaker") or item.get("role") or "")
            if not text.strip():
                continue
            messages.append(
                ConversationMessage(
                    conversation_id=conversation_id,
                    message_id=str(item.get("id", turn) if isinstance(item, dict) else turn),
                    text=text,
                    timestamp=float(turn),
                    speaker=speaker,
                )
            )
        if messages:
            conversations.append(messages)
    return conversations


def load_official_locomo10_json(path: str | Path) -> list[list[ConversationMessage]]:
    """Load the official snap-research/locomo `data/locomo10.json` schema.

    The official release stores each sample as:

    - `sample_id`
    - `conversation.speaker_a` / `conversation.speaker_b`
    - `conversation.session_<n>_date_time`
    - `conversation.session_<n>` as a list of turns with `speaker`, `dia_id`, `text`

    This loader is intentionally strict so benchmark runs fail fast if a mirror
    silently changes shape.
    """
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError("official LoCoMo data must be a JSON list")

    conversations: list[list[ConversationMessage]] = []
    for sample_index, sample in enumerate(data):
        if not isinstance(sample, dict):
            raise ValueError(f"sample {sample_index} must be an object")
        sample_id = sample.get("sample_id")
        conversation = sample.get("conversation")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"sample {sample_index} is missing string sample_id")
        if not isinstance(conversation, dict):
            raise ValueError(f"sample {sample_id} is missing conversation object")
        for speaker_key in ("speaker_a", "speaker_b"):
            if not isinstance(conversation.get(speaker_key), str):
                raise ValueError(f"sample {sample_id} is missing {speaker_key}")

        messages: list[ConversationMessage] = []
        session_numbers = _official_session_numbers(conversation)
        if not session_numbers:
            raise ValueError(f"sample {sample_id} has no session_<n> arrays")
        for session_position, session_number in enumerate(session_numbers):
            session_key = f"session_{session_number}"
            timestamp_key = f"{session_key}_date_time"
            turns = conversation.get(session_key)
            if not isinstance(conversation.get(timestamp_key), str):
                raise ValueError(f"sample {sample_id} is missing {timestamp_key}")
            if not isinstance(turns, list):
                raise ValueError(f"sample {sample_id} {session_key} must be a list")
            for turn_index, turn in enumerate(turns):
                if not isinstance(turn, dict):
                    raise ValueError(
                        f"sample {sample_id} {session_key}[{turn_index}] must be an object"
                    )
                speaker = turn.get("speaker")
                dia_id = turn.get("dia_id")
                text = turn.get("text")
                if not isinstance(speaker, str) or not isinstance(dia_id, str):
                    raise ValueError(
                        f"sample {sample_id} {session_key}[{turn_index}] needs speaker and dia_id"
                    )
                if not isinstance(text, str) or not text.strip():
                    continue
                messages.append(
                    ConversationMessage(
                        conversation_id=sample_id,
                        message_id=dia_id,
                        text=text,
                        timestamp=float(session_position * 1000 + turn_index),
                        speaker=speaker,
                    )
                )
        if messages:
            conversations.append(messages)
    return conversations


def _find_conversations(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if not isinstance(data, dict):
        raise ValueError("expected JSON object or list")
    for key in ["conversations", "data", "samples", "instances"]:
        value = data.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    if any(key in data for key in ["messages", "conversation", "dialogue"]):
        return [data]
    raise ValueError("could not find conversations in JSON")


def _official_session_numbers(conversation: dict[str, Any]) -> list[int]:
    numbers = []
    for key, value in conversation.items():
        match = re.fullmatch(r"session_(\d+)", key)
        if match and isinstance(value, list):
            numbers.append(int(match.group(1)))
    return sorted(numbers)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)

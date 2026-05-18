"""LoCoMo session-level retrieval helpers for R0 cosine and R1-lite OriginIdea.

This module keeps LoCoMo-specific data normalization separate from the shared
session-level dynamics used by the LongMemEval experiments.
"""

from __future__ import annotations

import re
import sys
import importlib.util
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
LONGMEMEVAL_DIR = ROOT / "benchmarks" / "longmemeval"
for path in (ROOT, LONGMEMEVAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _load_longmemeval_module(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, LONGMEMEVAL_DIR / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_lme_turns = _load_longmemeval_module("_locomo_lme_turns", "run_originidea_turns.py")
_lme_sessions = _load_longmemeval_module(
    "_locomo_lme_sessions", "run_originidea_sessions.py"
)

compute_mu = _lme_sessions.compute_mu
run_cosine_sessions = _lme_sessions.run_cosine_sessions
run_originidea_sessions = _lme_sessions.run_originidea_sessions
session_metrics_at_k = _lme_sessions.session_metrics_at_k
embed_texts = _lme_turns.embed_texts
normalize_rows = _lme_turns.normalize_rows
session_recall_at_k = _lme_turns.session_recall_at_k


@dataclass
class LocomoSession:
    session_id: str
    session_index: int
    text: str
    time: float
    is_evidence: bool = False


def parse_locomo_date(value: Any) -> float:
    text = str(value or "").strip()
    for fmt in (
        "%I:%M %p on %d %B, %Y",
        "%I:%M%p on %d %B, %Y",
        "%H:%M on %d %B, %Y",
        "%Y/%m/%d (%a) %H:%M",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(text, fmt).timestamp() / 86400.0
        except ValueError:
            continue
    return 0.0


def session_number_from_dialog_id(dialog_id: str) -> int | None:
    match = re.match(r"^D(\d+):\d+$", str(dialog_id).strip())
    if not match:
        return None
    return int(match.group(1))


def evidence_session_ids(qa: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for dialog_id in qa.get("evidence") or []:
        session_number = session_number_from_dialog_id(str(dialog_id))
        if session_number is not None:
            out.add(f"session_{session_number}")
    return out


def iter_session_numbers(conversation: dict[str, Any]) -> list[int]:
    numbers: list[int] = []
    for key, value in conversation.items():
        if not isinstance(value, list):
            continue
        match = re.match(r"^session_(\d+)$", key)
        if match:
            numbers.append(int(match.group(1)))
    return sorted(numbers)


def turn_to_text(turn: Any) -> str:
    if not isinstance(turn, dict):
        return str(turn)
    parts: list[str] = []
    speaker = str(turn.get("speaker") or "").strip()
    text = str(turn.get("text") or "").strip()
    if speaker and text:
        parts.append(f"{speaker}: {text}")
    elif text:
        parts.append(text)
    caption = str(turn.get("blip_caption") or turn.get("caption") or "").strip()
    if caption:
        parts.append(f"image_caption: {caption}")
    query = str(turn.get("query") or turn.get("search_query") or "").strip()
    if query:
        parts.append(f"search_query: {query}")
    return "\n".join(parts)


def observation_text(sample: dict[str, Any], session_number: int) -> str:
    observation = sample.get("observation") or {}
    raw = observation.get(f"session_{session_number}_observation")
    if not raw:
        return ""
    parts: list[str] = []
    if isinstance(raw, dict):
        for speaker, facts in raw.items():
            for fact in facts or []:
                if isinstance(fact, list) and fact:
                    text = str(fact[0])
                    evidence = f" [{fact[1]}]" if len(fact) > 1 else ""
                    parts.append(f"{speaker}: {text}{evidence}")
                else:
                    parts.append(f"{speaker}: {fact}")
    else:
        parts.append(str(raw))
    return "\n".join(parts)


def session_text(sample: dict[str, Any], session_number: int, mode: str) -> str:
    conversation = sample.get("conversation") or {}
    if mode == "session_summary":
        summaries = sample.get("session_summary") or {}
        return str(summaries.get(f"session_{session_number}_summary") or "").strip()
    if mode == "observation":
        return observation_text(sample, session_number).strip()

    turns = conversation.get(f"session_{session_number}") or []
    parts = [turn_to_text(turn) for turn in turns]
    return "\n".join(p for p in parts if p.strip()).strip()


def expand_sessions(sample: dict[str, Any], qa: dict[str, Any], *,
                    memory_text_mode: str = "dialog") -> list[LocomoSession]:
    conversation = sample.get("conversation") or {}
    gold_session_ids = evidence_session_ids(qa)
    sessions: list[LocomoSession] = []
    for number in iter_session_numbers(conversation):
        sid = f"session_{number}"
        text = session_text(sample, number, memory_text_mode)
        if not text:
            continue
        t = parse_locomo_date(conversation.get(f"session_{number}_date_time"))
        if t <= 0:
            t = float(number)
        sessions.append(
            LocomoSession(
                session_id=sid,
                session_index=number,
                text=text,
                time=t,
                is_evidence=sid in gold_session_ids,
            )
        )
    sessions.sort(key=lambda s: (s.time, s.session_index))
    return sessions


def answer_lag_days(sessions: list[LocomoSession], question_time: float) -> float:
    gold_times = [s.time for s in sessions if s.is_evidence]
    if not gold_times:
        return 0.0
    return max(0.0, question_time - max(gold_times))


def gold_session_span_days(sessions: list[LocomoSession]) -> float:
    gold_times = [s.time for s in sessions if s.is_evidence]
    if len(gold_times) < 2:
        return 0.0
    return max(gold_times) - min(gold_times)


__all__ = [
    "LocomoSession",
    "answer_lag_days",
    "compute_mu",
    "embed_texts",
    "evidence_session_ids",
    "expand_sessions",
    "gold_session_span_days",
    "normalize_rows",
    "parse_locomo_date",
    "run_cosine_sessions",
    "run_originidea_sessions",
    "session_metrics_at_k",
    "session_recall_at_k",
]

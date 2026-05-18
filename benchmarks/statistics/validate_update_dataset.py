#!/usr/bin/env python3
"""Validate exchange-level temporal memory retrieval scenarios.

In this dataset, one turn means one complete back-and-forth exchange:
two messages, one from each participant. It is not a single sentence or
single-speaker utterance.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ALLOWED_CATEGORIES = {
    "update_override",
    "decay_forget",
    "reactivation",
    "semantic_distractor",
    "stability_check",
}

LEGACY_FIELDS = {
    "scenario_type",
    "subtype",
    "facts_added",
    "facts_updated",
    "fact_id",
    "subject",
    "attr",
    "value",
    "memory_role",
    "importance",
    "updated_by",
    "query_type",
    "reinforced_fact_ids",
    "expected_answer",
    "expected_fact_ids",
    "forbidden_fact_ids",
    "tests",
}

EXPERIMENT_TERMS = ("λ", "cos 相似度", "召回", "评测", "Hawkes", "recency")
PLACEHOLDER_PATTERNS = ("【待填充",)
DEFAULT_TOP_K = (1, 3, 5)
MIN_TURNS = 42
MAX_TURNS = 64


@dataclass
class Issue:
    path: Path
    severity: str
    message: str

    def format(self) -> str:
        return f"{self.path}: [{self.severity}] {self.message}"


def add(issues: list[Issue], path: Path, severity: str, message: str) -> None:
    issues.append(Issue(path=path, severity=severity, message=message))


def parse_ts(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def iter_json_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.json")))
        else:
            files.append(path)
    return files


def find_bad_strings(value: Any, prefix: str = "$") -> list[tuple[str, str, str]]:
    found: list[tuple[str, str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            found.extend(find_bad_strings(child, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(find_bad_strings(child, f"{prefix}[{index}]"))
    elif isinstance(value, str):
        for marker in PLACEHOLDER_PATTERNS:
            if marker in value:
                found.append((prefix, "fatal", f"placeholder marker remains at {prefix}"))
        for term in EXPERIMENT_TERMS:
            if term in value:
                found.append((prefix, "warning", f"experiment term {term!r} appears at {prefix}"))
    return found


def find_legacy_fields(value: Any, prefix: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{prefix}.{key}"
            if key in LEGACY_FIELDS:
                found.append(child_path)
            found.extend(find_legacy_fields(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(find_legacy_fields(child, f"{prefix}[{index}]"))
    return found


def validate_message(
    issues: list[Issue],
    path: Path,
    message: Any,
    turn_idx: int,
    message_idx: int,
) -> str | None:
    if not isinstance(message, dict):
        add(issues, path, "fatal", f"turn {turn_idx} message {message_idx}: must be an object")
        return None

    speaker = message.get("speaker")
    if speaker not in {"user1", "user2"}:
        add(
            issues,
            path,
            "fatal",
            f"turn {turn_idx} message {message_idx}: speaker must be user1 or user2",
        )

    text = message.get("text")
    if not isinstance(text, str) or not text.strip():
        add(issues, path, "fatal", f"turn {turn_idx} message {message_idx}: text is required")

    return speaker if isinstance(speaker, str) else None


def validate_turn(
    issues: list[Issue],
    path: Path,
    turn: Any,
    expected_idx: int,
    timestamps: list[datetime],
) -> None:
    if not isinstance(turn, dict):
        add(issues, path, "fatal", f"turn {expected_idx}: must be an object")
        return

    idx = turn.get("idx")
    if idx != expected_idx:
        add(issues, path, "fatal", f"turn idx must be {expected_idx}, got {idx!r}")

    if "speaker" in turn or "text" in turn:
        add(
            issues,
            path,
            "fatal",
            f"turn {expected_idx}: top-level speaker/text is the old single-message format; use messages",
        )

    messages = turn.get("messages")
    if not isinstance(messages, list):
        add(issues, path, "fatal", f"turn {expected_idx}: messages must be an array")
    elif len(messages) != 2:
        add(issues, path, "fatal", f"turn {expected_idx}: messages must contain exactly 2 items")
    else:
        speakers = [
            validate_message(issues, path, message, expected_idx, message_idx)
            for message_idx, message in enumerate(messages)
        ]
        if set(speakers) != {"user1", "user2"}:
            add(issues, path, "fatal", f"turn {expected_idx}: messages must include user1 and user2 once each")

    timestamp = parse_ts(turn.get("t"))
    if timestamp is None:
        add(issues, path, "fatal", f"turn {expected_idx}: invalid t timestamp")
    else:
        if timestamps and timestamp <= timestamps[-1]:
            add(issues, path, "fatal", f"turn {expected_idx}: timestamp must be strictly increasing")
        timestamps.append(timestamp)

    tags = turn.get("tags")
    if tags is not None:
        if not isinstance(tags, list) or not all(isinstance(tag, str) and tag for tag in tags):
            add(issues, path, "fatal", f"turn {expected_idx}: tags must be an array of non-empty strings")


def validate_int_list(
    issues: list[Issue],
    path: Path,
    value: Any,
    field: str,
    eval_idx: int,
) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        add(issues, path, "fatal", f"eval {eval_idx}: {field} must be an array of integers")
        return []
    return list(value)


def validate_eval(
    issues: list[Issue],
    path: Path,
    ev: Any,
    eval_idx: int,
    category: str,
    n_turns: int,
) -> None:
    if not isinstance(ev, dict):
        add(issues, path, "fatal", f"eval {eval_idx}: must be an object")
        return

    query_turn = ev.get("query_turn")
    if not isinstance(query_turn, int):
        add(issues, path, "fatal", f"eval {eval_idx}: query_turn is required")
        return
    if query_turn <= 0 or query_turn >= n_turns:
        add(issues, path, "fatal", f"eval {eval_idx}: query_turn must be between 1 and {n_turns - 1}")

    ev_type = ev.get("type", category)
    if ev_type not in ALLOWED_CATEGORIES:
        add(issues, path, "fatal", f"eval {eval_idx}: invalid type {ev_type!r}")

    positive = validate_int_list(issues, path, ev.get("positive_turns"), "positive_turns", eval_idx)
    negative = validate_int_list(issues, path, ev.get("negative_turns", []), "negative_turns", eval_idx)
    top_k = validate_int_list(issues, path, ev.get("top_k", list(DEFAULT_TOP_K)), "top_k", eval_idx)

    if not positive:
        add(issues, path, "fatal", f"eval {eval_idx}: positive_turns must not be empty")
    if category in {"update_override", "semantic_distractor"} and not negative:
        add(issues, path, "fatal", f"eval {eval_idx}: {category} requires negative_turns")
    if not top_k:
        add(issues, path, "warning", f"eval {eval_idx}: top_k empty; default [1, 3, 5] will be used")

    for field, values in (("positive_turns", positive), ("negative_turns", negative)):
        for idx in values:
            if idx < 0 or idx >= n_turns:
                add(issues, path, "fatal", f"eval {eval_idx}: {field} references out-of-range turn {idx}")
            elif idx >= query_turn:
                add(issues, path, "fatal", f"eval {eval_idx}: {field} must reference turns before query_turn")

    overlap = set(positive) & set(negative)
    if overlap:
        add(issues, path, "fatal", f"eval {eval_idx}: positive_turns and negative_turns overlap: {sorted(overlap)}")

    pairs = ev.get("pairs", [])
    if pairs is None:
        pairs = []
    if not isinstance(pairs, list):
        add(issues, path, "fatal", f"eval {eval_idx}: pairs must be an array")
        pairs = []
    for pair_idx, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            add(issues, path, "fatal", f"eval {eval_idx} pair {pair_idx}: must be an object")
            continue
        pos = pair.get("positive")
        neg = pair.get("negative")
        if not isinstance(pos, int) or not isinstance(neg, int):
            add(issues, path, "fatal", f"eval {eval_idx} pair {pair_idx}: positive and negative must be integers")
            continue
        if pos not in positive:
            add(issues, path, "warning", f"eval {eval_idx} pair {pair_idx}: positive {pos} not listed in positive_turns")
        if neg not in negative:
            add(issues, path, "warning", f"eval {eval_idx} pair {pair_idx}: negative {neg} not listed in negative_turns")
        if pos >= query_turn or neg >= query_turn:
            add(issues, path, "fatal", f"eval {eval_idx} pair {pair_idx}: pair turns must precede query_turn")

    if category in {"update_override", "semantic_distractor"} and not pairs:
        add(issues, path, "warning", f"eval {eval_idx}: {category} should include at least one positive-vs-negative pair")


def validate_scenario(path: Path) -> list[Issue]:
    issues: list[Issue] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return [Issue(path, "fatal", f"cannot parse JSON: {exc}")]

    if not isinstance(data, dict):
        return [Issue(path, "fatal", "top-level JSON must be an object")]

    for _, severity, message in find_bad_strings(data):
        add(issues, path, severity, message)
    for legacy_path in find_legacy_fields(data):
        add(issues, path, "fatal", f"legacy field is no longer allowed: {legacy_path}")

    scenario_id = data.get("scenario_id")
    if not isinstance(scenario_id, str) or not scenario_id:
        add(issues, path, "fatal", "scenario_id is required")
    elif not re.match(r"^[a-z]+_[A-E]_[a-z0-9_]+_\d{3}$", scenario_id):
        add(issues, path, "warning", "scenario_id does not match recommended naming pattern")

    category = data.get("category")
    if category not in ALLOWED_CATEGORIES:
        add(issues, path, "fatal", f"category must be one of {sorted(ALLOWED_CATEGORIES)}")
        category = "unknown"

    persona = data.get("persona")
    if persona is not None and not isinstance(persona, (str, dict)):
        add(issues, path, "fatal", "persona must be a string or object when present")

    turns = data.get("turns")
    if not isinstance(turns, list):
        add(issues, path, "fatal", "turns must be an array")
        return issues
    if not MIN_TURNS <= len(turns) <= MAX_TURNS:
        add(
            issues,
            path,
            "fatal",
            f"turns length must be between {MIN_TURNS} and {MAX_TURNS}, got {len(turns)}",
        )

    timestamps: list[datetime] = []
    for expected_idx, turn in enumerate(turns):
        validate_turn(
            issues,
            path,
            turn,
            expected_idx,
            timestamps,
        )

    evals = data.get("evals")
    if not isinstance(evals, list) or not evals:
        add(issues, path, "fatal", "evals must be a non-empty array")
        return issues
    for eval_idx, ev in enumerate(evals):
        validate_eval(issues, path, ev, eval_idx, str(category), len(turns))

    return issues


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--warnings-as-errors", action="store_true")
    args = parser.parse_args(argv)

    files = iter_json_files(args.paths)
    if not files:
        print("No JSON files found.", file=sys.stderr)
        return 2

    all_issues: list[Issue] = []
    for file in files:
        all_issues.extend(validate_scenario(file))

    for issue in all_issues:
        print(issue.format())

    fatal_count = sum(issue.severity == "fatal" for issue in all_issues)
    warning_count = sum(issue.severity == "warning" for issue in all_issues)
    print(f"Checked {len(files)} file(s): {fatal_count} fatal, {warning_count} warning")

    if fatal_count or (args.warnings_as_errors and warning_count):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

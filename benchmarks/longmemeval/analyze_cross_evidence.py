from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hawkes_rag.embeddings import make_embedding_fn


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze whether LongMemEval multi-session questions contain the "
            "semantic-thin/cross-evidence signal needed by cross-excitation."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("benchmarks/longmemeval/cache/longmemeval_s.json"),
    )
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--embedding",
        choices=["minilm", "bge", "hashing"],
        default="minilm",
        help="Use minilm/bge for the real semantic analysis; hashing is only a smoke fallback.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=Path("benchmarks/longmemeval/cache/models"),
    )
    parser.add_argument("--query-threshold", type=float, default=0.4)
    parser.add_argument("--evidence-threshold", type=float, default=0.6)
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Optional smoke limit over all records before filtering; 0 means full dataset.",
    )
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(
            f"Missing {args.data}. Run `python3 benchmarks/longmemeval/download.py` "
            "or place longmemeval_s.json there."
        )

    records = load_records(args.data)
    if args.max_records > 0:
        records = records[: args.max_records]

    embed = make_embedding_fn(
        args.embedding,
        device=args.device,
        cache_dir=args.model_cache_dir if args.embedding != "hashing" else None,
    )

    multi_records = [record for record in records if is_multi_session(record)]
    hop_distribution: Counter[str] = Counter()
    per_question: list[dict[str, Any]] = []
    semantic_thin_count = 0
    sweet_spot_count = 0
    cross_signal_count = 0

    for record in multi_records:
        question_id = str(record.get("question_id", ""))
        query = str(record.get("question", ""))
        evidence_sessions = extract_evidence_sessions(record)
        hop = len(evidence_sessions)
        hop_distribution[hop_bucket(hop)] += 1

        if hop == 0:
            per_question.append(
                {
                    "question_id": question_id,
                    "hop": 0,
                    "answer_session_ids": evidence_session_ids(record),
                    "query_to_evidence_cosines": [],
                    "evidence_pair_cosines": [],
                    "semantic_thin_all_query_cosines_below_threshold": False,
                    "has_cross_evidence_signal": False,
                    "sweet_spot": False,
                    "warning": "no answer_session_ids could be matched to haystack_session_ids",
                }
            )
            continue

        query_vec = normalize(embed(query))
        evidence_vecs = [normalize(embed(session["text"])) for session in evidence_sessions]
        query_cosines = [float(np.dot(query_vec, vec)) for vec in evidence_vecs]

        pair_records: list[dict[str, Any]] = []
        pair_cosines: dict[tuple[int, int], float] = {}
        for left, right in combinations(range(hop), 2):
            cosine = float(np.dot(evidence_vecs[left], evidence_vecs[right]))
            pair_cosines[(left, right)] = cosine
            pair_records.append(
                {
                    "left_session_id": evidence_sessions[left]["session_id"],
                    "right_session_id": evidence_sessions[right]["session_id"],
                    "cosine": round(cosine, 6),
                }
            )

        semantic_thin = all(value < args.query_threshold for value in query_cosines)
        has_cross_signal = any(value > args.evidence_threshold for value in pair_cosines.values())
        sweet_spot = has_sweet_spot(query_cosines, pair_cosines, args.query_threshold, args.evidence_threshold)

        semantic_thin_count += int(semantic_thin)
        cross_signal_count += int(has_cross_signal)
        sweet_spot_count += int(sweet_spot)

        per_question.append(
            {
                "question_id": question_id,
                "question": query,
                "answer": record.get("answer"),
                "hop": hop,
                "answer_session_ids": [session["session_id"] for session in evidence_sessions],
                "query_to_evidence_cosines": [
                    {
                        "session_id": evidence_sessions[index]["session_id"],
                        "cosine": round(value, 6),
                    }
                    for index, value in enumerate(query_cosines)
                ],
                "query_cosine_min": round(min(query_cosines), 6),
                "query_cosine_max": round(max(query_cosines), 6),
                "query_cosine_mean": round(float(np.mean(query_cosines)), 6),
                "evidence_pair_cosines": pair_records,
                "evidence_pair_cosine_max": round(max(pair_cosines.values()), 6) if pair_cosines else None,
                "semantic_thin_all_query_cosines_below_threshold": semantic_thin,
                "has_cross_evidence_signal": has_cross_signal,
                "sweet_spot": sweet_spot,
            }
        )

    verdict = "go" if sweet_spot_count >= 30 else "stop" if sweet_spot_count < 10 else "borderline"
    result = {
        "data": str(args.data),
        "embedding": args.embedding,
        "thresholds": {
            "query_cosine_lt": args.query_threshold,
            "evidence_pair_cosine_gt": args.evidence_threshold,
        },
        "total_records": len(records),
        "multi_session_size": len(multi_records),
        "hop_distribution": dict(sorted(hop_distribution.items())),
        "semantic_thin_all_query_cosines_below_threshold_count": semantic_thin_count,
        "cross_evidence_signal_count": cross_signal_count,
        "sweet_spot_count": sweet_spot_count,
        "go_no_go": verdict,
        "per_question": per_question,
    }

    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.outputs_dir / "longmemeval_cross_evidence_analysis.json"
    md_path = args.outputs_dir / "longmemeval_cross_evidence_analysis.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(result), encoding="utf-8")

    print(render_console(result))
    print(f"json={json_path}")
    print(f"markdown={md_path}")


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"expected list in {path}")
        return payload
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def extract_evidence_sessions(record: dict[str, Any]) -> list[dict[str, str]]:
    session_ids = [str(value) for value in record.get("haystack_session_ids") or []]
    sessions = record.get("haystack_sessions") or []
    id_to_index = {session_id: index for index, session_id in enumerate(session_ids)}
    evidence_sessions: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw_session_id in evidence_session_ids(record):
        session_id = str(raw_session_id)
        index = id_to_index.get(session_id)
        if index is None or index >= len(sessions) or session_id in seen:
            continue
        seen.add(session_id)
        evidence_sessions.append({"session_id": session_id, "text": session_to_text(sessions[index])})
    return evidence_sessions


def is_multi_session(record: dict[str, Any]) -> bool:
    question_type = str(record.get("question_type", "")).lower().replace("_", "-")
    if question_type == "multi-session":
        return True
    return len(evidence_session_ids(record)) > 1


def evidence_session_ids(record: dict[str, Any]) -> list[str]:
    for key in ("answer_session_ids", "evidence_session_ids", "evidence_sessions"):
        values = record.get(key)
        if not values:
            continue
        session_ids: list[str] = []
        for value in values:
            if isinstance(value, dict):
                raw = value.get("session_id") or value.get("id")
            else:
                raw = value
            if raw is not None:
                session_ids.append(str(raw))
        if session_ids:
            return session_ids
    return []


def session_to_text(session: Any) -> str:
    if isinstance(session, str):
        return session
    if not isinstance(session, list):
        return json.dumps(session, ensure_ascii=False, sort_keys=True)
    chunks: list[str] = []
    for turn in session:
        if isinstance(turn, dict):
            role = str(turn.get("role", ""))
            content = str(turn.get("content", ""))
            if role and content:
                chunks.append(f"{role}: {content}")
            elif content:
                chunks.append(content)
        else:
            chunks.append(str(turn))
    return "\n".join(chunks)


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm == 0 or not math.isfinite(norm):
        return vector
    return vector / norm


def hop_bucket(hop: int) -> str:
    if hop <= 1:
        return "1-hop"
    if hop == 2:
        return "2-hop"
    if hop == 3:
        return "3-hop"
    return "4+-hop"


def has_sweet_spot(
    query_cosines: list[float],
    pair_cosines: dict[tuple[int, int], float],
    query_threshold: float,
    evidence_threshold: float,
) -> bool:
    for index, query_cosine in enumerate(query_cosines):
        if query_cosine >= query_threshold:
            continue
        for (left, right), pair_cosine in pair_cosines.items():
            if index in {left, right} and pair_cosine > evidence_threshold:
                return True
    return False


def render_console(result: dict[str, Any]) -> str:
    lines = [
        "LongMemEval cross-evidence analysis",
        f"multi_session_size={result['multi_session_size']}",
        f"hop_distribution={result['hop_distribution']}",
        (
            "semantic_thin_all_query_cosines_below_threshold_count="
            f"{result['semantic_thin_all_query_cosines_below_threshold_count']}"
        ),
        f"cross_evidence_signal_count={result['cross_evidence_signal_count']}",
        f"sweet_spot_count={result['sweet_spot_count']}",
        f"go_no_go={result['go_no_go']}",
    ]
    return "\n".join(lines)


def render_markdown(result: dict[str, Any]) -> str:
    top_sweet = [item for item in result["per_question"] if item.get("sweet_spot")][:20]
    lines = [
        "# LongMemEval Cross-Evidence Analysis",
        "",
        f"- data: `{result['data']}`",
        f"- embedding: `{result['embedding']}`",
        f"- multi-session subset size: **{result['multi_session_size']}**",
        f"- hop distribution: `{result['hop_distribution']}`",
        (
            "- semantic-thin count, all query-to-evidence cosines below threshold: "
            f"**{result['semantic_thin_all_query_cosines_below_threshold_count']}**"
        ),
        f"- cross-evidence signal count: **{result['cross_evidence_signal_count']}**",
        f"- sweet-spot count: **{result['sweet_spot_count']}**",
        f"- go/no-go: **{result['go_no_go'].upper()}**",
        "",
        "## Sweet-Spot Examples",
        "",
        "| question_id | hop | min q-e cosine | max e-e cosine |",
        "| --- | ---: | ---: | ---: |",
    ]
    for item in top_sweet:
        lines.append(
            "| {question_id} | {hop} | {query_cosine_min} | {evidence_pair_cosine_max} |".format(
                **item
            )
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()

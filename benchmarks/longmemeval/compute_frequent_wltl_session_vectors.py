"""Compute session-sum vectors and dot products for frequent W/L sweep question IDs.

Loads LongMemEval-S JSON, embeds sessions with Qwen3 embedding (L2-normalized rows, then sum),
matching run_originidea_sessions / sweep preset.
Writes markdown tables under outputs/longmemeval_originidea_sweep/.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.longmemeval.run_originidea_sessions import Session, expand_sessions  # noqa: E402
from benchmarks.longmemeval.run_originidea_turns import embed_texts  # noqa: E402
from hawkes_rag.embeddings import make_embedding_fn  # noqa: E402

DATA_PATH = ROOT / "benchmarks/longmemeval/cache/longmemeval_s.json"
OUT_MD = ROOT / "outputs/longmemeval_originidea_sweep/frequent_win_lose_session_geometry.md"


@dataclass
class QC:
    name: str
    type_key: str
    cat_avg_interval_benchmark: float
    win_ids: list[str]
    lose_ids: list[str]


def avg_interval_dates_only(record: dict) -> float:
    dates = list(record.get("haystack_dates") or [])
    if len(dates) < 2:
        return float("nan")
    from benchmarks.longmemeval.run_originidea_turns import parse_date

    t = np.array([parse_date(d) for d in dates], dtype=float)
    order = np.argsort(t)
    t = t[order]
    return float(np.diff(t).mean())


def question_sessions_texts(record: dict) -> tuple[list[float], list[str]]:
    """Times and texts aligned to embeddable sessions (non-empty merged text)."""
    sess = expand_sessions(record)
    times = [s.time for s in sess]
    texts = [s.text for s in sess]
    return times, texts


def format_vec_inline(v: np.ndarray, *, _decimals: int = 4) -> str:
    a = np.asarray(v, dtype=float).ravel()
    parts = ",".join(f"{float(x):.{_decimals}f}" for x in a)
    return f"[{parts}]"


def main() -> None:
    benchmarks = [
        QC(
            "multi-session",
            "multi-session",
            0.17,
            [
                "d23cf73b",
                "gpt4_ab202e7f",
                "5025383b",
                "gpt4_59c863d7",
                "c2ac3c61",
                "61f8c8f8",
                "92a0aa75",
                "28dc39ac",
                "60bf93ed_abs",
                "6d550036",
                "gpt4_2ba83207",
                "81507db6",
            ],
            [
                "gpt4_194be4b3",
                "73d42213",
                "gpt4_31ff4165",
                "80ec1f4f_abs",
                "d3ab962e",
                "1192316e",
                "80ec1f4f",
                "bc149d6b",
            ],
        ),
        QC(
            "temporal-reasoning",
            "temporal-reasoning",
            0.57,
            [
                "0bc8ad92",
                "gpt4_f420262c",
                "gpt4_68e94288",
                "gpt4_8279ba03",
                "gpt4_1e4a8aec",
                "gpt4_fa19884d",
                "bbf86515",
                "gpt4_d6585ce8",
            ],
            [
                "gpt4_45189cb4",
                "gpt4_7abb270c",
                "gpt4_e061b84g",
                "b46e15ee",
                "gpt4_4929293b",
                "9a707b82",
            ],
        ),
        QC(
            "knowledge-update",
            "knowledge-update",
            1.98,
            ["d7c942c3", "0f05491a", "59524333"],
            ["2698e78f", "ce6d2d27", "08e075c7", "89941a93", "9bbe84a2"],
        ),
    ]

    print("loading json...", flush=True)
    records: list[dict] = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    by_id: dict[str, dict] = {str(r["question_id"]): r for r in records}

    embed_fn = make_embedding_fn("qwen", device="auto")
    emb_dim = int(np.asarray(embed_fn("."), dtype=float).shape[0])

    md_lines: list[str] = [
        "# Frequent sweep win/lose · session vector geometry\n",
        "",
        "说明：题号取自 `sweep_all_multisession_temporalreasoning_knowledgeupdate.md`（≥3 次出现于赢列或输列）。"
        " Session 向量使用 `Qwen/Qwen3-Embedding-0.6B`，**每条 session 先做 L2 归一**，再对各 session 向量求和得到 `sum_session`。"
        " `sum_question_session` 为该大类 **全部题目**（非仅频繁题）各自的 `sum_session` 再做矢量和。"
        " 时间与间隔与 `dataset_stats.md` 一致：`相邻session平均间隔` 按 `haystack_dates` 全局排序后相邻差再平均（天）；"
        " 括号内为该类型数据集均值（benchmark）。",
        "",
    ]

    for qc in benchmarks:
        cat_records = [r for r in records if r.get("question_type") == qc.type_key]
        assert cat_records

        md_lines.extend(
            [
                f"## {qc.name}\n",
                "",
                f"*数据集该类（`dataset_stats.md`）：相邻 session 平均间隔（类型均值）≈ **{qc.cat_avg_interval_benchmark}** 天*",
                "",
            ]
        )

        sum_by_qid: dict[str, np.ndarray] = {}
        qvec_by_qid: dict[str, np.ndarray] = {}
        interval_dates: dict[str, float] = {}
        interval_sess_time: dict[str, float] = {}
        n_sess: dict[str, int] = {}

        print(f"embedding all sessions in category {qc.type_key} ({len(cat_records)} questions)...", flush=True)
        for r in cat_records:
            qid = str(r["question_id"])
            interval_dates[qid] = avg_interval_dates_only(r)
            times, texts = question_sessions_texts(r)
            n_sess[qid] = len(texts)
            if len(times) >= 2:
                tt = np.array(times, dtype=float)
                interval_sess_time[qid] = float(np.diff(tt).mean())
            else:
                interval_sess_time[qid] = float("nan")

            if not texts:
                sum_by_qid[qid] = np.zeros(emb_dim, dtype=float)
            else:
                mat = embed_texts(embed_fn, texts, batch_size=32)
                sum_by_qid[qid] = mat.sum(axis=0)

            qtxt = str(r.get("question", "") or "").strip()
            qvec_by_qid[qid] = np.asarray(embed_fn(qtxt or "."), dtype=float)

        agg = np.stack([sum_by_qid[str(r["question_id"])] for r in cat_records], axis=0).sum(axis=0)

        targets: list[tuple[str, str, int]] = []
        for qid in qc.win_ids:
            targets.append((qid, "经常赢（≥3 次出现于赢列）", qc.win_ids.index(qid) + 1))
        for qid in qc.lose_ids:
            targets.append((qid, "经常输（≥3 次出现于输列）", qc.lose_ids.index(qid) + 1))

        for qid, label, idx_in_group in targets:
            if qid not in by_id:
                md_lines.append(f"### 缺失题库：{qid}\n")
                continue
            r = by_id[qid]
            qvec = qvec_by_qid[qid]
            ssum = sum_by_qid[qid]
            md_lines.extend(
                [
                    f"### {label} · `{qid}`（该类内枚举 #{idx_in_group}）",
                    "",
                    "**本题 session 侧写**（`dataset_stats.md` 语义）：",
                    f"相邻 session 平均间隔（按 `haystack_dates` 排序）= **{interval_dates[qid]:.6f}** 天"
                    f" · 数据集该类均值 ≈ **{qc.cat_avg_interval_benchmark}** 天 · "
                    f"可回放 session 条数 **{n_sess[qid]}** · "
                    f"（回放时间轴相邻间隔均值 = {interval_sess_time[qid]:.6f} 天，应与上式接近）",
                    "",
                    "| quantity | value |",
                    "|---|---|",
                    "| question_id | `{}` |".format(qid),
                    "| 平均相邻 session 间隔（天，`haystack_dates`） | {:.6f} |".format(
                        interval_dates[qid]
                    ),
                    "| `sum_session` 维度 | {} |".format(ssum.shape[0]),
                    "| `‖sum_session‖` | {:.6f} |".format(float(np.linalg.norm(ssum))),
                    "| `query · sum_session` | {:.6f} |".format(float(qvec @ ssum)),
                    "| `‖sum_question_session‖` · 维 | {:.6f} · {} |".format(
                        float(np.linalg.norm(agg)), agg.shape[0]
                    ),
                    "| `sum_session · sum_question_session` | {:.6f} |".format(
                        float(ssum @ agg)
                    ),
                    "",
                    "**`sum_session` 坐标**（`Qwen/Qwen3-Embedding-0.6B`，单位向量之和，未再做归一）：",
                    "",
                    "```text",
                    format_vec_inline(ssum),
                    "```",
                    "",
                    "**`sum_question_session` 坐标**（该类型共 {} 题矢量和）：".format(len(cat_records)),
                    "",
                    "```text",
                    format_vec_inline(agg),
                    "```",
                    "",
                    "---",
                    "",
                ]
            )

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"wrote {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()

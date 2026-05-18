"""逐 session 打印 originidea 动力学的 trace：H_hat / μ / λ 分布等。

参考 benchmarks/longmemeval/trace_one_question.py 的打印风格，但严格按
originidea.md (session 版) 的公式：
  score_i = cos(q_i, m) · [μ + (1-μ)·λ^-(t_i)]
  μ = μ_base + (1-μ_base)·√(1-Ĥ),  Ĥ = -Σ p ln p / lnN,  p_m = λ²/Σλ²
  λ^+ = λ^- + (1-λ^-)·score_i
  λ(t) = λ^+·exp(-β·Δt)

用法：
  python3 trace_originidea_one_question.py [--qid <question_id>] \
    [--T-half 14] [--mu-base 0.1] [--inter-top-k 3] [--top-k 5]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hawkes_rag.embeddings import make_embedding_fn  # noqa: E402
from benchmarks.longmemeval.run_originidea_turns import (  # noqa: E402
    embed_texts,
    evidence_session_ids,
    is_multi_session,
    normalize_rows,
    parse_date,
)
from benchmarks.longmemeval.run_originidea_sessions import (  # noqa: E402
    Session,
    compute_mu,
    expand_sessions,
)


def fmt_ts(t_days: float) -> str:
    return datetime.fromtimestamp(t_days * 86400.0).strftime("%Y-%m-%d %H:%M")


def short(text: str, width: int = 70) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= width else text[: width - 1] + "…"


def h_hat(lambdas: np.ndarray) -> tuple[float, float, int]:
    """返回 (Ĥ, H_raw_nats, N_used)。"""
    n = lambdas.size
    if n <= 1:
        return 0.0, 0.0, n
    lam2 = lambdas ** 2
    total = float(lam2.sum())
    if total <= 0.0:
        return 0.0, 0.0, n
    p = lam2 / total
    nz = p > 0
    h = -float(np.sum(p[nz] * np.log(p[nz])))
    return h / math.log(n), h, n


def quantile_str(arr: np.ndarray) -> str:
    if arr.size == 0:
        return "(empty)"
    qs = np.quantile(arr, [0.1, 0.25, 0.5, 0.75, 0.9])
    return (
        f"q10={qs[0]:.4f} q25={qs[1]:.4f} q50={qs[2]:.4f} "
        f"q75={qs[3]:.4f} q90={qs[4]:.4f} (n={arr.size})"
    )


def trace_question(
    record: dict,
    embed_fn,
    *,
    beta: float,
    mu_base: float,
    inter_top_k: int,
    final_top_k: int,
    top_k_show: int,
) -> None:
    sessions: list[Session] = expand_sessions(record)
    if not sessions:
        print("[skip] no usable sessions")
        return
    answer_ids = evidence_session_ids(record)
    question = str(record.get("question", ""))
    question_time = parse_date(record.get("question_date"))
    if question_time <= 0:
        question_time = max(s.time for s in sessions) + 1.0

    n = len(sessions)
    sids = [s.session_id for s in sessions]
    s_times = np.asarray([s.time for s in sessions], dtype=float)
    snippets = [short(s.text, 70) for s in sessions]
    is_evi = [s.session_id in answer_ids for s in sessions]

    print("=" * 100)
    print(f"question_id : {record.get('question_id')}")
    print(f"question    : {short(question, 120)}")
    print(f"question_ts : {fmt_ts(question_time)}")
    print(f"answer_ids  : {sorted(answer_ids)}")
    print(f"num_sessions: {n}")
    T_half = math.log(2.0) / beta if beta > 0 else float("inf")
    print(
        f"hyperparams : beta={beta:.5f}/day  (T_half={T_half:.3f}d)  "
        f"mu_base={mu_base}  inter_top_k={inter_top_k}  final_top_k={final_top_k}"
    )
    print("=" * 100)

    texts = [s.text for s in sessions]
    sess_vec = normalize_rows(embed_texts(embed_fn, texts, batch_size=64))
    q_vec = embed_texts(embed_fn, [question], batch_size=1)[0]
    q_vec = q_vec / max(float(np.linalg.norm(q_vec)), 1e-12)

    lambdas = np.zeros(n, dtype=float)
    last_update = np.zeros(n, dtype=float)
    last_call_time = np.zeros(n, dtype=float)
    created = np.zeros(n, dtype=bool)

    print("\n# 时间序列回放（每条 session 既是查询又是新记忆）")
    print("# 列: ev_idx | t | Ĥ | μ | N_pool | λ⁻ 分布 | top-k 命中情况")

    for i in range(n):
        t_i = float(s_times[i])
        marker_in = "*" if is_evi[i] else " "
        header = (
            f"\n[event {i+1:>2}/{n}] t={fmt_ts(t_i)}  incoming={marker_in}{sids[i]}"
            f"   text: {snippets[i]}"
        )
        print(header)

        if not created.any():
            print("           (no prior memories)")
        else:
            idx_global = np.flatnonzero(created)
            decayed = lambdas[idx_global] * np.exp(
                -beta * (t_i - last_update[idx_global])
            )
            decayed = np.clip(decayed, 0.0, 1.0)
            mu = compute_mu(decayed, mu_base)
            h_norm, h_nats, N_used = h_hat(decayed)

            cos_vec = sess_vec[idx_global] @ sess_vec[i]
            mix_vec = mu + (1.0 - mu) * decayed
            scores = cos_vec * mix_vec

            print(
                f"           Ĥ={h_norm:.4f}  H={h_nats:.4f} nats  ln(N)={math.log(max(N_used,1)):.4f}  "
                f"μ={mu:.4f}  μ_base={mu_base}  √(1-Ĥ)={math.sqrt(max(1-h_norm,0)):.4f}"
            )
            pos_mask = decayed > 0
            print(
                f"           pool={len(idx_global)}  λ⁻>0 count={int(pos_mask.sum())}  "
                f"λ⁻ stats: {quantile_str(decayed)}"
            )
            if pos_mask.any():
                lam_pos = decayed[pos_mask]
                p_full = lam_pos ** 2 / max(float((lam_pos ** 2).sum()), 1e-12)
                p_sorted = np.sort(p_full)[::-1]
                cum = np.cumsum(p_sorted)
                ess = 1.0 / max(float((p_full * p_full).sum()), 1e-12)
                print(
                    f"           p concentration: top1={p_sorted[0]:.4f}  "
                    f"top3 cum={cum[min(2, len(cum)-1)]:.4f}  "
                    f"top5 cum={cum[min(4, len(cum)-1)]:.4f}  "
                    f"ESS(1/Σp²)={ess:.2f}  uniform-ESS={lam_pos.size}"
                )

            k_show = min(top_k_show, len(idx_global))
            top_cos_idx = np.argsort(-cos_vec)[:k_show]
            top_haw_idx = np.argsort(-scores)[:k_show]
            print(f"           top-{k_show} by cosine :")
            for rel in top_cos_idx:
                j = int(idx_global[int(rel)])
                m = "*" if is_evi[j] else " "
                print(
                    f"             cos={cos_vec[int(rel)]:+.3f}  λ⁻={decayed[int(rel)]:.4f}  "
                    f"score={scores[int(rel)]:+.3f}  id={m}{sids[j]} :: {snippets[j]}"
                )
            print(f"           top-{k_show} by hawkes-score :")
            for rel in top_haw_idx:
                j = int(idx_global[int(rel)])
                m = "*" if is_evi[j] else " "
                print(
                    f"             score={scores[int(rel)]:+.3f}  cos={cos_vec[int(rel)]:+.3f}  "
                    f"λ⁻={decayed[int(rel)]:.4f}  id={m}{sids[j]} :: {snippets[j]}"
                )

            # 激励更新
            order = np.argsort(-scores)
            k = min(inter_top_k, len(order))
            called_local = order[:k]
            updates_log = []
            for rel in called_local:
                rel = int(rel)
                score_i = float(scores[rel])
                if score_i <= 0.0:
                    continue
                lam_minus = float(decayed[rel])
                global_j = int(idx_global[rel])
                delta = (1.0 - lam_minus) * score_i
                new_lam = min(1.0, lam_minus + delta)
                lambdas[global_j] = new_lam
                last_update[global_j] = t_i
                last_call_time[global_j] = t_i
                updates_log.append(
                    f"id={sids[global_j]}: λ⁻={lam_minus:.4f} "
                    f"+ Δ={delta:.4f} (score={score_i:+.4f}) → λ⁺={new_lam:.4f}"
                )
            if updates_log:
                print(f"           excited {len(updates_log)}/{k}:")
                for u in updates_log:
                    print(f"             {u}")
            else:
                print(f"           excited 0/{k} (all scores ≤0)")

        # 创建当前 session
        lambdas[i] = 1.0
        last_update[i] = t_i
        last_call_time[i] = t_i
        created[i] = True

    # 终查询
    print("\n" + "#" * 100)
    print(f"[FINAL QUERY] {fmt_ts(question_time)}  -- {short(question, 120)}")
    print("#" * 100)
    decayed_q = lambdas * np.exp(-beta * (question_time - last_update))
    decayed_q = np.clip(decayed_q, 0.0, 1.0)
    mu_q = compute_mu(decayed_q, mu_base)
    h_norm_q, h_nats_q, N_used_q = h_hat(decayed_q)
    cos_q = sess_vec @ q_vec
    final_scores = cos_q * (mu_q + (1.0 - mu_q) * decayed_q)

    print(
        f"Ĥ={h_norm_q:.4f}  H={h_nats_q:.4f}  ln(N)={math.log(N_used_q):.4f}  "
        f"μ={mu_q:.4f}  √(1-Ĥ)={math.sqrt(max(1-h_norm_q,0)):.4f}"
    )
    print(f"λ⁻ at query : {quantile_str(decayed_q)}")
    pos = decayed_q[decayed_q > 0]
    if pos.size:
        p = pos ** 2 / float((pos ** 2).sum())
        p_sorted = np.sort(p)[::-1]
        cum = np.cumsum(p_sorted)
        ess = 1.0 / float((p * p).sum())
        print(
            f"p concentration at query : top1={p_sorted[0]:.4f}  "
            f"top3 cum={cum[min(2, len(cum)-1)]:.4f}  "
            f"top5 cum={cum[min(4, len(cum)-1)]:.4f}  "
            f"ESS={ess:.2f} (uniform={pos.size})"
        )
    bracket = mu_q + (1.0 - mu_q) * decayed_q
    print(f"bracket [μ+(1-μ)λ⁻] stats: {quantile_str(bracket)}")
    print(
        f"bracket vs cosine corr: corrcoef(cos, bracket)="
        f"{float(np.corrcoef(cos_q, bracket)[0,1]):.4f}"
    )

    k_show = min(top_k_show, n)
    top_cos = np.argsort(-cos_q)[:k_show]
    top_haw = np.argsort(-final_scores)[:k_show]
    print(f"\nTop-{k_show} by cosine :")
    for j in top_cos:
        j = int(j)
        m = "*" if is_evi[j] else " "
        print(
            f"  cos={cos_q[j]:+.3f}  λ⁻={decayed_q[j]:.4f}  bracket={bracket[j]:.4f}  "
            f"score={final_scores[j]:+.3f}  id={m}{sids[j]} :: {snippets[j]}"
        )
    print(f"\nTop-{k_show} by originidea :")
    for j in top_haw:
        j = int(j)
        m = "*" if is_evi[j] else " "
        print(
            f"  score={final_scores[j]:+.3f}  cos={cos_q[j]:+.3f}  λ⁻={decayed_q[j]:.4f}  "
            f"bracket={bracket[j]:.4f}  id={m}{sids[j]} :: {snippets[j]}"
        )

    cos_order = np.argsort(-cos_q)
    haw_order = np.argsort(-final_scores)
    rel_ranks_cos = sorted(int(np.where(cos_order == j)[0][0]) + 1
                            for j in range(n) if is_evi[j])
    rel_ranks_haw = sorted(int(np.where(haw_order == j)[0][0]) + 1
                            for j in range(n) if is_evi[j])
    print(f"\nranks of evidence sessions (1-indexed):")
    print(f"  cosine    : {rel_ranks_cos}")
    print(f"  originidea: {rel_ranks_haw}")

    # 全过程 H, μ 时间轨迹（额外打一次）
    print("\n# 复盘：每个 event 处的 (Ĥ, μ, pool_size, λ⁻>0)")
    lambdas2 = np.zeros(n, dtype=float)
    last_update2 = np.zeros(n, dtype=float)
    last_call2 = np.zeros(n, dtype=float)
    created2 = np.zeros(n, dtype=bool)
    print(f"  ev   t                   pool  λ>0   Ĥ        μ      median(λ⁻)")
    for i in range(n):
        t_i = float(s_times[i])
        if created2.any():
            idx_g = np.flatnonzero(created2)
            decayed = lambdas2[idx_g] * np.exp(-beta * (t_i - last_update2[idx_g]))
            decayed = np.clip(decayed, 0.0, 1.0)
            mu_v = compute_mu(decayed, mu_base)
            h_v, _, _ = h_hat(decayed)
            med = float(np.median(decayed)) if decayed.size else 0.0
            print(
                f"  {i+1:>2}   {fmt_ts(t_i)}  {len(idx_g):>3}   "
                f"{int((decayed>0).sum()):>3}  {h_v:.4f}  {mu_v:.4f}   {med:.4f}"
            )
            cos_vec = sess_vec[idx_g] @ sess_vec[i]
            mix_vec = mu_v + (1.0 - mu_v) * decayed
            scores = cos_vec * mix_vec
            order = np.argsort(-scores)[:inter_top_k]
            for rel in order:
                rel = int(rel)
                s_i = float(scores[rel])
                if s_i <= 0:
                    continue
                lam_minus = float(decayed[rel])
                gj = int(idx_g[rel])
                lambdas2[gj] = min(1.0, lam_minus + (1 - lam_minus) * s_i)
                last_update2[gj] = t_i
                last_call2[gj] = t_i
        else:
            print(f"  {i+1:>2}   {fmt_ts(t_i)}    0     0   ----     ----     ----")
        lambdas2[i] = 1.0
        last_update2[i] = t_i
        last_call2[i] = t_i
        created2[i] = True

    # 最终 query 处
    decayed_qq = lambdas2 * np.exp(-beta * (question_time - last_update2))
    decayed_qq = np.clip(decayed_qq, 0.0, 1.0)
    mu_qq = compute_mu(decayed_qq, mu_base)
    h_qq, _, _ = h_hat(decayed_qq)
    print(
        f"  Q    {fmt_ts(question_time)}  {n:>3}   "
        f"{int((decayed_qq>0).sum()):>3}  {h_qq:.4f}  {mu_qq:.4f}   "
        f"{float(np.median(decayed_qq)):.4f}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path,
                   default=Path("benchmarks/longmemeval/cache/longmemeval_s.json"))
    p.add_argument("--qid", default=None)
    p.add_argument("--T-half", type=float, default=14.0,
                   help="T_{1/2}(d)")
    p.add_argument("--mu-base", type=float, default=0.1)
    p.add_argument("--inter-top-k", type=int, default=3)
    p.add_argument("--final-top-k", type=int, default=5)
    p.add_argument("--top-k", type=int, default=5,
                   help="trace 中每步打印多少条 top")
    p.add_argument("--embedding", choices=["qwen", "bge", "hashing"], default="qwen")
    p.add_argument("--device", default="auto")
    p.add_argument("--model-cache-dir", type=Path,
                   default=Path("benchmarks/longmemeval/cache/models"))
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    beta = math.log(2.0) / args.T_half

    embed_fn = make_embedding_fn(
        args.embedding,
        device=args.device,
        cache_dir=args.model_cache_dir if args.embedding != "hashing" else None,
    )

    records = json.loads(args.data.read_text())
    if args.qid:
        rec = next((r for r in records if str(r.get("question_id")) == args.qid), None)
        if rec is None:
            raise SystemExit(f"qid not found: {args.qid}")
    else:
        pool = [r for r in records if is_multi_session(r)]
        rec = pool[0]

    out_stream = None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out_stream = args.output.open("w")

    class Tee:
        def __init__(self, *ss): self.ss = ss
        def write(self, s):
            for x in self.ss: x.write(s)
        def flush(self):
            for x in self.ss: x.flush()

    if out_stream is not None:
        sys.stdout = Tee(sys.__stdout__, out_stream)
    try:
        trace_question(
            rec, embed_fn,
            beta=beta,
            mu_base=args.mu_base,
            inter_top_k=args.inter_top_k,
            final_top_k=args.final_top_k,
            top_k_show=args.top_k,
        )
    finally:
        if out_stream is not None:
            sys.stdout = sys.__stdout__
            out_stream.close()
            print(f"[trace saved -> {args.output}]")


if __name__ == "__main__":
    main()

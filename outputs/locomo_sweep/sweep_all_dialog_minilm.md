# LoCoMo Session-Vector R0 vs R1-lite Sweep

- n_questions: 1982
- memory_text_mode: dialog
- embedding: minilm
- mu_bases: [0.1, 0.3, 0.5, 0.7, 0.9]
- intermediate_top_k: 3
- cosine baseline: recall@10=0.6201, hit@1=0.2634, hit@5=0.5106, mrr@10=0.3730, srr@10=0.3986
- best by mrr@10: R1_lite_T90d_mu0.9 (recall@10=0.6235, hit@1=0.2674, hit@5=0.5086, mrr@10=0.3747, srr@10=0.4002)

## Recipe Summary

| recipe | T_half | beta | recall@10 | hit@1 | hit@5 | mrr@10 | srr@10 | W/T/L |
|---|---:|---:|---:|---:|---:|---:|---:|
| R0_cosine | - | 0.00000000 | 0.6201 | 0.2634 | 0.5106 | 0.3730 | 0.3986 | 0/1982/0 |
| R1_lite_T14d_mu0.1 | 14.0 | 0.04951051 | 0.6064 | 0.1912 | 0.4617 | 0.3094 | 0.3292 | 140/1661/181 |
| R1_lite_T30d_mu0.1 | 30.0 | 0.02310491 | 0.5913 | 0.1806 | 0.4339 | 0.2926 | 0.3102 | 169/1559/254 |
| R1_lite_T60d_mu0.1 | 60.0 | 0.01155245 | 0.5806 | 0.1907 | 0.4238 | 0.2980 | 0.3144 | 180/1513/289 |
| R1_lite_T90d_mu0.1 | 90.0 | 0.00770164 | 0.5776 | 0.2013 | 0.4344 | 0.3063 | 0.3232 | 162/1543/277 |
| R1_lite_T120d_mu0.1 | 120.0 | 0.00577623 | 0.5842 | 0.2104 | 0.4470 | 0.3173 | 0.3359 | 152/1586/244 |
| R1_lite_T200d_mu0.1 | 200.0 | 0.00346574 | 0.6061 | 0.2291 | 0.4788 | 0.3408 | 0.3617 | 127/1687/168 |
| R1_lite_T14d_mu0.3 | 14.0 | 0.04951051 | 0.6127 | 0.2175 | 0.4758 | 0.3332 | 0.3548 | 117/1723/142 |
| R1_lite_T30d_mu0.3 | 30.0 | 0.02310491 | 0.6011 | 0.1973 | 0.4637 | 0.3134 | 0.3337 | 139/1649/194 |
| R1_lite_T60d_mu0.3 | 60.0 | 0.01155245 | 0.5950 | 0.2084 | 0.4511 | 0.3184 | 0.3378 | 143/1629/210 |
| R1_lite_T90d_mu0.3 | 90.0 | 0.00770164 | 0.6002 | 0.2185 | 0.4637 | 0.3298 | 0.3497 | 137/1653/192 |
| R1_lite_T120d_mu0.3 | 120.0 | 0.00577623 | 0.6080 | 0.2260 | 0.4798 | 0.3386 | 0.3592 | 130/1685/167 |
| R1_lite_T200d_mu0.3 | 200.0 | 0.00346574 | 0.6130 | 0.2442 | 0.4929 | 0.3555 | 0.3782 | 92/1777/113 |
| R1_lite_T14d_mu0.5 | 14.0 | 0.04951051 | 0.6183 | 0.2341 | 0.4934 | 0.3497 | 0.3722 | 86/1801/95 |
| R1_lite_T30d_mu0.5 | 30.0 | 0.02310491 | 0.6031 | 0.2291 | 0.4844 | 0.3426 | 0.3642 | 101/1733/148 |
| R1_lite_T60d_mu0.5 | 60.0 | 0.01155245 | 0.6048 | 0.2326 | 0.4884 | 0.3443 | 0.3663 | 108/1725/149 |
| R1_lite_T90d_mu0.5 | 90.0 | 0.00770164 | 0.6082 | 0.2407 | 0.4924 | 0.3504 | 0.3728 | 101/1747/134 |
| R1_lite_T120d_mu0.5 | 120.0 | 0.00577623 | 0.6164 | 0.2477 | 0.4955 | 0.3582 | 0.3813 | 90/1790/102 |
| R1_lite_T200d_mu0.5 | 200.0 | 0.00346574 | 0.6169 | 0.2563 | 0.5050 | 0.3665 | 0.3901 | 64/1843/75 |
| R1_lite_T14d_mu0.7 | 14.0 | 0.04951051 | 0.6249 | 0.2543 | 0.4985 | 0.3648 | 0.3889 | 64/1864/54 |
| R1_lite_T30d_mu0.7 | 30.0 | 0.02310491 | 0.6209 | 0.2523 | 0.5010 | 0.3625 | 0.3866 | 73/1838/71 |
| R1_lite_T60d_mu0.7 | 60.0 | 0.01155245 | 0.6254 | 0.2528 | 0.5010 | 0.3644 | 0.3890 | 72/1851/59 |
| R1_lite_T90d_mu0.7 | 90.0 | 0.00770164 | 0.6251 | 0.2573 | 0.5050 | 0.3674 | 0.3917 | 61/1871/50 |
| R1_lite_T120d_mu0.7 | 120.0 | 0.00577623 | 0.6239 | 0.2573 | 0.5071 | 0.3683 | 0.3928 | 52/1885/45 |
| R1_lite_T200d_mu0.7 | 200.0 | 0.00346574 | 0.6244 | 0.2634 | 0.5086 | 0.3726 | 0.3977 | 43/1905/34 |
| R1_lite_T14d_mu0.9 | 14.0 | 0.04951051 | 0.6221 | 0.2639 | 0.5116 | 0.3730 | 0.3985 | 22/1944/16 |
| R1_lite_T30d_mu0.9 | 30.0 | 0.02310491 | 0.6256 | 0.2669 | 0.5086 | 0.3745 | 0.4003 | 30/1936/16 |
| R1_lite_T60d_mu0.9 | 60.0 | 0.01155245 | 0.6232 | 0.2664 | 0.5101 | 0.3742 | 0.3996 | 26/1938/18 |
| R1_lite_T90d_mu0.9 | 90.0 | 0.00770164 | 0.6235 | 0.2674 | 0.5086 | 0.3747 | 0.4002 | 25/1940/17 |
| R1_lite_T120d_mu0.9 | 120.0 | 0.00577623 | 0.6243 | 0.2669 | 0.5086 | 0.3745 | 0.4000 | 23/1945/14 |
| R1_lite_T200d_mu0.9 | 200.0 | 0.00346574 | 0.6224 | 0.2639 | 0.5081 | 0.3729 | 0.3984 | 16/1955/11 |

## Best Buckets: Category

| bucket | cosine mrr@10 | best recipe | best mrr@10 | W/T/L |
|---|---:|---|---:|---:|
| 1 | 0.4697 | R1_lite_T200d_mu0.7 | 0.4782 | 16/251/15 |
| 2 | 0.4268 | R1_lite_T30d_mu0.9 | 0.4269 | 2/316/3 |
| 3 | 0.3001 | R1_lite_T14d_mu0.7 | 0.3107 | 2/88/2 |
| 4 | 0.3406 | R1_lite_T90d_mu0.9 | 0.3427 | 7/832/2 |
| 5 | 0.3491 | R1_lite_T30d_mu0.9 | 0.3515 | 8/435/3 |

## Best Buckets: Answer Lag

| bucket | cosine mrr@10 | best recipe | best mrr@10 | W/T/L |
|---|---:|---|---:|---:|
| high | 0.3546 | R1_lite_T120d_mu0.9 | 0.3538 | 3/641/7 |
| low | 0.3852 | R1_lite_T60d_mu0.1 | 0.4739 | 127/504/38 |
| mid | 0.3786 | R1_lite_T200d_mu0.9 | 0.3758 | 7/651/4 |

## Best Buckets: Evidence Count

| bucket | cosine mrr@10 | best recipe | best mrr@10 | W/T/L |
|---|---:|---|---:|---:|
| 1 | 0.3524 | R1_lite_T90d_mu0.9 | 0.3534 | 14/1627/8 |
| 2 | 0.4503 | R1_lite_T200d_mu0.7 | 0.4563 | 9/188/10 |
| 3+ | 0.5151 | R1_lite_T14d_mu0.9 | 0.5294 | 5/119/2 |

## Best Buckets: Sample

| bucket | cosine mrr@10 | best recipe | best mrr@10 | W/T/L |
|---|---:|---|---:|---:|
| conv-26 | 0.4167 | R1_lite_T200d_mu0.7 | 0.4282 | 2/192/3 |
| conv-30 | 0.4694 | R1_lite_T90d_mu0.9 | 0.4704 | 1/103/1 |
| conv-41 | 0.4648 | R1_lite_T90d_mu0.9 | 0.4809 | 5/185/3 |
| conv-42 | 0.3796 | R1_lite_T200d_mu0.5 | 0.3925 | 10/243/7 |
| conv-43 | 0.4089 | R1_lite_T200d_mu0.7 | 0.4087 | 3/236/3 |
| conv-44 | 0.3573 | R1_lite_T200d_mu0.5 | 0.3675 | 10/141/7 |
| conv-47 | 0.3085 | R1_lite_T120d_mu0.5 | 0.3325 | 13/168/9 |
| conv-48 | 0.3384 | R1_lite_T200d_mu0.9 | 0.3326 | 1/237/1 |
| conv-49 | 0.3428 | R1_lite_T14d_mu0.9 | 0.3439 | 1/193/2 |
| conv-50 | 0.2840 | R1_lite_T14d_mu0.7 | 0.2941 | 9/190/3 |

## Category Analysis: R0 Cosine vs Best R1-lite (T90d, μ=0.9)

Question categories in LoCoMo:
- **1** = Single-session factual (e.g. "What is Caroline's identity?")
- **2** = Single-session temporal (e.g. "When did Caroline go to the LGBTQ support group?")
- **3** = Multi-session aggregation (e.g. "What activities does Melanie partake in?")
- **4** = Multi-session reasoning (e.g. "Why did Caroline choose the adoption agency?")
- **5** = Adversarial / distractor (has `adversarial_answer` field)

### Category 1: Single-Session Factual (n=282)

| metric | R0 Cosine | R1-lite T90d μ=0.9 | delta |
|---|---:|---:|---:|
| recall@10 | 0.5780 | 0.5780 | 0.0000 |
| hit@1 | 0.3156 | 0.3262 | +0.0106 |
| hit@5 | 0.6915 | 0.6809 | -0.0106 |
| mrr@10 | 0.4697 | 0.4748 | +0.0051 |
| srr@10 | 0.6185 | 0.6216 | +0.0031 |

> R1-lite slightly better: hit@1 and mrr improve marginally.

### Category 2: Single-Session Temporal (n=321)

| metric | R0 Cosine | R1-lite T90d μ=0.9 | delta |
|---|---:|---:|---:|
| recall@10 | 0.7087 | 0.7035 | -0.0052 |
| hit@1 | 0.3146 | 0.3146 | 0.0000 |
| hit@5 | 0.5421 | 0.5545 | +0.0124 |
| mrr@10 | 0.4268 | 0.4262 | -0.0006 |
| srr@10 | 0.4453 | 0.4453 | 0.0000 |

> Roughly tied: hit@5 improves slightly but recall drops slightly.

### Category 3: Multi-Session Aggregation (n=92)

| metric | R0 Cosine | R1-lite T90d μ=0.9 | delta |
|---|---:|---:|---:|
| recall@10 | 0.4633 | 0.4560 | -0.0073 |
| hit@1 | 0.2065 | 0.2065 | 0.0000 |
| hit@5 | 0.4239 | 0.4130 | -0.0109 |
| mrr@10 | 0.3001 | 0.2964 | -0.0037 |
| srr@10 | 0.3312 | 0.3280 | -0.0032 |

> R1-lite worse across the board: temporal decay hurts cross-session aggregation.

### Category 4: Multi-Session Reasoning (n=841)

| metric | R0 Cosine | R1-lite T90d μ=0.9 | delta |
|---|---:|---:|---:|
| recall@10 | 0.6076 | 0.6136 | +0.0060 |
| hit@1 | 0.2461 | 0.2509 | +0.0048 |
| hit@5 | 0.4542 | 0.4530 | -0.0012 |
| mrr@10 | 0.3406 | 0.3427 | +0.0021 |
| srr@10 | 0.3406 | 0.3427 | +0.0021 |

> R1-lite slightly better: recall and hit@1 improve marginally.

### Category 5: Adversarial / Distractor (n=446)

| metric | R0 Cosine | R1-lite T90d μ=0.9 | delta |
|---|---:|---:|---:|
| recall@10 | 0.6390 | 0.6480 | +0.0090 |
| hit@1 | 0.2377 | 0.2399 | +0.0022 |
| hit@5 | 0.4978 | 0.4910 | -0.0068 |
| mrr@10 | 0.3491 | 0.3509 | +0.0018 |
| srr@10 | 0.3491 | 0.3509 | +0.0018 |

> R1-lite slightly better: recall improves the most among all categories.

### Summary

| category | R1 vs R0 verdict |
|---|---|
| 1 Single-session factual | Slight win (hit@1 +0.01, mrr +0.005) |
| 2 Single-session temporal | Tie |
| 3 Multi-session aggregation | Loss (all metrics decline) |
| 4 Multi-session reasoning | Slight win (recall +0.006) |
| 5 Adversarial / distractor | Slight win (recall +0.009) |

Key insight: R1-lite's temporal decay is harmful for multi-session aggregation questions,
which require gathering information across multiple time points — the decay weakens older
sessions that may contain critical evidence. Overall differences are very small, suggesting
limited benefit of temporal signals on LoCoMo.
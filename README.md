# hawkes_rag

面向 **智能体长期记忆** 的一套实现：把 **向量检索（RAG）** 与 **受 Hawkes 过程启发的 λ（强度）动力学** 放在同一条链路里——记忆不仅有「像不像」，还有「该不该被记起」的强度，并随时间与使用反馈演进。

公开仓库刻意 **不包含** benchmark、数据集与大体积实验产物；若要复现实验流水线，需在本地另行维护整套评测目录。

---

## 设计动机（直觉）

- **纯向量 RAG**：相似度高的片段总会挤到前排，难以表达「新近强化 / 过时衰减 / 不同类型记忆应有不同黏性」等行为。
- **本项目的折中**：每条记忆携带一个 **[0,1] 区间的强度 λ**（实现里写入为 `lambda_plus`，读出时用距离上次事件的时间做指数衰减得到 **λ⁻**），检索得分在相似度（余弦）与 λ 之间 **`μ` 调制混合**：λ 分布越「平」、越容易众声喧哗时，系统自动更信几何相似度；λ 分布越尖锐、只有少数记忆真正「炽热」时，更强调 λ 本身。
- **Hot / Cold**：高强度的记忆走 **hot** 通道（带 λ），低强度记忆走 **cold** 通道（更偏字面与语义兜底），避免「被遗忘但关键词仍命中」的内容永远进不了候选集。
- **可选交叉编码器重排**：在 hot 上对粗排结果再做一轮 **rerank**，若置信不足或 hot 分部太平，触发 **cold 混合检索**（BM25 + 向量按比例融合）。

以上机制的实现入口主要在 `hawkes_agent.memory.InMemoryVectorStore`、`hawkes_agent.dynamics`、`hawkes_agent.recall.RecallMiddleware`。

---

## 记忆载体：`MemoryRecord`

每条记忆是一块可检索文本 + 向量 + 类型与命名空间：

| 字段（概念） | 含义 |
|-------------|------|
| `embedding` | 归一化后的查询向量 |
| **`lambda_plus`（写入态强度）** | 上一次 **强化或抑制更新** 之后的强度上界 |
| **`t_last_event`** | 上次 λ 更新时间（仿真时钟 `now`） |
| `type_class` | **`volatile` / `stable` / `identity`** 等，对应 **不同的遗忘速率 `β`**（见 `DynamicsConfig.beta_by_type`） |
| `namespace` | 会话或用户维度上的逻辑分池 |

存入时：`lambda_plus = 1.0`，时间与创建时间对齐到当前 `now`。

---

## 时间衰减：从 λ⁺ 到 λ⁻（读取时的有效强度）

对每条记录在查询时刻 `now`：

$$
\lambda^- = \min\left(1,\ \max\left(0,\ \lambda^+ \cdot e^{-\beta \cdot \Delta t}\right)\right),\quad \Delta t = now - t_{\text{last\_event}}
$$

- **`β` 越大，遗忘越快**（`volatile` 默认远大于 `stable` / `identity`）。
- λ⁻ 直接影响 **是否在 hot 集合**（与 `DynamicsConfig.hot_lambda_threshold` 比较）以及 **带 λ 的检索打分**。

实现：`hawkes_agent.dynamics.decayed_lambda`，由 `InMemoryVectorStore.decayed_lambdas` 批量套用。

---

## 检索打分：自适应 μ + Hawkes 型组合

给定候选集的 **余弦相似度向量 `cos`** 与 **衰减后 λ⁻ 向量**，先算批次级 **μ ∈ [μ\_base, 1]**（`compute_mu`）：

- 用 **λ² 归一化成概率分布**，算归一化熵 **ĥ**，分布越分散 **ĥ 越大**；
- $$
  \mu = \mu_{\text{base}} + (1-\mu_{\text{base}})\sqrt{1-\hat{h}}
  $$
  **直觉**：大家都很「炽」且不区分主次时增大对相似度的倚重；只有少数记忆强度高时更接近「Hawkes 风格」的强度调制。

随后 **逐条** 得分：

$$
\text{score}_i = \cos_i \cdot \bigl(\mu + (1-\mu)\cdot \lambda^-_i\bigr)
$$

- **λ⁻ 越高**，在 **`(1-μ)`** 这段权重里越占优势；
- **`cos_i` 有 `cosine_floor` 下限**（避免数值噪声）。
- **`use_lambda=False` 的路径**（cold 通道里用于粗排）退回为 **仅用余弦**，此时实现里等价于 **μ=1**。

实现：`hawkes_agent.dynamics.recall_scores`，由 `_recall_from_records` 调用；返回的每条结果里带 **`hawkes_score`** 字段（启用 λ 时等于该打分），便于上游日志与后续的 **采纳 / 抑制**。

---

## Hot / Cold 双通道

1. **按 λ⁻ 与阈值拆分**：
   - **hot**：$\lambda^- \geq$ `hot_lambda_threshold`
   - **cold**：反之
2. **`recall_hot_cold`**：
   - hot：上面公式、`use_lambda=True`
   - cold：`use_lambda=False`（仅用余弦粗排）
   - 两条结果合并时 **同一 id 取更高分**，再排序截断。
3. **`recall_hot_cold_reranked`（更重的一条路径）**：
   - 先从 hot 中取规模 **`intermediate_top_k`** 的粗候选；
   - 可对 hot 批量调用 **`Reranker`**（可选用本地 Qwen 交叉编码器），得到 **`rerank_score`**；
   - **若 rerank / Hawkes / 分部熵 表明「不可靠」**：触发 cold，cold 侧重 **BM25 + 余弦混合** ——向量与 BM25 分别 min-max 归一化后以 **`DynamicsConfig.alpha`** 加权；
     - cold 并入时 **候选总预算仍为 `intermediate_top_k`**，实现里近似 **约 1/4 hot + 3/4 cold** 的资源切分；
   - **触发 cold 的常见原因（元信息里可追溯）**：`low_confidence`（top1 rerank / Hawkes 低于 `tau_r`、`tau_h`）、`flat_hot_distribution`（hot rerank 分熵高于 `hot_entropy_threshold`）、`insufficient_hot_coverage`（高分条数少于 `min_hot_injected`）。

---

## Cross-encoder（可选）

`RecallMiddleware` 通过 `AgentHarnessConfig.reranker_backend` / `reranker_model` 装配 **`Reranker`**。

- **`rerank_top_k`**：只对 hot 前 K 条做交叉编码（0 表示不裁 batch）。
- 交叉编码打分可喂入 **prior**（来自 Hawkes 分或 cold 候选分），详见 `_apply_rerank`。

默认配置里模型路径示例见 `hawkes_agent.config`（重型权重放在 **`models/`**，该目录在 `.gitignore` 中）。

---

## 写入与闭环：采纳 / 强化 / 矛盾抑制

链路由 `RecallMiddleware` 收口：

1. **`write_turn`**：新记忆 `add_memory`，初始 λ⁺=1，`t_last_event=now`。
2. **`score_adoption`**：根据 **`adoption_method`**：
   - `embedding`：`answer` 与记忆片段的向量相似度；
   - `token_overlap`：重叠比例；
   与阈值 **`theta_a`** 比较，得到被「采纳」的记忆 id。
3. **`reinforce`**：对采纳片段，  
   $$
   \lambda^+_{\text{new}} = \lambda^- + (1-\lambda^-)\cdot s
   $$
   其中 **`s` 为该次检索阶段的 Hawkes 分或等价 score**（`reinforce_lambda`）。
4. **`suppress`**：对判定为与被采纳答案矛盾的片段（由上游逻辑给出 id 列表），按 **与查询的余弦** 作为压制强度 **`contradiction_similarity`**：
   $$
   \lambda^+_{\text{new}} = \lambda^- \cdot (1 - \texttt{clamp}(\text{contradiction\_similarity}))
   $$
   实现：`suppress_lambda`。  
   同时提供 **`prescreen_contradiction_signal`**：对已检索但未采纳、却仍与高余弦匹配的片段做一次 **矛盾风险预筛**（阈值 **`theta_c` / `theta_a`**，`contradiction_top_k`）。

以上把 **检索 → 生成 → （可选矛盾检查）→ 强化/抑制** 闭合成可观测、可重复的 λ 动力学。

---

## 配置：`DynamicsConfig`（核心旋钮）

节选常见字段及其角色：

| 字段 | 角色 |
|------|------|
| `beta_by_type` | 各类记忆的指数衰减速率 |
| `mu_base` | μ 下界 |
| `hot_lambda_threshold` | hot/cold 分界 |
| `intermediate_top_k` / `final_top_k` / `hot_top_k` / `cold_top_k` | 候选规模与拆分 |
| `alpha` | cold 路径里向量 vs BM25 权重 |
| `tau`、`tau_r`、`tau_h`、`hot_entropy_threshold`、`min_hot_injected` | 阈值与 cold 触发判据 |

完整定义见 **`hawkes_agent/config.py`**。

---

## `hawkes_rag` 包

提供 **sentence-transformers 封装、SQLite 向量缓存路径**等，便于在多轮实验中复用 embedding。仓库默认路径仍指向历史中用于评测的缓存目录布局；若在 **无 `benchmarks/` 的极简检出**中使用，建议在调用侧 **显式传入 `cache_dir`**，不要把缓存假定在已忽略的目录结构上。

---

## 安装与仓库布局

**Python 3.11+** 推荐。

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-agent-harness.txt
```

```
hawkes_agent/           # λ 动力学、向量库、RecallMiddleware、重排钩子
hawkes_rag/             # 嵌入与缓存工具
scripts/                # 小工具脚本（例如 sweep 汇总可视化）
requirements-agent-harness.txt
```

主线依赖含 **LangGraph、LiteLLM、Pydantic AI、sentence-transformers**（见依赖文件）。

---

## 许可证

若在 GitHub 公开发布，请自行补充 **`LICENSE`** 并替换本节说明。

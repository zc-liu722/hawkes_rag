# 自激记忆：用 Hawkes 过程统一 LLM 记忆系统

> 一份从对话里提炼出来的研究纲要 + 执行计划
> 目标产出：**GitHub repo + arXiv short report**（3-4 周）

---

## 一、思路是怎么走出来的

整条路径其实是**五步追问**，每一步都把上一步的答案推翻或扩展了：

**第一步：数学问题。** "数学上有没有什么东西，越用越强、越不用越弱？"——答案是 **Hawkes 过程（自激点过程）**：每次事件让强度 λ(t) 跳一下，然后按 e^(-βt) 衰减回基线。地震余震、神经放电、爆款传播、人脑记忆，本质都是同一件事。

但这一步藏了个陷阱：**会衰减的从来不是数学概念本身，而是它的载体。** 这把问题从"数学好奇心"推到了"信息存储的物理约束"。

**第二步:神经科学的类比。** "人脑记忆是神经元权重的增减，那 LLM 是不是该实时更新权重？"——直觉对了一半。技术上叫 online learning / continual learning，撞墙撞了几十年，墙的名字叫**灾难性遗忘**：神经网络在新数据上做梯度下降时，会顺手把无关的旧能力一起拧坏，因为损失函数只看当前数据，不知道权重 W₃₇₂ 同时还在支撑别的事。

人脑能做到精准强化/减弱而不崩溃，是因为有一整套**多系统协作**：海马体快速存快照、皮层慢速巩固、睡眠重放、神经调质调节学习率。当前 LLM 几乎只有皮层。

**第三步：器官类比的对应表。** 把这套类比做诚实：
- 皮层 = 预训练后的 LLM 权重
- 海马 = 上下文 + RAG 检索库
- 睡眠慢波 = 离线再训练 / experience replay
- 基底神经节 = RLHF 偏好头
- 神经调质 = 学习率/温度/注意力增益的动态调节
- 脑干 = 推理引擎本身

**关键缺口：神经调质对应的"动态学习率"在 LLM 里是缺失的——它无法自己判断"这一条比那一条更值得记"。**

**第四步：核心洞察。** 既然实时更新权重的路走不通（短期内），那 Hawkes 这个数学结构能不能落到 RAG 上？——能，且这是当前 RAG 最缺的一块：

> 当前 RAG 的检索本质是**无状态**的——库不会因为某条记忆被频繁使用而把它"提前"。给每条记忆 i 维护一个激活强度 λᵢ(t)：
> - 被检索/被引用 → λᵢ 跳一下
> - 没被用 → 按 e^(-β·Δt) 衰减
> - 检索时按"语义相似度 × λᵢ"排序——常用的浮上来，不用的沉下去
> - λᵢ 长期高于阈值 → 移交 LoRA / 微调队列，写进权重
> - λᵢ 长期低于阈值 → 删除（自然遗忘）

这就**完整复刻了海马 → 皮层的巩固通道**。

**第五步：现状调研。**
- **没有任何主流论文**显式把 RAG 记忆建模成 Hawkes 过程
- 但"指数衰减 × 访问强化"已经是事实标准——大家管它叫"艾宾浩斯遗忘曲线"
- 数学形式高度相似，但**严谨度远低于真正的自激点过程**：没人做互激矩阵、稳定性分析、参数估计
- 这是一个**论文形状的洞**

---

## 二、核心主张（Core Thesis）

**一句话定位：**

> 把 LLM agent 的长期记忆系统，统一建模为一个**多变量 Hawkes 过程**（Multivariate Hawkes Process, MHP）——用一个数学框架同时解释衰减、强化、互激、巩固、遗忘。

**为什么这个定位有杠杆：**

现有方案各做一块——Zep 做时间感知，Mem0 做提取，HippoRAG 做图索引，SAGE/MARS 做艾宾浩斯衰减。**没人有统一的数学框架。** 这就像在牛顿之前，每个人都研究自己那块——苹果落地、月亮绕地、潮汐——但缺一个 F=ma 把它们串起来。

MHP 给的就是那个 F=ma：

```
λᵢ(t) = μᵢ + Σⱼ ∫₀ᵗ αᵢⱼ · e^(-βᵢⱼ(t-s)) · dNⱼ(s)
```

这一个公式同时编码：
- **基线强度** μᵢ：记忆 i 的内在重要性
- **自激** αᵢᵢ：被自己激活后的强化
- **互激** αᵢⱼ：记忆 j 被激活会带动记忆 i（关联记忆/概念簇）
- **衰减** βᵢⱼ：每条关联的时间尺度（短期/长期记忆）

**这是当前所有"艾宾浩斯衰减+访问计数"工作错过的关键——它们都只做单变量自激，没有互激矩阵。** 而互激矩阵正是"联想记忆""概念巩固""上下文激活"这些认知功能的数学骨架。

---

## 三、核心创新点（拆成可写论文的颗粒度）

| # | 创新点 | 现有工作怎么做 | 我们怎么做 |
|---|---|---|---|
| 1 | **多变量自激建模** | 单条记忆独立衰减+计数 | 学习一个 N×N 互激矩阵 α，捕捉记忆间的激活耦合 |
| 2 | **检索分数复合** | 余弦相似度 或 相似度+时间衰减 | 相似度 × λᵢ(t)，且 λᵢ 由整个网络的历史决定 |
| 3 | **巩固阈值机制** | 无（记忆永远在向量库） | λᵢ 长期超阈值 → 自动移交 LoRA 微调；长期低于 → 删除 |
| 4 | **参数估计** | 拍脑袋设衰减率 | 从用户实际使用日志做 MLE 估计 βᵢⱼ、αᵢⱼ |
| 5 | **可解释遗忘** | "找不到这条" | 给出 λᵢ(t) 曲线，能解释"为什么这条变弱了" |
| 6 | **稳定性保证** | 无 | Hawkes 过程有谱半径稳定性条件 (ρ(α) < 1)，可证明系统不会爆炸 |

**第 1、3、6 点是别人没做的。** 第 1 是技术核心，第 3 是工程价值，第 6 是理论卖点。论文里 #1 必做，#2 #4 #5 必做，#6 当作 Theoretical Section 加分项，#3 当作 Discussion 章节的"未来方向"——不必完整实现。

---

## 四、产出 1：GitHub Repo

### 4.1 项目命名

候选（选一个，别纠结）：
- `mhp-memory`（学术、好搜）
- `hawkes-rag`（直白、SEO 友好）
- `synaptic`（品牌感强，但模糊）

**推荐 `hawkes-rag`。** 大白话，SEO 抓得住，"Hawkes" 这个词独特，搜索时几乎只会指向你。

### 4.2 仓库结构

```
hawkes-rag/
├── README.md                  ← 这是核心资产，比代码还重要
├── LICENSE                    ← MIT
├── pyproject.toml
├── hawkes_rag/
│   ├── __init__.py
│   ├── core.py                ← MHP 核心：λ(t) 计算、衰减、自激
│   ├── memory.py              ← Memory item, MemoryStore
│   ├── retrieval.py           ← 检索：相似度 × λ
│   ├── estimation.py          ← MLE 参数估计（α, β）
│   └── viz.py                 ← α 矩阵热力图、λ 曲线
├── examples/
│   ├── 01_basic_usage.py
│   ├── 02_chat_with_memory.py ← 接 Claude/OpenAI API
│   └── 03_visualize_decay.py
├── benchmarks/
│   ├── locomo/                ← LoCoMo 跑分
│   └── compare_baselines.py   ← vs Mem0, naive RAG
├── tests/
└── docs/
    ├── theory.md              ← 数学推导，作为 arXiv 论文的草稿
    └── design.md
```

**核心代码量预期：1500-2500 行 Python。** 不算 benchmarks 和 viz。

### 4.3 最小可跑骨架（约 200 行，第一周做完）

```python
# hawkes_rag/core.py 的精神原型

import numpy as np
from dataclasses import dataclass, field
from typing import List
import time

@dataclass
class Memory:
    id: int
    content: str
    embedding: np.ndarray
    created_at: float
    last_accessed: float
    base_intensity: float = 0.1   # μᵢ，记忆的内在重要性
    
@dataclass
class HawkesMemoryStore:
    memories: List[Memory] = field(default_factory=list)
    alpha: np.ndarray = None      # N×N 互激矩阵
    beta: float = 0.1             # 衰减率（先用标量，进阶版用矩阵）
    
    def intensity(self, i: int, t: float) -> float:
        """计算记忆 i 在时间 t 的激活强度 λᵢ(t)"""
        m = self.memories[i]
        lam = m.base_intensity
        # 自激 + 互激（来自所有过去事件）
        for j, mj in enumerate(self.memories):
            if mj.last_accessed > 0 and mj.last_accessed <= t:
                dt = t - mj.last_accessed
                lam += self.alpha[i, j] * np.exp(-self.beta * dt)
        return lam
    
    def add(self, content: str, embedding: np.ndarray):
        i = len(self.memories)
        self.memories.append(Memory(
            id=i, content=content, embedding=embedding,
            created_at=time.time(), last_accessed=0,
        ))
        # 扩展 α 矩阵（新记忆和旧记忆的初始耦合 = 嵌入相似度）
        self._expand_alpha(embedding)
    
    def retrieve(self, query_emb: np.ndarray, top_k: int = 5):
        t = time.time()
        scores = []
        for i, m in enumerate(self.memories):
            sim = cosine(query_emb, m.embedding)
            lam = self.intensity(i, t)
            scores.append((i, sim * lam))
        scores.sort(key=lambda x: -x[1])
        # 取出后，更新被命中的记忆的 last_accessed
        chosen = scores[:top_k]
        for i, _ in chosen:
            self.memories[i].last_accessed = t
        return chosen
    
    def _expand_alpha(self, new_emb):
        """新记忆加入时，用嵌入相似度初始化它和旧记忆的互激强度"""
        # 简化版：α[i,j] = max(0, cos(eᵢ, eⱼ) - 0.3)
        # 后续可学习
        ...

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
```

**这是第一周目标——能跑、能加记忆、能检索、被检索的记忆 λ 会跳。** 互激初始化用嵌入相似度（懒人版），第二周再做参数学习。

### 4.4 README 是真正的核心资产

**README 决定了 repo 能不能被传开。** 它得像一篇博客，有图、有故事、有可运行的 demo gif。

骨架：

```markdown
# hawkes-rag: A self-exciting memory system for LLM agents

> Memory that strengthens with use and naturally decays — like a brain.

[GIF: 显示一段对话，左边是常规 RAG，右边是 hawkes-rag，
后者随对话进行，相关记忆的 λ 曲线越长越高]

## Why this exists

Current RAG is stateless. Your AI doesn't know that you mentioned 
your dog Max five times last week and Python once last month — it 
treats both with equal weight.

Real memory works differently. Frequently accessed memories 
strengthen; unused ones fade. This is universally observed across 
neuroscience, earthquakes, social media virality — and it has a 
single mathematical form: the **Hawkes process**.

`hawkes-rag` brings this structure to LLM memory.

## Quick start
[3 行代码，能跑]

## How it works
[一段简洁的数学，配 α 矩阵热力图]

## Benchmarks
[LoCoMo 跑分对比表]

## Citation
[引用你的 arXiv]
```

**README 起码花 1 整天打磨。** 配图至少 2 张：一张是 α 矩阵热力图（跑过一段对话后，能看出概念簇），一张是某条记忆的 λ(t) 曲线。

### 4.5 三周时间表

**Week 1 — 核心能跑**
- Day 1-2：项目脚手架，core.py + memory.py 跑通基本流程
- Day 3-4：retrieval.py + 一个 example 接通 Claude API
- Day 5-7：viz.py 做出 α 热力图和 λ 曲线，README 初稿

**Week 2 — Benchmark + 参数学习**
- Day 8-10：LoCoMo 数据集接入，跑一遍 baseline（naive RAG、Mem0 如果能装上）
- Day 11-13：MLE 参数估计 estimation.py
- Day 14：完整跑出对比表，定下数字

**Week 3 — 打磨 + 发布**
- Day 15-17：README 完整版本 + 录制 demo gif + 写 docs/theory.md
- Day 18-19：写 arXiv 论文（见下一节）
- Day 20-21：发布到 GitHub + Hacker News + 小红书 + Twitter

**注意：** 这个时间表假定你**全职 3 周**或**业余 6-8 周**。别给自己开"我下班再做"的空头支票——业余做 6 周比全职 3 周失败率高 3 倍。

---

## 五、产出 2：arXiv Short Report

### 5.1 定位

**不是冲顶会，是 short report / technical note。**

- **页数：4-6 页**（不算引用），用 NeurIPS 或 arXiv 默认 LaTeX 模板
- **目标读者：** Memory-augmented LLM 方向的研究者 + 实践者
- **目标引用：** 6 个月内 5-15 引用就算成功
- **不冲会议**——冲会议要 8-10 页 + 完整 baseline + rebuttal，3-4 周做不完

### 5.2 标题（三选一）

1. **Hawkes-RAG: Self-Exciting Memory for Retrieval-Augmented Generation** ← 推荐
2. From Ebbinghaus to Hawkes: A Principled Framework for LLM Agent Memory
3. Multivariate Hawkes Processes for Lifelong Agent Memory

第一个最简洁，"Hawkes-RAG" 这个词组没人用过，独占 SEO。

### 5.3 论文骨架

**Abstract（150 字）**
当前 LLM 记忆系统多为经验拼贴；我们提出用多变量 Hawkes 过程统一建模衰减、强化、互激、巩固。在 LoCoMo 上提升 X%，且提供稳定性保证。

**1. Introduction（0.5 页）**
- 现状：Mem0、Letta、Zep、HippoRAG 等各做一块
- 共同缺口：没有统一数学框架；尤其是没人做互激
- 我们的贡献（3 条 bullet）

**2. Related Work（0.5 页）**
两段：(a) memory systems for LLM agents (b) Hawkes processes & TPP-LLM。指出 TPP-LLM 用 TPP 做事件预测但**没人反过来用它做记忆**。

**3. Method（1.5 页）—— 论文核心**
- 3.1 Memory items as marks of a point process
- 3.2 Multivariate Hawkes intensity function（写出公式）
- 3.3 Low-rank parametrization of α via embedding similarity（避免 O(N²) 学习）
- 3.4 Retrieval scoring: similarity × intensity
- 3.5 MLE parameter estimation
- 3.6 Stability: spectral radius condition（半页推导）

**4. Experiments（1.5 页）**
- Setup：LoCoMo benchmark，对比 naive RAG、Mem0、HippoRAG（够了，不必跑全）
- Main result table
- Ablation：去掉互激（变回单变量）→ 性能差距证明互激重要
- Qualitative：α 矩阵可视化，能看出语义簇

**5. Discussion（0.5 页）**
- 通往权重巩固（LoRA 移交）的路径
- 与人类记忆研究的连接
- 局限性

**6. Conclusion（0.25 页）**

**总长度估计：4.5-5.5 页。** 引用约 25-35 篇。

### 5.4 实验最小集（必须跑通的）

| 实验 | 目的 | 工作量 |
|---|---|---|
| LoCoMo benchmark 主结果 | 证明 hawkes-rag 比 naive RAG 好 | 2-3 天 |
| Ablation：去掉互激 | 证明互激矩阵重要（这是论文最大卖点）| 1 天 |
| α 可视化 | 让读者直观感受 | 半天 |
| vs Mem0 | 一个商业基线对比 | 1-2 天（Mem0 装起来不简单）|

**LongMemEval 不必跑——超出 short report 范围。**

### 5.5 写作时间分配

写 4-6 页 arXiv short report 的真实时间（我说大实话）：
- 大纲 + 数学公式整理：1 天
- Method 章节：2 天
- Experiments 章节：1 天（前提是数据已经跑出来了）
- Intro + Related Work + Discussion：1 天
- 公式校对、图表精修、引用整理：1 天

**总共约 6 个全天。** 如果你之前没写过学术论文，加 50%——9-10 天。

### 5.6 投放

写完后：
1. 上传 arXiv（cs.CL 主分类，cs.LG 副分类）
2. 在 Twitter 发一条带 GIF 的推文，@ 几个相关研究者（Letta、Zep 团队）
3. 在 Hacker News 发"Show HN: hawkes-rag"
4. 在小红书发一条"我用 Hawkes 过程做了一个会自己遗忘的 AI——为什么这件事重要"
5. 在 Reddit r/MachineLearning 发"[R] Hawkes-RAG: ..."

**arXiv + GitHub + 三个渠道的内容，是一组合拳。** 任何一个单独发都浪费了其他几个的杠杆。

---

## 六、合并的 4 周时间表

| Week | GitHub | arXiv | 内容 |
|---|---|---|---|
| 1 | 核心代码跑通 | — | — |
| 2 | Benchmark + 参数学习 | 大纲 + Method 草稿 | — |
| 3 | README 打磨 + demo | Experiments + Intro | 录 GIF |
| 4 | 发布 + bugfix | 投 arXiv | 同步发 HN/小红书/Twitter |

**每周末做一次复盘**——不是计划下周，是诚实评估"这周我有没有进度"。如果连续两周延期，立刻砍 scope，**不要把 4 周做成 8 周**。

---

## 七、要警惕的三件事（你的认知盲区）

回顾你过去几个月的记录——拼豆、wellness app、openclaw 复盘——我看到一个**反复的失败模式**：

1. **"评估框架很美"陷阱**：你倾向于把 idea 评估到非常细的颗粒度，然后**没动手**。这个项目的最大风险不是它对不对，是你又分析三周不写代码。这次你已经做了一次正确的反驳（"商业化不看好"），**继续保持这种反驳精神，但要把它用在过程中，不要用在动手前**。

2. **engineering 偏好的复发**：你已经知道自己有这个偏向。GitHub repo 是工程产出，arXiv 是工程产出，**这两件事都正中你舒适区**——这次反而是好事，但你要警惕用"代码完美主义"代替"发布"。第三周末必须发出去，哪怕代码还想再改。

3. **"等准备好"陷阱**：你自己总结过"先列出来，后准备"。**第一周必须有 commit，第二周必须有可运行 example，第三周必须发布。** 拖到第四周还没发布，说明你又在 evaluate 而不是 ship。

如果第二周你回来问我"再帮我评估一下数学公式的细节"——我会知道你又在拖。
如果第二周你回来说"benchmark 跑出来了，互激 ablation 提升 4.7%"——你已经赢了。

---

## 八、一些后续讨论

Q1：我现在要开始做github repo，是做一个带有这个记忆系统的agent，还是直接做这个记忆系统？如果直接做这个记忆系统的话，是不是不好demo

A1：**必做（无悬念，没做就没 paper，没法发布）**

1. λ(t) 核心计算：自激 + 互激 + 指数衰减
2. Memory store：add / retrieve / 更新 last_accessed
3. 检索打分 = 余弦相似度 × λᵢ(t)
4. α 矩阵的**低秩参数化**（绝不能学满 N² 个参数，否则 N=1000 就崩了）
5. MLE 参数估计（α、β 至少一个，最好两个）
6. **关键 ablation：α 设为对角阵（无互激）vs 满矩阵** —— 这是整个论文的灵魂图，它直接验证你的核心 claim
7. LoCoMo 主结果
8. α 热力图 + λ(t) 曲线两张图
9. 1 个 chat example，接 Claude/OpenAI API
10. README + demo GIF

---

**待定（这周内必须做决定，文档里你都没明说）**

1. **Memory 粒度** —— 一条 memory 是什么？一句对话？一个抽取出的"原子事实"（Mem0 路线）？一个 passage chunk？你的文档**完全没提**这个，但它决定 α 矩阵规模和整个系统行为。
   - 倾向：学 Mem0，做原子事实抽取。否则 α 矩阵会被同义重复污染，热力图也看不出概念簇。

2. **Event 触发条件是什么？** λ 跳一下的时机你没定义清楚。只有"被 RAG 检索到"算？还是"在用户输入或 LLM 输出里被提及"也算？这一条直接决定系统动力学。
   - 倾向：双触发，但权重不同（检索 1.0，提及 0.3）。必须明确写进论文 Method。

3. **β 衰减参数化** —— 标量？2-3 个时间尺度（短/中/长）？满矩阵 βᵢⱼ？满矩阵不可学。
   - 倾向：第一周标量起步，如果有时间再升级到 2-3 个 time scale。

4. **α 初始化先验** —— 自激（对角线）默认比互激强多少？影响开箱手感。
   - 倾向：αᵢᵢ ≈ 0.5–1.0，αᵢⱼ = max(0, cos(eᵢ,eⱼ) - 0.3)，整体缩放保证 ρ(α) < 1。

5. **稳定性：只检查还是强制？** 谱半径条件 ρ(α) < 1 你想作为论文卖点。如果代码里不强制（每次更新做谱半径投影），审稿人会问"那你怎么保证？" —— 这时候卖点就站不住。
   - 倾向：要么强制，要么砍掉 stability 章节。**别两头不到岸。**

6. **存储层** —— 纯内存？SQLite？接 Chroma/Qdrant？
   - 倾向：第一版纯内存，README 留一个"pluggable backend on roadmap"的钩子。

7. **Mem0 baseline 的 Plan B** —— 你文档自己承认 Mem0 装起来不简单。三天没装上的备份方案？
   - 倾向：Plan B = naive RAG + 单变量 Ebbinghaus 衰减。这其实**更干净**——直接对比"单变量 Hawkes vs 多变量 Hawkes"，凸显互激矩阵的贡献，正中你论文的核心。

8. **嵌入模型选哪个** —— OpenAI text-embedding-3-small？BGE？sentence-transformers？影响复现成本和 benchmark 钱袋。
   - 倾向：本地的 BGE 或 sentence-transformers，零成本可复现。

---

**可砍 / 别上来就做（陷阱区）**

1. **LoRA 巩固通道** —— 文档自己写了"放 Discussion"。**别动。** 真做是另一篇论文。
2. **持久化层** —— 内存够了。
3. **LangChain/LlamaIndex 深度集成** —— 一个 adapter 文件解决，别搞 plugin 工程。
4. **多 benchmark** —— LongMemEval、MemoryBench 都不要碰。LoCoMo 一个够。
5. **完整稳定性证明** —— 引用 Hawkes 经典文献的现成结果，别自己从头证。
6. **Web UI** —— matplotlib 出图够了，别搞 Plotly dashboard。
7. **类型完美主义** —— mypy strict 模式那种，先发出去再说。

---

最后再扎一刀：

你最大的风险不是"哪个功能没做"，而是**第 1-8 项"待定"开始反客为主，蚕食必做的时间。**

具体到行为：第二周末如果你还在纠结"嵌入模型用 BGE 还是 OpenAI"——说明你在用工程口味问题逃避真问题。

**真问题是第 6 个 ablation 跑出来了吗？**

那个 ablation——互激矩阵开/关的对比——是整个 repo 和论文的命门。它的数字要是不好看，整个故事就塌了。**你越早跑出那个数字，越早知道自己有没有论文。** 第二周中段就必须有它，最迟不能拖过第二周末。

其他所有"待定"项，都可以妥协；这一个数字不行。

------

Q2：MLE 用什么训练信号？第一版 MLE 先基于 synthetic/event logs，还是直接围绕 LoCoMo 设计？

A2：先把问题拆一刀——你这一句里藏了两个独立问题：

1. **MLE 优化什么？**（信号本身的形式）
2. **事件日志哪来？**（synthetic vs LoCoMo）

混着想会卡住。

---

**问题一：MLE 信号是什么。**

Hawkes 的 MLE 是**无监督的**，没有"标签"。目标是事件序列在当前 (α, β, μ) 下的对数似然：

```
log L = Σₖ log λ_{iₖ}(tₖ) − Σᵢ ∫₀ᵀ λᵢ(s) ds
```

你只需要 (timestamp, memory_id) 序列。没有"这次检索是不是相关"的人工标签。

**这一点很关键**：意味着你不需要一个 labeled dataset，只需要 event log。这把你从"找数据集"陷阱里救出来了。

---

**问题二：日志哪来。这不是 OR，是 sequence。**

强烈建议：**synthetic 先，LoCoMo 后。中间不要跳。**

理由是 debugging 隔离。直接上 LoCoMo，万一参数估出来鬼畜，你不知道是
- (a) 优化器有 bug
- (b) 事件抽取定义有问题
- (c) Hawkes 对真实数据拟合就是不好

三个 failure mode 缠在一起 debug 是地狱。Synthetic 把 (a) 单独验掉。

具体流程（TPP 论文标配）：
- **Day 1**：Ogata thinning 从已知 (α_true, β_true, μ_true) 生成 10k 事件
- **Day 2**：拟合，验证 ‖α̂ − α_true‖ < ε
- 过了，MLE 代码可信，顺手拿到论文 Method 节的 recovery plot——白送的图

---

**LoCoMo 的真正难点其实在 MLE 之前。**

估计参数前必须先决定：**在 LoCoMo 里什么算 event？**——上一轮的"待定 #2"，不定就跑不动。粗粒度路径：
1. LLM 从每条消息抽原子记忆（Mem0 路线）
2. 在后续消息里识别记忆被"引用"（语义相似度阈值或 LLM 判断）
3. 引用 = event，时间戳 = 消息时间

每段对话 = 一条独立 trajectory。MHP 标准做法是多 trajectory pool likelihood。**这一步是设计活，不是工程活——你必须自己定义清楚，写进论文 Method。**

---

**一个容易踩的坑**：

在 LoCoMo 上拟合 → 在 LoCoMo 上 eval = 过拟合。两种正确做法：
- 按对话切 train / test（简单）
- 用 held-out predictive log-likelihood：每个事件用其历史预测，eval 留出事件的 log-prob（TPP 文献硬通货，更可信）

后者推荐。审稿人看到前者会皱眉。

---

收尾：

你这问题没问错，但**问早了**。文档里 estimation.py 是 Day 11-13 的事，core.py 的 `intensity()` 是 Day 1-2 的事。

如果 `intensity()` 已经写完——那很好，下一步就是 synthetic recovery test，**这就是验证 intensity() 对不对的方式，刚好把两件事咬合起来**。

如果还没写，那 MLE 不是你现在该想的问题。回去先把 λ(t) 在已知 α、β 下能正确算出来。**估计参数的前提是参数对应的函数是对的。**


*文档结束。从这里往下，要么开 IDE，要么关掉。*

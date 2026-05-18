# 基于使用动力学的 Agent 记忆系统架构图

本文把 `originidea.md` 的 λ 动力学机制和 `memory-system-design-v2.md` 的系统设计合并成几张图。核心直觉是：记忆不是被 LLM 高频整理出来的，而是先作为 turn 粒度片段进入统一记忆池；白天靠确定性中间件做召回、衰减、激励和矛盾降权；夜间再用 dreaming 对高价值幸存者做低频巩固与裁决。

## 1. 总体架构

```mermaid
flowchart TB
    User["用户 turn"] --> Middleware["记忆中间件<br/>Recall / Inject / Update"]
    Middleware --> Agent["主 Agent / 主 LLM"]
    Agent --> Answer["用户可见回答"]
    Agent --> Middleware

    subgraph Store["单一 MemoryStore：turn 粒度原始片段池"]
        M1["片段原文"]
        M2["embedding"]
        M3["lambda 活跃度"]
        M4["type_class / beta"]
        M5["创建与调用时间"]
    end

    Middleware <--> Store

    subgraph AlwaysOn["天然上下文"]
        MemoryMd["MEMORY.md<br/>高价值长期记忆<br/>常驻 system prompt"]
    end

    MemoryMd --> Agent

    subgraph Offline["离线 dreaming pass"]
        Dream["抽取 / 固化 / 矛盾裁决"]
    end

    Store --> Dream
    Dream --> MemoryMd
    Dream --> Store
```

要点：

- 只有一个主 Agent，记忆系统是夹在 Agent 与存储之间的无状态确定性中间件。
- MemoryStore 不分冷热库，冷热只是同一池内 λ 高低形成的连续光谱。
- MEMORY.md 是最高层长期记忆，来自夜间 dreaming 的低频巩固。

## 2. 日间在线运行流程

```mermaid
flowchart LR
    A["START<br/>收到 user turn"] --> B["recall_node<br/>零 token 召回"]
    B --> B1["计算 embedding 相似度 cos"]
    B1 --> B2["读取 lambda 并自然衰减"]
    B2 --> B3["按熵计算 mu"]
    B3 --> B4["score = cos * [mu + (1-mu)lambda]"]
    B4 --> B5["tau 阈值 + 自适应注入量"]

    B5 --> C["main_llm_node<br/>纯净回答用户"]
    C --> D["采纳检测<br/>回答与片段重叠 / 语义使用"]

    C --> E["零 token 矛盾预筛<br/>高 cos 但未被采纳"]
    E -->|低于 theta_c| G["adoption_update_node"]
    E -->|超过 theta_c| F["contradiction_micro_node<br/>轻量二次调用<br/>只判定哪些旧片段被推翻"]

    F --> F1["负向更新<br/>lambda+ = lambda- * (1 - s_c)"]
    F1 --> G
    D --> G
    G --> G1["正向更新<br/>lambda+ = lambda- + score * (1-lambda-)"]
    G1 --> H["END<br/>回写 MemoryStore"]
```

这张图对应 LangGraph 的 5 节点结构：`recall_node -> main_llm_node -> contradiction_micro_node? -> adoption_update_node`。矛盾节点只在预筛报警时触发，因此日常路径仍然接近零额外 token。

## 3. λ 活跃度状态机

```mermaid
stateDiagram-v2
    [*] --> Created: 写入新 turn
    Created --> Decaying: lambda(t0)=1

    Decaying --> Candidate: 召回打分进入候选<br/>score > tau
    Decaying --> Decaying: 未进入候选<br/>lambda *= exp(-beta * delta_t)

    Candidate --> Adopted: 被主回答实际采纳
    Candidate --> Contradicted: 被当前用户陈述推翻
    Candidate --> Ignored: 注入但未采纳且无矛盾

    Adopted --> Decaying: 正向激励<br/>lambda+ = lambda- + score(1-lambda-)
    Contradicted --> Decaying: 负向抑制<br/>lambda+ = lambda-(1-s_c)
    Ignored --> Decaying: 不激励<br/>继续自然衰减

    Decaying --> DreamingCandidate: 高 lambda 且反复采纳<br/>或反复被矛盾
    DreamingCandidate --> Consolidated: dreaming 裁决
    Consolidated --> Decaying: 写入 MEMORY.md<br/>原片段可沉降
```

核心闭环：

- 越被真正使用，λ 越强。
- 越久不用，λ 自然衰减。
- 被新事实推翻，λ 主动下沉，但原文不删除，仍可被冷召回或夜间裁决使用。

## 4. 三层召回与注入策略

```mermaid
flowchart TB
    Query["当前 query / user turn"] --> L1["第 1 层：MEMORY.md<br/>常驻 system prompt<br/>不检索"]
    Query --> L2["第 2 层：热召回<br/>score = 语义相似度 × 活跃度混合项"]
    Query --> L3{"热召回 top-1<br/>是否低于阈值?"}
    L3 -->|否| Inject["注入当前 turn 上方"]
    L3 -->|是| Cold["第 3 层：冷召回<br/>绕过 lambda<br/>全池纯语义检索"]
    Cold --> Inject

    L2 --> Gate{"score >= tau?"}
    Gate -->|是| Inject
    Gate -->|否| Drop["丢弃<br/>避免污染上下文"]

    Inject --> Adaptive["自适应注入量<br/>断崖领先少塞<br/>分数平缓多塞<br/>全低于 tau 不塞"]
    Adaptive --> MainLLM["主 LLM 回答"]
```

这里的关键不是“永远塞 top-k”，而是允许系统沉默。低相关、低活跃、过时的片段会被 τ 阈值挡掉；如果用户刻意找旧信息，冷召回仍能兜底。

## 5. 评测与消融架构

```mermaid
flowchart LR
    Dataset["评测集<br/>LoCoMo / LongMemEval"] --> Harness["Benchmark Harness"]
    Harness --> System["Full System<br/>LangGraph + MemoryStore"]
    Harness --> Baselines["Baselines<br/>Mem0 / Zep / full-context / pure RAG"]

    subgraph SystemNodes["可替换节点"]
        R["recall_node<br/>lambda / mu / tau / cold recall"]
        L["main_llm_node"]
        C["contradiction_micro_node"]
        U["adoption_update_node"]
        D["dreaming_pass"]
    end

    System --> SystemNodes

    SystemNodes --> Metrics["必报指标<br/>accuracy<br/>每 query LLM 调用数<br/>每 query token 数"]

    subgraph Ablation["类别 - 机制消融"]
        A1["knowledge update<br/>关 contradiction_micro / 关 dreaming 裁决"]
        A2["temporal reasoning<br/>单一 beta"]
        A3["abstention<br/>固定 top-k 且无 tau"]
        A4["multi-session<br/>关冷召回 / 关 MEMORY.md"]
        A5["single-session recall<br/>score 退化为纯 cos"]
    end

    SystemNodes --> Ablation
    Baselines --> Metrics
    Ablation --> Metrics
```

评测设计的重点是把“准确率”和“成本曲线”一起报。这个系统的主张不是只追求分数，而是证明：日间大部分记忆操作是确定性零 token，只有稀疏矛盾分诊和低频 dreaming 消耗 LLM。

## 6. 机制速览

```mermaid
flowchart TD
    New["新 turn 写入"] --> Init["lambda = 1"]
    Init --> Decay["按事实类型 beta 自然衰减"]

    Decay --> Recall["召回时计算 score"]
    Recall --> Score["score = cos(q,m) * [mu + (1-mu)lambda]"]
    Score --> Entropy["mu 由 lambda 分布熵决定"]

    Score --> Select["过 tau 的片段进入上下文"]
    Select --> Adopt["被采纳"]
    Select --> Reject["被推翻"]
    Select --> Silent["未使用"]

    Adopt --> Up["正向激励<br/>向 1 靠近但有阻尼"]
    Reject --> Down["负向抑制<br/>按相似度复利下沉"]
    Silent --> Decay

    Up --> Decay
    Down --> Decay
```

一句话版：这是一个让记忆“越用越强、越不用越弱、被推翻则主动降权”的动态 RAG 系统。

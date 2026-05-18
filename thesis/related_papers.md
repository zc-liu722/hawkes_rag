# OriginIdea 相关论文与工作目录

更新时间：2026-05-18  
本目录围绕 `originidea.md` 中的“记忆激励与自衰减机制”整理。核心问题可以概括为：在传统向量 RAG 的语义相似度之外，为每条记忆维护一个随时间衰减、随成功调用增强的活跃度 `lambda(t)`，用它改写检索排序，使记忆具有“越用越强、越不用越弱”的动态性质。

## 0. 阅读地图

建议把相关工作分成六条线：

1. **RAG 与检索增强基础**：证明 OriginIdea 的改动点不是“是否检索”，而是“检索排序如何带动态记忆状态”。
2. **长期对话记忆 benchmark**：决定实验该打哪些公开任务，尤其是 multi-session、temporal reasoning、knowledge update、adversarial distractor。
3. **Agent memory 系统**：把 OriginIdea 放到 MemGPT、Generative Agents、A-MEM、Zep、Mem0 等系统旁边比较。
4. **Temporal / time-aware RAG**：区分“时间过滤、近因偏置、时间图谱”和 OriginIdea 的“使用激励 + 自衰减”。
5. **冲突、版本化与结构化记忆**：当前实验里 `update_override` 和 semantic distractor 是弱项，这条线对方法 v2 最关键。
6. **认知记忆与时间点过程**：为 `lambda(t)`、指数衰减、Hawkes-like 激励提供理论语言。

## 1. 最优先精读清单

| 优先级 | 论文 / 工作 | 主要内容 | 为什么和 OriginIdea 强相关 | 链接 |
|---|---|---|---|---|
| P0 | LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory | 提出长期交互记忆 benchmark，覆盖信息抽取、多 session 推理、时间推理、知识更新、拒答等能力。 | OriginIdea 已在 LongMemEval-S 上测试；其 `knowledge-update` 与 `temporal-reasoning` 正好对应新旧事实竞争和时间有效性。 | https://arxiv.org/abs/2410.10813 |
| P0 | Evaluating Very Long-Term Conversational Memory of LLM Agents / LoCoMo | 评估超长多轮、多 session 对话记忆，任务含 QA、事件总结、多模态对话生成。 | OriginIdea 在 LoCoMo 上收益很小；这篇是必须解释“为什么动态偏置对 aggregation 不一定有效”的核心 benchmark。 | https://arxiv.org/abs/2402.17753 |
| P0 | MemGPT: Towards LLMs as Operating Systems | 用操作系统式虚拟上下文管理，把上下文分层为主上下文、外部上下文/归档记忆，并让模型主动管理读写。 | Agent memory 经典基线；OriginIdea 可定位为“检索排序层的动态记忆状态”，而不是完整 OS-style memory manager。 | https://arxiv.org/abs/2310.08560 |
| P0 | Generative Agents: Interactive Simulacra of Human Behavior | 以自然语言保存 agent 经验，使用检索、反思、计划构建可信行为模拟。 | 它的 memory retrieval 结合 recency、importance、relevance，是 OriginIdea 的直接前史之一。 | https://arxiv.org/abs/2304.03442 |
| P0 | A-MEM: Agentic Memory for LLM Agents | 用 Zettelkasten 思路和 agent-driven decision 动态组织记忆，构建可演化的记忆网络。 | OriginIdea 目前主要是动态排序；A-MEM 提醒需要在写入、链接、组织阶段也引入动态机制。 | https://arxiv.org/abs/2502.12110 |
| P0 | Zep: A Temporal Knowledge Graph Architecture for Agent Memory | 用 Graphiti temporal knowledge graph 维护 agent memory，强调动态知识整合、时间关系与企业场景。 | 对 OriginIdea 的 `update_override` 弱项很重要：旧事实覆盖新事实不能只靠衰减，可能需要 temporal KG / conflict set。 | https://arxiv.org/abs/2501.13956 |
| P0 | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | RAG 原始代表作，将参数记忆与非参数检索记忆结合用于知识密集型生成。 | 作为 baseline 与相关工作开头；OriginIdea 是在标准 RAG 检索得分上加入动态 memory state。 | https://arxiv.org/abs/2005.11401 |
| P0 | The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process | 用连续时间 LSTM 建模自激励/抑制事件流，扩展 Hawkes process。 | OriginIdea 的“召回后增强、未召回衰减”可借用 self-exciting temporal point process 的理论语言。 | https://arxiv.org/abs/1612.09328 |

## 2. RAG 基础与强检索基线

| 论文 / 工作 | 主要内容 | 可借鉴点 | 链接 |
|---|---|---|---|
| Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | 将检索文档作为非参数记忆，与 seq2seq 生成结合。 | 作为“静态 RAG”起点；OriginIdea 的贡献应写成检索得分与记忆状态的动态化。 | https://arxiv.org/abs/2005.11401 |
| REALM: Retrieval-Augmented Language Model Pre-Training | 在预训练中引入可微检索器，让模型学习访问外部知识。 | 对比“训练检索器”路线；OriginIdea 更像 inference-time / memory-state reranking。 | https://arxiv.org/abs/2002.08909 |
| Dense Passage Retrieval for Open-Domain Question Answering | DPR 用双塔 dense retriever 做开放域 QA 检索。 | 必备 dense retrieval 基线；也可作为 embedding-only 静态检索的理论背景。 | https://arxiv.org/abs/2004.04906 |
| Fusion-in-Decoder | FiD 把多个检索段分别编码，在 decoder 融合证据。 | 对多证据聚合很关键；OriginIdea 需要避免动态偏置伤害 multi-evidence coverage。 | https://arxiv.org/abs/2007.01282 |
| Atlas: Few-shot Learning with Retrieval Augmented Language Models | 大规模预训练 retrieval-augmented LM，在少样本知识任务上很强。 | 强检索增强基线；可用于说明检索质量与生成质量的耦合。 | https://arxiv.org/abs/2208.03299 |
| REPLUG: Retrieval-Augmented Black-Box Language Models | 不改黑盒 LM，只将检索结果拼接进 prompt，并用 LM 反馈调 retriever。 | OriginIdea 也适合黑盒 LLM；可把 `lambda` 看作一个轻量、可解释的 retriever prior。 | https://arxiv.org/abs/2301.12652 |
| HyDE: Precise Zero-Shot Dense Retrieval without Relevance Labels | 先让 LLM 生成 hypothetical document，再用其 embedding 检索真实文档。 | 可作为 query expansion / query rewriting 强基线，尤其对语义干扰任务重要。 | https://arxiv.org/abs/2212.10496 |
| FLARE / Active Retrieval Augmented Generation | 生成过程中根据低置信预测主动检索未来需要的内容。 | 对比“一次检索”与“生成过程动态检索”；OriginIdea 是记忆状态动态，FLARE 是 query 时机动态。 | https://arxiv.org/abs/2305.06983 |
| Self-RAG | 训练模型判断是否需要检索、评估检索内容并自我反思。 | 可作为检索后验证器方向；OriginIdea v2 可加入“retrieval usefulness feedback”更新 `lambda`。 | https://arxiv.org/abs/2310.11511 |
| CRAG: Corrective Retrieval Augmented Generation | 对检索结果做质量评估，低质量时触发纠错/补检索。 | 适合加入“检索后 verifier”，避免弱相关记忆因活跃度过高而被持续强化。 | https://arxiv.org/abs/2401.15884 |
| RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval | 对文本块递归聚类、摘要，形成树状检索结构。 | 对 OriginIdea 的多粒度记忆有启发：turn/session/fact/summary 的 `lambda` 可分层维护。 | https://arxiv.org/abs/2401.18059 |
| In Defense of RAG in the Era of Long-Context Language Models / OP-RAG | 研究长上下文模型时代 RAG 的价值，提出保持顺序的检索增强。 | 对 LoCoMo/长对话很重要；提示 top-k 越多不一定越好，应关注顺序和覆盖。 | https://arxiv.org/abs/2409.01666 |

## 3. 长期对话记忆 Benchmark

| 论文 / 工作 | 主要内容 | 对 OriginIdea 的用法 | 链接 |
|---|---|---|---|
| LongMemEval | 长期交互记忆 benchmark，覆盖 extraction、multi-session reasoning、temporal reasoning、knowledge update、abstention。 | 继续作为公开主 benchmark；建议按类别报告 Recall/MRR/SRR，并加入 answer generation。 | https://arxiv.org/abs/2410.10813 |
| LoCoMo | 超长 multi-session conversational memory benchmark，含事实、时间、多 session 聚合、推理、干扰项。 | 用于证明方法不伤害 aggregation；如果收益小，要解释任务结构与检索粒度。 | https://arxiv.org/abs/2402.17753 |
| LongMemEval-V2: Evaluating Long-Term Agent Memory Toward Experienced Colleagues | 面向 web agent 经验记忆，包含 static state recall、dynamic state tracking、workflow knowledge、environment gotchas、premise awareness，历史轨迹可达 115M tokens。 | 这是 2026 年很新的强相关 benchmark；适合未来证明 OriginIdea 是否能服务“经验型记忆”，不只是用户事实记忆。 | https://arxiv.org/abs/2605.12493 |
| LoCoMo-Plus: Beyond-Factual Cognitive Memory Evaluation Framework for LLM Agents | 关注 cue-trigger semantic disconnect 下的认知记忆，即后续问题不显式复述约束，但模型应记住用户状态/目标/价值。 | 对 OriginIdea 的“reactivation”类别很相关：早期低频记忆如何被间接 cue 唤醒。 | https://arxiv.org/abs/2602.10715 |
| BEAM / Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs | 构造最高 10M tokens 的长对话与 2000 个验证问题，并提出 LIGHT 记忆框架。 | 用于避免 LongMemEval-S 可被长上下文“塞进去”的质疑；可测试真正超上下文规模的 memory retrieval。 | https://arxiv.org/abs/2510.27246 |
| CorpusQA: A 10 Million Token Benchmark for Corpus-Level Analysis and Reasoning | 扩展到 10M tokens 的 corpus-level 分析推理 benchmark。 | 对多证据聚合和全局信息合成有用，能检验 OriginIdea 是否过度偏向单证据召回。 | https://arxiv.org/abs/2601.14952 |

## 4. Agent Memory 与长期记忆系统

| 论文 / 工作 | 主要内容 | 与 OriginIdea 的关系 | 链接 |
|---|---|---|---|
| MemGPT | OS-inspired virtual context management，分层管理上下文和归档记忆。 | 强系统基线；OriginIdea 可作为 archival memory retrieval scoring 的一个模块。 | https://arxiv.org/abs/2310.08560 |
| Generative Agents | 保存完整经验、检索相关记忆、生成反思并用于计划。 | 其 relevance/recency/importance 检索打分是 OriginIdea 的近亲；OriginIdea 可以更细地建模 activation dynamics。 | https://arxiv.org/abs/2304.03442 |
| Reflexion: Language Agents with Verbal Reinforcement Learning | agent 将失败反馈写成语言反思，存入 episodic memory，用于后续尝试。 | “成功/失败反馈如何更新记忆强度”的代表；OriginIdea 可以把 positive call、negative call、verified useful 分开更新。 | https://arxiv.org/abs/2303.11366 |
| MemoryBank: Enhancing Large Language Models with Long-Term Memory | 为 LLM companion 建长期记忆库，使用 Ebbinghaus 遗忘曲线等机制管理记忆。 | 与 OriginIdea 的遗忘曲线/记忆衰减非常直接相关，应精读其 memory updating 与 retrieval 设计。 | https://arxiv.org/abs/2305.10250 |
| LongMem: Augmenting Language Models with Long-Term Memory | 冻结 backbone，把长历史缓存为可检索 memory bank，由 side-network 检索/读取。 | 对“模型内 memory augmentation”路线有代表性；OriginIdea 是更轻量的外部记忆排序机制。 | https://arxiv.org/abs/2306.07174 |
| RET-LLM: Towards a General Read-Write Memory for Large Language Models | 为 LLM 加通用读写记忆单元，用三元组抽取、存储和召回知识，强调可更新与可解释。 | 对 fact-level memory、可更新记忆、时间问答很有价值，适合补足 OriginIdea 的 turn/session 粒度不足。 | https://arxiv.org/abs/2305.14322 |
| Memory^3: Language Modeling with Explicit Memory | 将知识外化为 explicit memory，探索比参数和文本 RAG 更便宜的记忆格式。 | 可帮助讨论“记忆表示”而不只是“记忆排序”。 | https://arxiv.org/abs/2407.01178 |
| A-MEM | 通过 Zettelkasten 式链接和 agent-driven decision 动态组织记忆。 | 直接竞争方向；OriginIdea 可吸收其 memory linking / evolution 思路。 | https://arxiv.org/abs/2502.12110 |
| Zep | temporal KG / Graphiti 管理 agent memory，支持历史关系和动态知识整合。 | 解决 update、conflict、validity interval 的强参考。 | https://arxiv.org/abs/2501.13956 |
| Mem0 | 面向生产 agent 的可扩展长期记忆架构，动态抽取、整合、检索 salient information，并有 graph memory 版本。 | 强工程系统基线；适合比较 latency、token cost、LoCoMo 表现。 | https://arxiv.org/abs/2504.19413 |
| MIRIX | 多 agent 记忆系统，包含 Core、Episodic、Semantic、Procedural、Resource Memory、Knowledge Vault 等记忆类型。 | 提示 OriginIdea 可以从“单一记忆池”扩展为多类型记忆池，各自有不同衰减/激励规则。 | https://arxiv.org/abs/2507.07957 |
| LightMem | 用小语言模型驱动轻量 agent memory，模块化 retrieval、writing、long-term consolidation，区分在线调用和离线整合。 | 对低成本可部署方向重要；OriginIdea 的 `lambda` 更新也应考虑在线/离线分离。 | https://arxiv.org/abs/2604.07798 |
| MemReranker: Reasoning-Aware Reranking for Agent Memory Retrieval | 面向 agent memory retrieval 的推理感知 reranker，通过蒸馏构建小模型。 | 可能是 OriginIdea 的强 rerank baseline；如果只靠公式赢不过 reranker，需要转向可解释/低成本/可组合优势。 | https://arxiv.org/abs/2605.06132 |

## 5. Temporal RAG、时间有效性与新旧事实

| 论文 / 工作 | 主要内容 | 与 OriginIdea 的关系 | 链接 |
|---|---|---|---|
| FreshLLMs: Refreshing Large Language Models with Search Engine Augmentation | 用搜索引擎增强应对动态世界知识和 false premise，提出 FreshQA。 | 强调“事实会过期”；OriginIdea 的 `lambda` 不能等同于 truth freshness，需要区分活跃度和事实有效性。 | https://arxiv.org/abs/2310.03214 |
| TimeR4: Time-aware Retrieval-Augmented Large Language Models for Temporal Knowledge Graph QA | 面向 temporal KG QA 的 time-aware RAG。 | 对 temporal validity、时间约束过滤、时间图谱问答有帮助。 | https://aclanthology.org/2024.emnlp-main.394/ |
| DeFuzzRAG: Handling Fuzzy Time Expressions for Temporal Robustness in RAG | 处理 “a few years later” 等模糊时间表达，通过小模型推断时间范围并用 metadata filtering 对齐时间意图。 | 对 OriginIdea 的 temporal-reasoning 很重要；可作为 time-aware retrieval baseline。 | https://ojs.aaai.org/index.php/AAAI/article/view/40276 |
| Temporal GraphRAG / RAG Meets Temporal Graphs | 将外部语料建成双层 temporal graph，包含 temporal KG 与 hierarchical time graph。 | 对“时间图谱 + RAG”强相关，可补足 OriginIdea 缺少结构化时间关系的问题。 | https://arxiv.org/abs/2510.13590 |
| In Defense of RAG in the Era of Long-Context Language Models | 提出 order-preserve RAG，说明长上下文和 RAG 的质量/成本取舍。 | 对 long conversation retrieval 特别重要；OriginIdea 的 top-k 选择也应保留时间顺序和证据覆盖。 | https://arxiv.org/abs/2409.01666 |

## 6. 冲突、覆盖与结构化记忆方向

这些工作不一定都直接叫“memory”，但对 OriginIdea 目前最弱的 `update_override`、semantic distractor、多证据聚合很关键。

| 论文 / 工作 | 主要内容 | 可转化为 OriginIdea v2 的点 | 链接 |
|---|---|---|---|
| RET-LLM | 把知识抽取成 triplets，并支持 scalable、aggregatable、updatable、interpretable memory。 | 将 turn-level memory 拆成 fact/event triples，给每个 fact 建 validity interval 和 conflict relation。 | https://arxiv.org/abs/2305.14322 |
| Zep / Graphiti | temporal KG 维护历史关系，动态整合对话与业务数据。 | 建 `conflict_set`、`supersedes`、`valid_from/valid_to`，不要只靠旧事实衰减。 | https://arxiv.org/abs/2501.13956 |
| OG-RAG: Ontology-Grounded Retrieval Augmented Generation | 用领域 ontology anchor retrieval，保留实体关系。 | 对 semantic distractor 有启发：query 的目标实体/事件类型应参与检索打分。 | https://arxiv.org/abs/2412.15235 |
| RAPTOR | 树状摘要与检索，兼顾局部细节和高层摘要。 | 用分层记忆解决 multi-session aggregation：早期证据不能因低活跃度完全沉底。 | https://arxiv.org/abs/2401.18059 |
| FiD | decoder 融合多个检索证据。 | 如果检索 top-k 内包含多证据，生成端也要能融合；检索指标之外要做 answer generation。 | https://arxiv.org/abs/2007.01282 |
| CRAG | 通过 evaluator 判断检索质量并纠错。 | 检索后 verifier 可决定本轮召回是否真的“成功调用”，避免把噪音记忆强化。 | https://arxiv.org/abs/2401.15884 |
| Self-RAG | 模型自我判断 retrieval 需求和 evidence usefulness。 | 可以把 `lambda` 的更新从“进入 top-k 就强化”改成“被验证有用才强化”。 | https://arxiv.org/abs/2310.11511 |

## 7. 认知记忆、遗忘曲线与 Hawkes-like 动力学

| 论文 / 工作 | 主要内容 | 与 OriginIdea 的关系 | 链接 |
|---|---|---|---|
| Spectra of Some Self-Exciting and Mutually Exciting Point Processes | Hawkes process 经典源头，定义自激励/互激励点过程。 | OriginIdea 的“召回事件提高未来召回强度”可借用 Hawkes process 语言，但要小心它目前不是严格点过程建模。 | https://doi.org/10.1093/biomet/58.1.83 |
| The Neural Hawkes Process | 用神经网络建模事件流的自激励和抑制。 | 可启发 learned activation dynamics：不同记忆类型、查询类型对应不同衰减/激励曲线。 | https://arxiv.org/abs/1612.09328 |
| MemoryBank | 在 LLM companion memory 中显式使用遗忘曲线与用户画像。 | 直接对照 OriginIdea 的指数衰减；应比较它如何定义 importance、decay、recall。 | https://arxiv.org/abs/2305.10250 |
| Replication and Analysis of Ebbinghaus' Forgetting Curve | 复现/分析 Ebbinghaus 遗忘曲线，讨论不同拟合形式。 | 支持 `lambda(t)` 的心理学动机；但论文中应避免声称指数衰减就是唯一正确的人类记忆模型。 | https://doi.org/10.1371/journal.pone.0120644 |
| Forgetting Curves: Implications for Connectionist Models | 讨论遗忘曲线形态与连接主义模型，指出单纯指数曲线并非总是最好。 | 提醒 OriginIdea 可以比较 exponential、power-law、sum-of-exponentials 等 decay kernel。 | https://doi.org/10.1016/S0010-0285(02)00012-9 |
| Ephemerally Self-Exciting Point Process | 提出短暂自激励点过程的变体。 | 与“被触发后短期更容易再次出现，但激励会消散”的记忆活跃度很像。 | https://arxiv.org/abs/1811.04282 |

## 8. 推荐实验与论文写作对应关系

| OriginIdea 问题 | 应优先读 / 对比的工作 | 建议实验 |
|---|---|---|
| 证明不是普通 recency trick | Generative Agents、DeFuzzRAG、TimeR4、FreshLLMs | 加 recency-only、cosine+recency、metadata time filtering、temporal query expansion。 |
| `update_override` 弱 | LongMemEval、Zep、RET-LLM、Temporal GraphRAG | 加版本化记忆：`supersedes(old_fact, new_fact)`、validity interval、conflict set；单独报告 knowledge-update。 |
| semantic distractor 弱 | OG-RAG、RET-LLM、HyDE、reranker 工作 | 加 entity/event binding；query entity 与 memory entity 不匹配时降权。 |
| multi-session aggregation 可能受损 | LoCoMo、BEAM、FiD、RAPTOR、OP-RAG | 从独立 top-k 排序改成 coverage-aware selection，报告 evidence coverage 而不只 MRR。 |
| `lambda` 更新过于经验公式 | Hawkes 1971、Neural Hawkes、MemoryBank、遗忘曲线文献 | 做 decay kernel ablation：exponential vs power-law vs learned half-life；做 no-excitation/no-decay 消融。 |
| top-k 进入即强化可能强化噪音 | Self-RAG、CRAG、MemReranker | 加 post-retrieval usefulness verifier；只有被 answer 使用或 verifier 判有用才增强。 |

## 9. 论文贡献定位建议

如果继续写成一篇论文，建议不要只说“给 RAG 加一个衰减公式”，而是定位为：

> Side-effect-aware dynamic memory retrieval for long-term conversational agents.

可以组织为三个贡献：

1. **方法贡献**：提出带 activation state 的动态记忆检索框架，支持使用激励、自然衰减、自适应语义/活跃度混合。
2. **机制贡献**：证明只看 positive recall 不够，长期记忆检索还要惩罚 stale/conflicting/distractor memory 的副作用。
3. **实验贡献**：在 LongMemEval、LoCoMo、以及带 positive/negative 标注的自建 benchmark 上，分析 dynamic retrieval 对 reactivation、stability、decay-forget、knowledge-update、semantic-distractor 的不同影响。

## 10. 下一步精读顺序

1. LongMemEval、LoCoMo、LongMemEval-V2、BEAM：先把评测空间摸清。
2. MemGPT、Generative Agents、A-MEM、Zep、Mem0：搞清 agent memory 系统的主流架构。
3. DeFuzzRAG、TimeR4、Temporal GraphRAG、FreshLLMs：把时间相关 baseline 补齐。
4. Hawkes 1971、Neural Hawkes、MemoryBank、Ebbinghaus 相关文献：给公式找理论支撑并设计 decay kernel 消融。
5. Self-RAG、CRAG、MemReranker：为检索后验证与 reranking 做强基线。


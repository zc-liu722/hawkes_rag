# 基于使用动力学的 Agent 记忆机制
## 含实时矛盾分诊、评测方案与工程落地的完整设计(v3）

---

> **一句话定位**
> 一个单一介质、turn 粒度、写入零 LLM 的记忆系统;白天靠 λ 使用动力学完成零 token 的召回、遗忘筛选与**实时矛盾分诊**,夜间用一次低频 dreaming 只对高 λ 幸存者做巩固与矛盾**裁决**;三种召回源层级兜底,激活信号取"被采纳"而非"被召回"以根除 learning leak,矛盾信号经一个零 token 闸门稀疏触发、只在疑似矛盾时才花一次轻量调用,从而把负反馈补进控制回路而不破坏"日常零 token"。

本文档相对初稿(originidea / memorystructure)的四处实质性增量:

1. **形式化层**:新增与正向激励完全对称的**负向矛盾更新公式**(第一部分 §5)。
2. **系统层**:新增**实时矛盾分诊**一节,采用闸门触发的二次调用方案(方案 C),并据此更新架构原则与差异化定位(第二部分 §4bis、§7、§8)。
3. **评测层**:新增完整的**评测方案**,含基准选择、agent 框架选型(LangGraph + PydanticAI)、系统架构、dreaming 在 benchmark 的触发契约与类别—机制 ablation 矩阵(第三部分)。
4. **工程落地层**:新增**第四部分·工程落地与接入**,把每个接入缺口对到可复用现成件,定义 State schema、混合存储字段、可插拔时钟,并将三个"评测刚需约束"写成硬性条款(C-1 模型钉死、C-2 embedding 冻结、C-3 数据集时间时钟)。这三条不满足时评测结果会错且不可察觉,故同时回填进第一、二部分相关章节。

---

# 第一部分 · 形式化机制

每条记忆表示为向量 $m$,其在时刻 $t$ 的活跃度为 $\lambda(t)\in(0,1]$。数值越大越易被召回;越小说明已逐渐遗忘。

## 1. 记忆召回分数

传统 RAG 仅按当前查询 $q_i$ 与记忆 $m$ 的语义相似度召回。本机制在其上引入活跃度,最终召回分数为:

$$
\text{score}_i=\cos(q_i,m)\cdot\bigl[\mu+(1-\mu)\,\lambda^{-}(t_i)\bigr]
$$

其中 $\lambda^{-}(t_i)$ 为记忆在 $t_i$ **被调用前**的活跃度;$\mu$ 为**噪音系数**,控制活跃度与语义分数的竞争关系,随活跃度分布动态调整。

设检索池记忆条数为 $N$,第 $m$ 条当前活跃度为 $\lambda_m\in(0,1]$。定义归一化概率:

$$
p_m=\frac{\lambda_m^{2}}{\sum_{j=1}^{N}\lambda_j^{2}}
$$

香农熵 $H=-\sum_{m=1}^{N}p_m\ln p_m$(约定 $0\cdot\ln 0=0$),归一化熵 $\hat{H}=H/\ln N$(若 $N=1$ 约定 $\hat{H}=0$)。则:

$$
\mu=\mu_{base}+(1-\mu_{base})\cdot\sqrt{\,1-\hat{H}\,}
$$

默认 $\mu_{base}=0.1$、曲率 $k=\tfrac12$,即 $\mu=0.1+0.9\sqrt{1-\hat{H}}$。

**直觉**:分布越集中(熵低,系统已长出明确偏好)$\mu$ 越大,让语义相似度重新主导;分布越平摊(熵高,系统尚无可信偏好)$\mu$ 越小,让活跃度有发言权。等于让系统自己判断"我的使用历史现在可信吗"。

## 2. 初始化与创建后的衰减

记忆被创建时初始强度 $\lambda(t_0)=1$,创建视为初次激励。首次成功调用前 $t_0<t<t_1$:

$$
\lambda(t)=e^{-\beta(t-t_0)}
$$

$\beta>0$ 为衰减系数,**按事实类型分档**:易变信息 $\beta$ 大,稳定信息 $\beta$ 小,身份类信息 $\beta\to 0$。

## 3. 调用后的正向激励更新

只有 $\text{score}_i>0$ 才可被调用。被成功调用后:

$$
\lambda^{+}(t_i)=\lambda^{-}(t_i)+\bigl[1-\lambda^{-}(t_i)\bigr]\cdot\text{score}_i
$$

$[1-\lambda^{-}(t_i)]$ 是"剩余可激励空间",本身即富者愈富的内生阻尼器:$\lambda$ 越高单次增量越小,系统自动收敛、不会爆炸。

## 4. 两次成功调用之间的自然衰减

上次成功调用时刻为 $t_i$,则 $t_i<t<t_{i+1}$:

$$
\lambda(t)=\lambda^{+}(t_i)\cdot e^{-\beta(t-t_i)}
$$

## 5. 调用后的负向矛盾更新(新增)

**动机**:§3 与 §4 只提供正反馈与被动遗忘,日间回路缺少负反馈——一条"语义高度相似但已过时"的记忆会因话题相关被反复召回、反复采纳、反复 $\lambda^{+}$ 强化(learning leak 的另一个泄漏口)。负向更新把控制回路从"只有正向 + 衰减"补全为"正向 + 负向 + 衰减"的闭环。

当一条召回片段 $m$ 被判定为**被当前用户陈述推翻**(判定机制见第二部分 §4bis),其活跃度执行与 §3 完全对称的下拉:

$$
\boxed{\;\lambda^{+}(t_i)=\lambda^{-}(t_i)\cdot\bigl[\,1-s_c\,\bigr]\,,\qquad s_c=\cos(q_i,m)\;}
$$

**与正向激励的对称性**:激励向上推、以剩余空间 $[1-\lambda^{-}]$ 为阻尼;矛盾向下拉、以 $\lambda^{-}$ 本身为"还能掉多少"的阻尼。两式互为镜像。

**性质**:

- **恒在 $(0,1]$ 内**:$s_c\in[0,1)$ 时 $\lambda^{+}=\lambda^{-}(1-s_c)\in(0,\lambda^{-}]$,不破坏定义域,不像硬置 0 那样不可逆。
- **自带幻觉矛盾过滤**:真正的事实矛盾必然话题高度相关(同主语同属性,$\cos$ 大)→ 惩罚重;主 agent 幻觉出的伪矛盾 $\cos\to 0$ → 惩罚自动趋近 0。无需额外的 embedding 闸门。
- **复利衰减**:被同类陈述反复矛盾时 $\lambda^{-}\cdot(1-s_c)^n$,真过时事实数轮内被埋,单次假阳性不致一击毙命。

**职责边界**:矛盾机制**只做一件事——压低被推翻的旧片段**。"激活被使用的新内容"无需新增机制:用户新陈述本身是一个新 turn,按写入路径 $\lambda$ 初始化为 1,**生来就是热的**;主 agent 实际用以回答的片段走既有 §3 采纳激励。在矛盾机制里再做一次"激活"会双重计数。

## 总结:状态转移与超参数

| 阶段 | 公式 |
|---|---|
| 创建至首次调用前 $t_0<t<t_1$ | $\lambda(t)=e^{-\beta(t-t_0)}$ |
| 召回打分 | $\text{score}_i=\cos(q_i,m)\,[\mu+(1-\mu)\lambda^{-}(t_i)]$ |
| 被采纳(正向) | $\lambda^{+}(t_i)=\lambda^{-}(t_i)+[1-\lambda^{-}(t_i)]\,\text{score}_i$ |
| 被推翻(负向) | $\lambda^{+}(t_i)=\lambda^{-}(t_i)\,[1-s_c],\ s_c=\cos(q_i,m)$ |
| 未调用(衰减) | $\lambda(t)=\lambda^{+}(t_i)\,e^{-\beta(t-t_i)}$ |

超参数:$\beta$(按事实类型分档,需调);$\mu_{base}=0.1$、$k=\tfrac12$(有默认值,深入亦可调);$\tau$(注入阈值,见第二部分 §5);`intermediate_top_k`(每轮取前 k 条视为成功调用);$\theta_c$(矛盾预筛触发阈值,新增,见 §4bis)。

---

# 第二部分 · 系统设计

## 1. 存储介质:单一介质,turn 粒度

所有记忆是统一一个池。记忆单元是一个 turn 或 session 的原始片段,带 embedding,永不拆解成命题,附一个活跃度标量 $\lambda\in(0,1]$。冷热不是两个库,而是同一池子里 $\lambda$ 高低的连续光谱——区别只在召回路径,不在存储。写入 = 片段追加入池 + $\lambda$ 初始化为 1,全程零 LLM。

> **硬性条款 · 存储形态(见第四部分 §3)**:embedding 几乎不变而 $\lambda$ 每轮在变,二者读写频率正交,因此底层是**混合存储**——向量索引负责冷召回 ANN,可变 KV/列负责 $\lambda$ 与元数据。单组件 Qdrant 即可吃下(payload 可变标量 + 过滤 + namespace),不引入 FAISS+sidecar。
>
> **硬性条款 · 衰减是惰性求值,不是定时任务**:不得建立任何 daemon 周期性 tick 全池 $\lambda$。只存 `lambda_plus`(上次事件后的 $\lambda^{+}$)与 `t_last_event`,在**读取时**才求值 $\lambda(t)=\lambda^{+}\cdot e^{-\beta\Delta t}$。建衰减守护进程既浪费又引入时钟竞态,实现必须遵此。

## 2. 活跃度动力学(白天,零 token)

召回打分、衰减(按事实类型分档)、激励(正向)、矛盾(负向)均为第一部分确定性计算,白天全程不调用 LLM。

> **硬性条款 C-3 · $\Delta t$ 取数据集时间,不取 wall-clock(见第四部分 §6)**:$e^{-\beta\Delta t}$ 依赖时间流逝。在 benchmark 中没有真实流逝时间——若 $\Delta t$ 用 run 的运行墙钟(数秒跑完十个 session),衰减项恒约为 1,衰减机制在评测里**根本不发生作用**,而 temporal 子类与 §5 类型分档验证会得到无意义结果且无从察觉原因。系统必须经一个可插拔 `Clock` 接口取时间:生产用 wall-clock,评测用数据集 session 时间戳。LongMemEval 的 session 自带时间戳,$\Delta t$ 必须取数据集时间。

## 3. 三种召回源:层级兜底

按成本递进,大多数查询在前两层解决:

1. **天然上下文(MEMORY.md)——常驻,不检索。** dreaming 提炼出的高价值长期记忆挂在 system prompt 顶部。只装幸存者所以体量小,最关键的记忆一直在场,不需"召回"这个动作。
2. **热召回——每轮隐式触发,零 token。** 每个 user turn 用 score 公式取候选,经阈值门控注入。默认路径。
3. **冷召回——条件触发,零 token。** 仅当热召回 top-1 的 score 低于阈值时,绕过 $\lambda$、在全池跑纯语义检索,闭合"刻意找旧信息"的需求。旧信息从未删除,只是被 $\lambda$ 排到后面。

## 4. 激活动作:被采纳才激活,不是被召回就激活

"被召回"(进候选、过阈值、塞进上下文)**不**触发激励——否则语义相似但已过时的片段会每次被召回、每次被强化、永不衰减。只有"被采纳"(实际进入模型最终回答)才触发 $\lambda^{+}$ 激励。塞进上下文的片段集合天然是激活候选集,被采纳的子集即激活集合。

## 4bis. 实时矛盾分诊:闸门触发的二次调用(新增)

这是把负反馈(第一部分 §5)接入日间回路的机制。设计上必须同时满足两个约束:**(a) 不损害主 agent 回答主任务的注意力**;**(b) 不违反"日常零 token"与 §7 无子 agent 原则**。

### 4bis.1 为什么不是其它两种方案

- **纯启发式判定不可行**:embedding 相似度衡量话题相关性而非真值一致性。"我在东京"与"我搬回伦敦了"在向量空间高相似(同主语同属性、仅取值相反),$\cos$ 无法区分"补充"与"推翻"。否定、时间性取代、实体—属性—取值冲突本质都是语义推理,零 token 启发式结构上走不通。
- **旁路独立 LLM 不可取**:每个 user turn 多一次调用,违反 §7 判据(矛盾判定是单步集合标注,非自主多步有状态推理),并亲手破坏"日常零 token"。
- **同生成搭便车有真实代价**:把"回答用户"与"输出结构化 schema"塞进同一次生成,主任务复杂时模型会优先服务用户可见部分而草草或遗漏 schema;且格式遵从随输出长度衰减,最复杂(最该触发矛盾判定)的回合最易丢信号——最坏的失败相关性。

### 4bis.2 方案 C:零 token 预筛 + 稀疏二次调用

**核心洞察**:矛盾是稀疏事件。绝大多数 turn 不存在"新陈述推翻旧记忆"。为稀疏事件给每轮都背 schema 负担或每轮加一次调用,都是为小概率付全额成本。

**流程**:

1. **第一次生成保持纯净**:只回答用户,不带任何 schema,注意力全部在主任务。采纳信号继续用既有启发式(回答文本与片段的 n-gram/语义重叠)获取,这一路本就不需自报,不受影响。
2. **零 token 矛盾预筛**:生成结束后,中间件计算一个免费的反向信号——**召回时高 $\cos$、但未被采纳**的片段。语义上高度相关(高 $\cos$ 被召回)却没进最终回答(未采纳),正是"用户在谈这个主题但模型选择不用这条旧信息"的指纹,即过时/被推翻片段的典型特征。$\cos$ 在召回时已算过,采纳集本就在收,此信号零额外成本。
3. **闸门**:仅当预筛信号越过阈值 $\theta_c$(本轮存在"高相关但被冷落"片段)才触发**一次轻量调用**:把那 1–3 条可疑片段 + 用户 turn 单独喂入,只问"这几条里哪些被这句话推翻了?",返回编号。该调用上下文极小、任务单一、无主任务干扰,schema 遵从率接近 100%。命中编号执行第一部分 §5 的负向更新。

### 4bis.3 为何同时绕开两个顾虑

- **对注意力顾虑**:主回答那次生成纯净、零 schema 负担,"答了问题忘了 schema"在 C 里不存在,因为主生成根本不要求 schema。
- **对零 token / 无子 agent 顾虑**:它不是旁路独立 agent,是主 agent 的第二次调用,且绝大多数 turn 不触发(预筛稀疏)。"日常零 token"在统计意义上保住——不是每轮零 token,是绝大多数轮零 token,只在预筛报警的稀疏 turn 付一次小调用。它依然是单步、无状态(喂候选→返回编号,无跨轮记忆),不构成 §7 中"需要独立 agent"的情形;是中间件触发的一次确定性子程序,不是一个 agent。

### 4bis.4 实时与夜间的分层(不是替代)

- **实时(日间)= 分诊**:浅、局部、即时。只看本 turn 恰好召回的那几条,只判"对着当前这句话"的冲突。作用:别在本次会话反复打脸用户、别让过时片段这一整天反复被注入与被 $\lambda^{+}$ 强化。结构上做不了全局调和(看不到两条从未一起被召回的记忆间的矛盾)。
- **夜间(dreaming,§6.3)= 裁决**:有全量上下文、低频、批量,在 fact level 改写 MEMORY.md,能决定两条都高 $\lambda$ 的冲突谁胜出、能处理"新陈述本身才是错的"。

实时**抑制症状**(降权,使其停止反复浮现),夜间**裁决病因**(定真值、改写权威层)。实时路径**喂**夜间路径:一条白天被矛盾惩罚压过多次的片段,正是 dreaming 应优先调取裁决的高信号候选。两者互补,不冗余、不打架。

$\theta_c$ 是新增超参,与系统已有的 $\tau$、`intermediate_top_k` 同构,是本设计本就接受的旋钮类型,非新范式。

## 5. 注入策略:有效且不污染上下文

**分区放置**:MEMORY.md 放 system prompt 顶部(权威、稳定);热召回片段放当前 turn 正上方,明确标记包裹(检索来的、未必全相关、可不用);冷召回触发时临时注入,用完即走,不进对话历史。

**阈值是抗污染总闸**:低于注入阈值 $\tau$ 的片段一律丢弃。score 公式天然把"语义又不像、又早被遗忘"的片段压到阈值下,这套机制本身就是抗污染过滤器——相对纯 RAG 的固有优势。

**注入量随 score 分布自适应,非固定 k**:最高分断崖领先 → 只塞 1–2 条;分数平缓 → 多塞几条让模型自挑;全部低于 $\tau$ → 一条不塞。"该沉默时能沉默"是区别于 RAG(永远塞 top-k)的能力。

**附带元数据**:每条片段标注时间、距今多久、当前 $\lambda$,把时间感知的一部分交给模型而非全压给机制。

## 6. Dreaming:夜间巩固,LLM 唯一进场点

每天触发一次,全系统唯一消耗 LLM 处,低频、批量、可缓存、成本可预测,且只对高价值区操作:

1. **抽取**:只扫 $\lambda$ 高且被反复采纳的幸存片段,LLM 提炼写入 MEMORY.md。让 $\lambda$ 动力学先做一整天零成本筛选,只有挺过衰减与矛盾惩罚、被反复采纳的记忆才配消耗 token 抽取。
2. **固化**:已写入 MEMORY.md 的片段语义已被天然上下文承载,原片段可沉降($\lambda$ 降低),原文留池作审计,不物理删除。
3. **矛盾裁决**:在 fact level 对 MEMORY.md 做写入/修改/删除。新晋升内容与已有冲突时一次性裁决并改写。低频、批量、有完整上下文,既省实时 token,也降幻觉概率。**实时分诊(§4bis)产出的"被反复矛盾片段"是这里的优先输入。**

## 7. 架构原则:无子 agent(更新)

系统只有一个 agent(主 agent)。记忆是夹在主 agent 与存储之间的**无状态确定性中间件**:拦截 query → 打分排序 → 阈值过滤 → 注入 → 回收采纳与矛盾信号更新 $\lambda$。dreaming 作为离线任务异步挂在旁边。

判据:一个职责需要独立 agent,当且仅当它需要自主、多步、有状态的推理。本系统中召回是单步计算、激活是集合操作、相关性判断已被 score 公式吸收、矛盾消解已离线化、**实时矛盾分诊是闸门触发的单步无状态子程序**——没有一项满足,故不引入子 agent。方案 C 相对"旁路独立 LLM"更忠于此原则:它没有把矛盾判定变成一个常驻的、自主的第二 agent。

## 8. 差异化定位(更新)

相对主流系统:Mem0/Zep 实时抽取记忆(高频 token、有抽取无演化或演化重型),本机制延迟到夜间且只抽幸存者;Generative Agents 有 reflection 但需全量反思且 importance 静态,本机制用 $\lambda$ 先做零成本筛选、importance 由使用动态涌现;Letta 处处用 LLM 管记忆且记忆在 agent loop 内自编辑,本机制日常零 LLM、记忆是 loop 外的确定性中间件、仅 dreaming 一天进场一次。**矛盾处理上**:主流系统要么不处理更新(纯 RAG),要么用实时 LLM 抽取/裁决(Mem0 的 add/update/delete、Zep 的时序图重建),成本高;本机制用零 token 预筛把矛盾判定从"每轮都做"压缩为"稀疏触发",负反馈入回路而日常成本曲线几乎不动。

---

# 第三部分 · 评测方案

记忆系统只有搭在 agent 框架上才能真正生效与评测。本部分给出基准选择、框架选型与可投稿的实验设计。

## 1. 评测目标与基准选择

本机制的核心创新与核心风险都集中在"知识更新"上(§4bis 实时矛盾分诊、§5 负向更新)。基准必须能正面检验这一点。

- **LoCoMo**(1540 题,单跳/多跳/开放域/时序)——入门 sanity check 与宽对比。几乎测不到知识更新能力,单独看说明不了核心创新。
- **LongMemEval**(500 题,六类:单会话用户回忆、单会话助手回忆、偏好回忆、知识更新、时序推理、多会话综合)——**主战场**。S 变体每题仅 1–3 个相关 session 埋在 45+ 干扰 session 中,大海捞针设置,正面检验冷召回兜底 + $\tau$ 抗污染;且独有 abstention 类,正对本机制"该沉默时能沉默"相对纯 RAG 的固有优势。

**策略**:LoCoMo 宽对比 + sanity;LongMemEval 主战场,**按六子类分别报分**,重点突出 knowledge update / temporal / abstention 三类。

## 2. Agent 框架选型

### 2.1 硬接口要求

剥到最里层,本系统对宿主框架只有一个硬要求:**在主 agent 那次 LLM 调用之前能确定性注入召回片段(前钩 = RecallMiddleware),之后能确定性拿回采纳/矛盾信号(后钩 = adoption + contradiction 更新)。** dreaming 是 session 边界外挂,不算 loop 内要求。

一个框架能否承载本系统,不取决于它多先进、star 多少,只取决于它**把这两个钩子暴露成一等公民,还是埋进自己的 loop 抽象里**。

### 2.2 四范式坍缩

2026 年生产模式稳定为四种:图式(LangGraph 等)、角色式(CrewAI/Agno)、交接式(OpenAI Agents SDK)、层级式(Google ADK)。以硬接口要求与 §7 原则为尺:

- **角色式 / 层级式 / 会话式(CrewAI / Agno / Google ADK / AutoGen/AG2)——范式级冲突,排除**:内核是多 agent 协作,与 §7"系统只有一个 agent"对立;且 CrewAI、Mastra、Google ADK 自带语义记忆,会与待评测的本机制竞争并需先拆除、还要证明无残留污染。
- **交接式(OpenAI Agents SDK)——loop 不透明**:handoff 抽象与单 agent 记忆实验无关,且无长流程 checkpointing、错误处理粒度粗,难以干净楔入前后确定性钩。
- **Letta——最像答案的错误答案**:其哲学是 agent 在 loop 内用工具自主管理自己的记忆,与 §7"召回单步、矛盾离线化、不需自主推理所以不引入子 agent"是精确反面。用它等于论文未写先自相矛盾。
- **图式(LangGraph)——唯一同构选择**。

### 2.3 LangGraph 三点结构契合

| 设计要求 | LangGraph 原语 |
|---|---|
| 记忆是夹在 agent 与存储间的无状态确定性中间件 | 有向图 + 条件边,节点即纯函数;`recall → llm → update` 逐节点确定性 |
| 日常零 token,框架不得塞自己的 LLM 记忆机制 | 只有 checkpointing,无语义记忆——对"记忆"零意见,不竞争、不污染 |
| 评测需可复现、可插桩、可逐机制消融 | 持久化 checkpointing(确定性重放)、每节点可单独计 token、换节点即做 ablation |

叠加:LangGraph 是受监管行业有状态生产工作流的默认选择、2026 年初 star 超过 CrewAI、Mem0 benchmark harness 原生支持其接线——"先进/主流"与"评测管道已铺好"两条同时满足。三个诉求(兼容、主流、理念接近)在 LangGraph 上是同一个答案。

### 2.4 PydanticAI 作补强(不作壳)

本系统唯一脆弱的 LLM 交互是 §4bis.2 步骤 3 那次轻量调用的结构化自报(`{contradicted:[ids]}`,以及若采用同生成变体时的 `{adopted, contradicted}`)。PydanticAI 通过显式 schema 校验提供类型安全的结构化输出。正确组合是 **LangGraph 当编排骨架 + 在调用节点内用 PydanticAI 锁住自报契约**,而非二选一。

## 3. 系统架构:5 节点图

```
START → recall_node → main_llm_node ─┬─(预筛报警)→ contradiction_micro_node → adoption_update_node → END
                                     └─(未报警)──────────────────────────→ adoption_update_node → END

dreaming_pass:不在图内,挂 session 边界离线触发
```

模块定义:

- **recall_node**:RecallMiddleware,确定性零 token。实现第一部分 score / μ-熵 / $\tau$ / 自适应 k / 三层兜底。LangGraph 节点本质是纯函数,完美承载"无状态中间件"。
- **main_llm_node**:唯一调 LLM 的节点。第一次生成纯净;内部用 PydanticAI 约束输出契约。
- **contradiction_micro_node**:条件边触发(零 token 预筛信号 > $\theta_c$ 才进)。喂 1–3 条可疑片段 + user turn,返回被推翻编号,执行第一部分 §5 负向更新 $\lambda^{+}=\lambda^{-}(1-s_c)$;未报警则条件边直接跳过。
- **adoption_update_node**:确定性回写 §3 正向激励 $\lambda^{+}$。
- **MemoryStore**(节点共享后端):turn 粒度池,每条存 `{id, embedding, λ, type_class, t_created, t_last_called}`,type_class 决定 $\beta$ 分档。Qdrant/FAISS,薄实现。
- **dreaming_pass**:session 边界 hook,系统唯一 LLM-heavy 进场点,做 §6 抽取/固化/裁决。

LangGraph 在此只用到约 5% 能力(管道 + 一条条件边 + checkpointing),但这 5% 恰是评测刚需:逐节点 token 计量、确定性重放、换节点 ablation。刻意让框架层薄——所有"智能"留在记忆模块与那一次离线 dreaming 里。

## 4. Dreaming 在 benchmark 的触发契约

LongMemEval 是"先按序喂 N 个 session,最后问一问题",无"天"的概念。自然且可复现的映射:**每个 session 结束、下一个 session 开始前跑一次 dreaming = 一个 dreaming 周期**。此映射须在论文写死并说明理由,否则审稿人会质疑 dreaming 触发频率是调出来的。该映射与"低频批量"的设计意图一致。

## 5. 类别—机制 ablation 矩阵

| LongMemEval 子类 | 主要检验机制 | 对应 ablation |
|---|---|---|
| knowledge update | 实时矛盾分诊 + 负向更新 + 夜间裁决 | 关 contradiction_micro / 关 dreaming 裁决 |
| temporal reasoning | $\beta$ 按事实类型分档衰减 | 单一 $\beta$(不分档) |
| abstention | $\tau$ 阈值 + 自适应 k | 固定 top-k 不设 $\tau$(退化纯 RAG 注入) |
| multi-session | 冷召回兜底 + MEMORY.md 常驻 | 关冷召回 / 关 MEMORY.md |
| single-session recall | $\lambda$ 使用动力学本体 | 关 $\lambda$,score 退化为纯 $\cos$ |

每行是"一个机制对一个子类的因果验证"。full system vs 逐个消融,跑 LoCoMo + LongMemEval,对比 Mem0 / Zep / full-context 三条基线(经 Mem0 harness 的 adapter 接口免费获得)。产出不是"我们分数高",而是"每个机制各自贡献了哪一类多少分"——支撑第一部分公式的实验证据。

## 6. 必报指标

除准确率外,**必须报每 query 平均 LLM 调用数与 token 数**,与 Mem0/Zep 直接对比。本机制命脉是"日常零 token":日常路径调用数≈0,成本只在 dreaming 与稀疏矛盾触发时发生。准确率 / token 这条曲线是相对 §8 对手最有说服力的图,不可只报准确率把它埋掉。

## 7. 落地顺序

1. LangGraph 接出最小 5 节点图 + MemoryStore + recall_node,LoCoMo 跑通拿第一个数(sanity)。
2. 接 Mem0 harness adapter 接口,拿 Mem0/Zep/full-context 三条基线。
3. 加 main_llm_node 的 PydanticAI 自报契约 + adoption_update_node。
4. 加 contradiction_micro_node(方案 C 预筛 + 二次调用 + 负向更新),LongMemEval 上跑 knowledge update 子类。
5. 逐个加 dreaming 三职能,每加一个在 LongMemEval 上跑一次对应 ablation。

每步都有可对比的数,不会到最后才发现某机制是负贡献。

---

# 第四部分 · 工程落地与接入

本部分把回路周边的接入缺口逐一对到可复用现成件,并给出可对照实现的字段与契约。**核心判断**:缺的不是框架,agent loop(LangGraph 5 节点 + PydanticAI)已定且很小;缺的是约 8 个集成决策,其中 C-1/C-2/C-3 三条不满足时评测结果会错且不可察觉,列为硬性条款。不可复用的部分恰好只有记忆算法本身(λ 动力学、μ-熵仲裁、负向更新、方案 C 闸门)——那是研究贡献,定义上不该有现成件;其余全是 plumbing,装配量约数百行胶水。

## 1. 三个 LLM 调用点是三个正交配置,不是一个模型

`main_llm`、`contradiction_micro`、`dreaming` 在 cost / latency / capability 上正交:

| 调用点 | 延迟敏感 | 上下文 | 频率 | 建议模型档 |
|---|---|---|---|---|
| main_llm(主回答) | 高 | 大(对话 + 注入) | 每轮 | 强模型 |
| contradiction_micro(矛盾微调用) | 低 | 极小(1–3 片段 + turn) | 稀疏 | 小/便宜 + 结构化 |
| dreaming(夜间巩固) | 无 | 批量 | 每 session 边界一次 | 最强,可缓存 |

故需一个**模型路由层**,按 call-site 分配不同模型且换 provider 不动 graph 代码。

> **硬性条款 C-1 · backbone 钉死**:三方对比(本机制 / Mem0 / Zep)与所有 ablation 必须同一 backbone(如统一 gpt-4o-mini)。模型散落各节点硬编码会使对比表第一轮 review 被打回。LongMemEval 成绩按 backbone 分别报,backbone 是控制变量不是可调项。

## 2. LLM 接入端口

复用件:**LiteLLM**。统一接口、100+ provider、按 call-site 配置不同模型、内置 per-call token/cost callback(直接喂第三部分 §6 的成本曲线)。这是 C-1 的落地载体——一处配置集中管理三个调用点的模型映射。

## 3. Embedding 端口(独立一根线)

score 的 cos、冷召回 ANN、矛盾 $s_c$ 全由 embedding 驱动。

> **硬性条款 C-2 · embedding 整 run 冻结**:一次完整 run 内不得换 embedding 模型或维度。中途重嵌则池内全部 λ 加权 cos 失去可比性,已跑 session 数据全废。

复用件:本地 **sentence-transformers**,版本钉死(API embedding 会被 provider 静默升级,破坏复现)。此线**不**经 LiteLLM 代理,避免版本随 provider 失控。

## 4. 存储:混合形态 + 字段定义

复用件:**Qdrant**(单组件:向量索引做冷召回 ANN + payload 存可变 λ 与元数据 + namespace 隔离 + 标量过滤)。每条记录字段:

```
id            : str            # 片段唯一 id
embedding     : vector         # 冻结模型产出(C-2)
text          : str            # 原始 turn/session 片段,永不拆解
lambda_plus   : float (0,1]    # 上次事件后的 λ⁺(惰性衰减基)
t_last_event  : dataset_time   # 上次事件的数据集时间(配合 Clock,C-3)
t_created     : dataset_time   # 创建时间(类型分档衰减用)
type_class    : enum           # volatile|stable|identity → 决定 β
namespace     : str            # user_id / question_id 隔离(评测并发用)
```

读取时求值 $\lambda=\lambda^{+}\cdot e^{-\beta\,\Delta t}$,$\Delta t$ 经 `Clock` 取数据集时间(C-3)。禁止衰减 daemon(见第二部分 §1 硬性条款)。

## 5. 节点间通信:LangGraph State schema

State 在节点间传递,必须显式携带"召回那一刻"的快照,**下游不得重算**(届时 λ 已变):

```
messages            : list                 # 当前对话
retrieved_segments  : list[{id, cos_at_recall, lambda_minus_snapshot, text}]
adopted_ids         : list[str]            # adoption 启发式产出
contradicted_ids    : list[str]            # contradiction_micro 产出(或空)
prescreen_signal    : float                # 零 token 矛盾预筛分(高 cos 未采纳)
session_id          : str
namespace           : str
```

- 采纳判定(方案 C 的启发式路):最终回答与每条注入片段的**归一化 token-set 重叠或 embedding 相似度**,$\ge\theta_a$ 即采纳;$\theta_a$ 入附录超参表。
- 矛盾预筛信号:$\text{prescreen}=\max\{\text{cos\_at\_recall}_j \mid j\notin \text{adopted\_ids}\}$,即"召回时最相关却未被采纳"的那条的 cos;$>\theta_c$ 则触发 contradiction_micro 节点。两个量都只用 State 里随召回带下来的 `cos_at_recall` 与 `lambda_minus_snapshot`。

## 6. 可插拔时钟(C-3 的落地)

一个约 3 行的 `Clock` 接口,无需框架:

```
class Clock:                       # 生产实现:返回 wall-clock
    def now(self) -> Time: ...
class DatasetClock(Clock):         # 评测实现:返回当前 session 数据集时间戳
    def now(self) -> Time: ...     # 由 benchmark 驱动器在 session 切换时推进
```

所有 $\Delta t$ 计算只经 `clock.now()`,严禁直接调系统时间。LongMemEval 驱动器在喂下一 session 前推进 `DatasetClock`。

## 7. Dreaming 工程形态与 MEMORY.md 实体

- 评测期:dreaming 就是 session 之间的一次普通函数调用,**不引入 Celery/调度框架**。生产期才需轻量调度(**APScheduler** 足够)+ 一把"无 turn 在飞时才写"的锁。
- **MEMORY.md 不是文件**:它是每 namespace 一段带版本的文本 blob,实现成存储里一个 versioned text 字段,每轮注入 system prompt 顶部。"MEMORY.md"是概念名,严禁操作真实文件系统(评测并发多 namespace 时文件会打架)。

## 8. 可观测性(headline 图的唯一来源)

每个 LLM 调用按 **(call-site, query-id)** 打点 token/latency;每轮记录 injected/adopted/contradicted 的 id、冷召回是否触发、预筛是否跳闸。复用件:LiteLLM cost callback + **Langfuse**(或纯 OpenTelemetry)做 trace。无此层则第三部分 §6 的"准确率 / token"曲线画不出来。

## 9. 失败处理 = 分层设计的容错叙事

contradiction_micro 超时 / 拒答 / 输出坏(即便有 PydanticAI)→ **该轮跳过负向更新**。这不是将就:漏掉的矛盾自然下沉到夜间 dreaming 全局裁决——§4bis.4"实时分诊、夜间裁决"的分层**本身就是 graceful degradation 论证**。投稿时应明确点出此点。

## 10. 可复用栈总表

| 缺口 | 复用件 | 性质 |
|---|---|---|
| 编排骨架 | LangGraph | 已定 |
| 结构化自报契约 | PydanticAI | 已定 |
| LLM 接入 / 多 provider 路由 / token 计费 | **LiteLLM** | 评测刚需(C-1 载体) |
| Embedding 端口 | sentence-transformers(钉版本,本地) | 评测刚需(C-2) |
| 向量检索 + 可变 λ payload + namespace | **Qdrant**(单组件) | 刚需 |
| 时间抽象 | 自写 `Clock` 接口(~3 行) | 评测刚需(C-3) |
| 基线对比 | Mem0 开源 benchmark harness | 已定(三条基线免费) |
| Token/trace 观测 | Langfuse 或 OpenTelemetry | 评测刚需(headline 图) |
| dreaming 调度 | 评测期:函数调用;生产期:APScheduler | 生产期才需 |
| 记忆算法本体 | **无现成件——研究贡献本身** | 自实现 |

---

## 附录 · 超参数总表

| 符号 | 含义 | 默认 / 说明 |
|---|---|---|
| $\beta$ | 衰减系数 | 按事实类型分档(易变大 / 稳定小 / 身份→0),需调 |
| $\mu_{base}$ | 噪音系数基线 | 0.1,可调 |
| $k$ | μ 曲率 | 1/2,可调 |
| $\tau$ | 注入阈值(抗污染总闸) | 需调 |
| `intermediate_top_k` | 每轮取前 k 条视为成功调用 | 需调,过小激励过集中、过大无差别激活 |
| $\theta_c$ | 矛盾预筛触发阈值(新增) | 需调,与 $\tau$、`intermediate_top_k` 同构 |
| $\theta_a$ | 采纳判定重叠阈值(新增) | 回答与片段归一化 token-set / embedding 重叠 ≥ $\theta_a$ 即判采纳;eval 敏感旋钮,需调 |

**评测控制变量(整 run 内冻结,不参与调参)**:backbone LLM(三方对比与 ablation 须同一型号,C-1);embedding 模型 + 维度(中途换则历史 λ 加权 cos 全失效,C-2);`Clock` 取数据集时间(C-3)。这三项不是超参,是评测正确性前提,违反则结果无效且不可察觉。

**敏感性分析计划(投稿用)**:需扫描——$\beta$ 分档值、$\tau$、`intermediate_top_k`、$\theta_c$;固定默认即可——$\mu_{base}$、$k$、$\theta_a$(报一次取值与一次 ±0.05 鲁棒性即可,不必全扫)。

---

*文档版本 v3 · 在 v2(实时矛盾分诊方案 C、负向矛盾更新公式、LangGraph + PydanticAI 评测选型)基础上新增第四部分·工程落地与接入:可复用栈映射、State schema、混合存储字段、可插拔时钟,及三条评测刚需硬性条款(C-1 backbone 钉死 / C-2 embedding 冻结 / C-3 数据集时间时钟)并回填至第一、二部分。可作为正式设计文档或论文初稿骨架,工程上可对照直接落地。*

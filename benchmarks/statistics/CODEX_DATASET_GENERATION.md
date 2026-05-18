# Codex 离线生成 Turn-Level Statistics 数据集手册

本文件给 Codex 使用，不调用 DeepSeek，不调用任何外部模型 API。后续分批生成数据集时，按这里的流程直接在仓库中创建/修改 JSON 文件，并在本文末尾更新生成登记。

当前 validator: `benchmarks/statistics/validate_update_dataset.py`

最终输出目录:

```text
benchmarks/statistics/A/
benchmarks/statistics/B/
benchmarks/statistics/C/
benchmarks/statistics/D/
benchmarks/statistics/E/
```

## 1. 当前任务

生成一批自然中文长期对话 scenario，用于 turn-level temporal memory retrieval 评测。每个 scenario 是一个 JSON 文件，包含 42-64 个连续 turn，以及至少 1 个 eval。这里的 **turn 是完整对话回合**：一来一回才算 1 个 turn，不是一条单句、单条消息或单个 speaker 发言。检索单位是 turn，不是 fact。

核心目标:

- 对话要像真实生活/工作聊天，不像测试题。
- eval 标注要能明确区分应该召回的 turn 和不该排高的干扰 turn。
- 每批生成后必须跑 validator。
- 每批生成后必须更新本文的“生成登记”和“已完成清单”。

## 2. 不要做的事

- 不要调用 `generate_with_deepseek.py`。
- 不要调用外部 LLM API。
- 不要在最终 JSON 中加入旧 schema 字段: `facts_added`, `facts_updated`, `fact_id`, `expected_answer`, `expected_fact_ids`, `forbidden_fact_ids`, `tests` 等。
- 不要在自然对话 text 中出现实验术语: `Hawkes`, `recency`, `召回`, `评测`, `λ`, `cos 相似度`。
- 不要把每个关键信息写成“旧信息/新信息/当前有效/已过期”的说明句。
- 不要让 query 前 3-5 轮直接复述答案，否则样本会退化成纯近因测试。

## 3. 最终 JSON Schema

顶层字段只允许:

```json
{
  "scenario_id": "string",
  "category": "update_override | decay_forget | reactivation | semantic_distractor | stability_check",
  "description": "string",
  "persona": "string or object",
  "turns": [],
  "evals": []
}
```

turn 字段:

```json
{
  "idx": 0,
  "t": "2025-03-03T09:20:00",
  "messages": [
    {"speaker": "user1", "text": "自然中文对话的发起句"},
    {"speaker": "user2", "text": "自然中文对话的回应句"}
  ],
  "tags": ["optional:short_tag"]
}
```

一个 turn 必须正好包含两条 `messages`，两个说话人各出现一次。第一条是本回合的“来”，第二条是本回合的“回”。同一条 message 里可以包含一两句自然语言，但不要把一句话拆成一个 turn，也不要把 user1 和 user2 的两条发言拆成两个 turn。

eval 字段:

```json
{
  "query_turn": 48,
  "type": "update_override",
  "positive_turns": [20],
  "negative_turns": [7],
  "pairs": [
    {"positive": 20, "negative": 7}
  ],
  "top_k": [1, 3, 5]
}
```

硬性规则:

- `turns` 长度必须是 42-64，含义是 42-64 个“一来一回”的对话回合，约等于 84-128 条单独发言。
- `idx` 从 0 开始连续递增。
- `t` 必须是 ISO timestamp，且严格递增。
- 每个 turn 必须有 `messages`，且正好两条。
- 每条 message 的 `speaker` 只能是 `user1` 或 `user2`。
- 同一个 turn 内两条 message 的 `speaker` 必须不同，即 user1/user2 各一次。
- 旧格式的 turn 顶层 `speaker` / `text` 不再允许，因为它表示单条消息，不表示一来一回。
- `query_turn` 必须在 turn 范围内，且大于 0。
- `positive_turns`, `negative_turns`, `pairs` 引用的 turn 必须早于 `query_turn`。
- `positive_turns` 不能为空。
- `update_override` 和 `semantic_distractor` 必须有 `negative_turns`。
- `positive_turns` 和 `negative_turns` 不能重叠。

## 4. 五类样本定义

| 类别 | 目录/前缀 | 目标 | positive_turns | negative_turns |
|---|---|---|---|---|
| A `update_override` | `A/override_A_*` | 当前事实压过旧事实 | 能独立回答 query 的更新后/当前信息 | 曾经有效、会导向旧答案的被覆盖信息 |
| B `decay_forget` | `B/decay_B_*` | 稳定偏好压过一次性兴趣 | 稳定偏好/当前画像 | 一次性、短期、已不应主导的信息 |
| C `reactivation` | `C/react_C_*` | 主题锚点重新唤起沉睡旧记忆 | 早期低频但相关的细节 | 可为空；或语义相近错误项 |
| D `semantic_distractor` | `D/distractor_D_*` | 当前目标压过相似干扰目标 | 能独立回答 query 的当前正确实体/目标 | 旧目标、取消目标、相似错误实体 |
| E `stability_check` | `E/stable_E_*` | 稳定事实低频出现仍可召回 | 稳定事实所在 turn | 可为空；建议加入真实相似干扰项 |

标注理念补充:

- `positive_turns` 不是“有帮助的上下文”，而是检索到这个 turn 后应能直接回答 query。只说“按那条短信”“新主体资料”“放回原位”“那个袋子”的回合，若脱离前文不能给出答案，通常不要标为 positive。
- `negative_turns` 不是“出现了相同关键词”，而是检索到这个 turn 后可能给出错误答案。已经明确否定旧事实、纠正错误目标、支持当前目标的回合，通常不要标为 negative。
- `pairs` 用于表达强排序约束，应优先写“完整 positive > 真正会误导的 negative”。弱 positive、弱 negative 可以留在对话里当自然干扰，但不要随手放进 pair。
- A 类的主 negative 应优先选“旧事实曾经被当作有效安排”的 turn，而不是“旧提醒弹出但当场被否定”“幸好没按旧信息走”的 turn。
- D 类的主 positive 通常只保留完整当前目标 turn；路线、装备、入口、短信置顶、家人追问等辅助 turn 除非能独立回答 query，否则不进 positive。
- E 类允许没有 negative，但为了避免退化成纯正例召回，优先设计 1-3 个相似数字、相似地点、相似偏好、相似流程的真实干扰，并写 pair。

## 5. Codex 生成流程

### 5.1 选题

先看本文“已完成清单”，不要重复 `scenario_id`。新题建议从真实生活/工作里选:

- 医疗、预约、通勤、外卖、快递、课程、会议、报销、设备、账号、家庭安排。
- 避免敏感真实个人信息；号码、地址、账号统一用虚构但自然的内容。
- 每批建议 3-10 个 scenario，生成多了容易质量漂移。

命名规则:

```text
A: override_A_{topic_slug}_{nnn}.json
B: decay_B_{topic_slug}_{nnn}.json
C: react_C_{topic_slug}_{nnn}.json
D: distractor_D_{topic_slug}_{nnn}.json
E: stable_E_{topic_slug}_{nnn}.json
```


### 5.2 先构思自然对话，不先写 eval

每个 scenario 先在心里形成这 4 个元素:

1. persona: 两个对话者的关系和背景。
2. time span: 跨几天、几周或几个月。
3. event pool: 6-10 个可穿插事件。
4. target memory: 最后 eval 会问什么，但不要让对话像围绕测试题编排。

自然对话质量要求:

- 有短句、追问、打断、犹豫、转移话题。
- 每个单句子字数都要大于20，生成10条以上的字数大于50的长句子，生成5条以上字数大于100的长句子。
- 不要连续 5 轮都像完整备忘录。
- 不要每轮都“好的/收到/没问题”式过度配合。
- 关键信息可以出现 1-3 次，但 query 前近邻不能直接复述答案。

### 5.3 写 turns

建议结构:

- 42-64 turns；每个 turn 是一来一回，不是单句。
- 3-6 个自然时间段，每段之间隔天、隔周或隔月。
- 关键事实至少距离 query 5 个 turn 以上。
- 允许 query 不是最后一个 turn，但通常让 query 靠后更好检查。
- `tags` 只用于人工读懂，不参与自然语言。可用 `old:*`, `current:*`, `stable:*`, `distractor:*`, `reactivation_cue:*`。

### 5.4 事后标注 eval

写完 turns 后，再做标注:

1. 找出能直接回答 query 的 turn，放入 `positive_turns`。
2. 找出会导致错误答案的 turn，放入 `negative_turns`。
3. 对 A/D 类，至少写 1 个 `{positive, negative}` pair。
4. 提醒、铺垫、主题触发 turn 不要放进 `positive_turns`，除非它本身能直接回答 query。
5. 标注的是整个一来一回的 turn 编号；只要该回合内任一 message 能直接回答 query，就可以作为 positive。
6. 如果自然对话中没有足够清楚的 positive/negative，不要硬凑，重写该 scenario。

类别细则:

- A `update_override`: positive 应包含当前有效值本身；negative 应包含旧有效值本身。自带纠偏的旧提醒可以作为弱干扰，但不要作为唯一 negative 或主 pair。
- B `decay_forget`: positive 可以是代表性稳定偏好锚点，不一定穷尽全部稳定 turn；核心 pairs 应体现稳定偏好压过临时例外。如果后续评测使用 positive recall，请改为穷尽所有明显稳定偏好 turn。
- C `reactivation`: 近期 cue 只负责唤起主题，不应泄露完整答案；早期 positive 必须包含被唤起的具体细节。辅助确认 turn 不应替代原始细节 turn。
- D `semantic_distractor`: positive 应完整回答 query 中的目标实体、地点、时间、物品、代码等核心字段；negative 应是相似但错误的实体/目标。正确澄清、排除错误、路线对比通常不标 negative。
- E `stability_check`: positive 通常是低频稳定事实第一次或唯一一次出现的 turn；若加入 negative，选真实可能混淆的非答案信息，并确保它不能回答 query。

### 5.5 自检

逐项检查:

- query 是否像真实人在办事时会问的话。
- positive 是否真的能回答 query。
- negative 是否真的会误导答案。
- pair 是否是强 positive 排在强 negative 前，而不是辅助上下文之间的比较。
- 是否把“正确澄清/否定旧信息”的 turn 误标成 negative。
- 是否把“需要前文才能理解”的辅助 turn 误标成 positive。
- query 前 3-5 轮是否泄露答案。
- 每个 turn 是否都是一来一回，而不是把单条发言当成 turn。
- timestamp 是否严格递增。
- 是否有旧 schema 字段。
- 是否出现实验术语。
- 是否和已有 scenario 太像。

### 5.6 验证

对单个文件:

```bash
python3 benchmarks/statistics/validate_update_dataset.py benchmarks/statistics/A/override_A_new_topic_101.json
```

对一批目录:

```bash
python3 benchmarks/statistics/validate_update_dataset.py benchmarks/statistics/A benchmarks/statistics/B benchmarks/statistics/C benchmarks/statistics/D benchmarks/statistics/E
```

如果 validator 报 fatal，必须修。warning 要判断是否会污染评测；不确定时也修。

## 6. 可复制的单条生成模板

创建新 JSON 前，先用这个小纲要约束自己:

```text
scenario_id:
category:
persona:
time_span:
event_pool:
- 
- 
- 
target_query:
direct_answer_turns:
wrong_or_stale_turns:
query_turn:
notes_to_avoid_leakage:
```

然后直接生成最终 JSON 文件。不要保存这个纲要，除非需要在本文的批次登记里说明。

## 7. 批次登记格式

每次生成完一批，追加一行:

```text
YYYY-MM-DD | batch_id | category | scenario_ids | files | validator | notes
```

状态值:

- `planned`: 已计划，未写 JSON。
- `generated`: 已写 JSON，未验证。
- `validated`: validator 通过。
- `needs_fix`: 已写 JSON，但 validator 或人工检查未过。
- `discarded`: 生成后废弃，不进入正式目录。

## 8. 运行提示词

阅读 benchmarks/statistics/CODEX_DATASET_GENERATION.md，并严格按其中 schema、质量要求和 validator 流程生成数据。

任务：派5个subagent，每个subagent生成一类问题，共A、B、C、D、E四类问题，每类共 5 个scenario，编号从 xxx 开始。

类别定义：

类别名：A update_override
目标目录：benchmarks/statistics/A
文件名规则：override_A_{topic_slug}_{nnn}.json
category：update_override
目标：当前事实压过旧事实。
positive_turns：更新后/当前信息所在 turn。
negative_turns：旧信息/被覆盖信息所在 turn。
要求：必须有 negative_turns；必须有至少 1 个 pair。类别名：B decay_forget
目标目录：benchmarks/statistics/B
文件名规则：decay_B_{topic_slug}_{nnn}.json
category：decay_forget
目标：稳定偏好压过一次性兴趣。
positive_turns：稳定偏好、长期画像、持续有效习惯所在 turn。
negative_turns：一次性、短期、临时兴趣或特殊情境信息所在 turn。
要求：query 应该自然询问长期偏好或默认选择，而不是问某次临时事件。

类别名：C reactivation
目标目录：benchmarks/statistics/C
文件名规则：react_C_{topic_slug}_{nnn}.json
category：reactivation
目标：近期主题锚点重新唤起早期低频旧记忆。
positive_turns：早期低频但能直接回答 query 的细节 turn。
negative_turns：可为空；如果有，应是语义相近但错误的细节。
要求：近期 turns 可以重新提到相关主题，但不能直接复述答案；query 需要依赖早期细节。

类别名：D semantic_distractor
目标目录：benchmarks/statistics/D
文件名规则：distractor_D_{topic_slug}_{nnn}.json
category：semantic_distractor
目标：当前正确实体/目标压过相似干扰目标。
positive_turns：当前正确实体、目标、对象或地点所在 turn。
negative_turns：旧目标、取消目标、相似错误实体或相似但不该选的对象所在 turn。
要求：必须有 negative_turns；必须有至少 1 个 pair；干扰项要真的语义相近，不能太明显无关。

类别名：E stability_check
目标目录：benchmarks/statistics/E
文件名规则：stable_E_{topic_slug}_{nnn}.json
category：stability_check
目标：稳定事实低频出现但仍应被记住。
positive_turns：稳定事实所在 turn。
negative_turns：可为空；如果有，只能是非答案干扰项，不能制造覆盖关系。
要求：稳定事实不需要反复出现；query 应自然询问这个长期事实、偏好、身份、固定安排或常用信息。

特别注意 turn 定义：
1. 一个 turn 是完整对话回合，不是一条 sentence、单条消息或单个 speaker 发言。
2. 每个 turn 必须正好包含 2 条 messages。
3. 同一个 turn 内 user1/user2 各出现一次。
4. 第一条 message 是本回合的“来”，第二条 message 是本回合的“回”。
5. 不要把一个句子拆成一个 turn；也不要把 user1 和 user2 的发言拆成两个 turn。
6. 对话总共有 42-64 个 turn，约等于 84-128 条单独发言。

关于 sentence / message 质量：
1. 注意文档中对字数的要求，尽量生成多点长且自然的句子，凸显真实性
2. 不要为了凑字数把每个 turn 都写成长段说明。
3. 不要让每个 turn 都是“提问-确认”模板。

query 要求：
1. query 必须放在最后一个 turn。
2. 每个 scenario 的 query_turn 必须等于最后一个 turn 的 idx。
3. 最后一个 turn 的自然对话内容就是 eval 要查询的问题。
4. query 前 3-5 个 turn 不能直接复述答案。
5. 不同 scenario 的 query 句式必须明显不同，不要都写成“所以最后到底是哪个”。

生成要求：
1. 必须逐个 scenario 独立构思，不要用同一个对话骨架批量替换主题。
2. 每个 scenario 先独立设计 persona、time_span、event_pool、target memory，再写完整自然对话，最后事后标注 eval。
3. turn 数必须自然变化，在 42-64 之间，不要固定相同长度。
4. 由于 query_turn 固定为最后一轮，positive_turns、negative_turns、pairs 的位置必须自然变化，不要固定同一组 idx。
5. 不同 scenario 的主题、人物关系、开头、结尾、闲聊句、确认句、query 句式都要明显不同。
6. 对话要像真实中文长期生活/工作聊天，有跳话题、犹豫、补充、短句、长句和无关事件穿插。
7. 每个 scenario 至少穿插 6-10 个自然事件，不要整段都围绕目标答案。
8. 当前正确实体和相似干扰实体都必须自然出现，且干扰项要有迷惑性。
9. positive_turns 标注当前正确实体所在 turn；negative_turns 标注相似但错误、旧的、取消的或不该选的实体所在 turn。
10. 铺垫、提醒、主题触发 turn 不要误标为 positive，除非该 turn 本身能直接回答 query。
11. 不要在自然对话 text 中出现实验术语：Hawkes、recency、召回、评测、λ、cos 相似度。
12. 不要出现旧 schema 字段。
13. 可以用脚本检查、重命名、验证 JSON，但不要用脚本把同一套对话模板批量填充成多个文件。

开始前：
1. 检查目标目录已有文件。
2. 避免复用已有 scenario_id、主题、人物关系、开头句式、query 句式和 eval 位置分布。
3. 如果已有编号 000-004 已存在，就从下一个可用编号开始；如果目录为空，再从 000 开始。

生成后：
1. 运行：
   python3 benchmarks/statistics/validate_update_dataset.py
2. fatal 和warning必须修复。
4. 人工复核每个 scenario：
   - turn 是否真是一来一回；
   - query_turn 是否为最后一个 turn；
   - positive_turns 是否能直接回答 query；
   - negative_turns 是否真会误导；
   - query 前 3-5 轮是否泄露答案；
   - 是否存在模板化句式或过度重复确认。
5. 更新 benchmarks/statistics/CODEX_DATASET_GENERATION.md 的生成登记。

最后汇报：
- 每个文件名；
- turn 数；
- query_turn；
- positive_turns；
- negative_turns；
- pairs；
- validator 结果。

## 8. 生成登记

| 日期 | batch_id | 类别 | scenario_ids | 状态 | 说明 |
|---|---|---|---|---|---|
| 2026-05-16 | A_000_004 | update_override | override_A_clinic_slot_000; override_A_invoice_title_001; override_A_pickup_gate_002; override_A_countertop_color_003; override_A_classroom_room_004 | validated | 生成 A 类 5 个场景，文件位于 `benchmarks/statistics/A/`；A 目录 validator 通过 0 fatal / 0 warning。 |
| 2026-05-16 | B_000_004 | decay_forget | decay_B_weekend_breakfast_000; decay_B_focus_music_001; decay_B_commute_default_002; decay_B_tea_order_003; decay_B_reading_format_004 | validated | 生成 B 类 5 个场景，文件位于 `benchmarks/statistics/B/`；全量 validator 通过 0 fatal / 0 warning，并人工复核 query_turn、positive/negative 标注与 query 前泄露。 |
| 2026-05-16 | C_000_004 | reactivation | react_C_bakery_invoice_000; react_C_luggage_key_001; react_C_ceramics_glaze_002; react_C_clinic_parking_003; react_C_projector_adapter_004 | validated | 生成 C 类 5 个场景，文件位于 `benchmarks/statistics/C/`；全量 validator 通过 0 fatal / 0 warning，并人工复核早期细节、近期主题锚点与 query 前泄露。 |
| 2026-05-16 | D_000_004 | semantic_distractor | distractor_D_pharmacy_pickup_000; distractor_D_knee_class_001; distractor_D_client_meeting_room_002; distractor_D_parcel_pickup_003; distractor_D_denture_appointment_004 | validated | 生成 D 类 5 个场景，文件位于 `benchmarks/statistics/D/`；全量 validator 通过 0 fatal / 0 warning，并人工复核当前正确实体、相似干扰项和 pairs。 |
| 2026-05-16 | E_000_004 | stability_check | stable_E_locker_combo_000; stable_E_train_seat_001; stable_E_pickup_point_002; stable_E_fiddle_leaf_003; stable_E_tv_pin_004 | validated | 生成 E 类 5 个场景，文件位于 `benchmarks/statistics/E/`；全量 validator 通过 0 fatal / 0 warning，并人工复核稳定事实低频出现且 query 前未泄露。 |
| 2026-05-16 | A_005_009 | update_override | override_A_wifi_password_005; override_A_train_seat_006; override_A_workshop_material_007; override_A_storage_code_008; override_A_wedding_song_009 | validated | 生成 A 类 5 个场景，文件位于 `benchmarks/statistics/A/`；全量 validator 通过 0 fatal / 0 warning，并人工复核当前事实覆盖旧事实、query_turn 与 query 前泄露。 |
| 2026-05-16 | B_005_009 | decay_forget | decay_B_evening_jog_005; decay_B_visible_storage_006; decay_B_meeting_notes_007; decay_B_fragrance_free_008; decay_B_quiet_nap_009 | validated | 生成 B 类 5 个场景，文件位于 `benchmarks/statistics/B/`；全量 validator 通过 0 fatal / 0 warning，并人工复核长期偏好与一次性兴趣的 positive/negative 标注。 |
| 2026-05-16 | C_005_009 | reactivation | react_C_chorus_stand_005; react_C_photo_drive_006; react_C_yoga_spray_007; react_C_garden_pump_008; react_C_podcast_mic_009 | validated | 生成 C 类 5 个场景，文件位于 `benchmarks/statistics/C/`；全量 validator 通过 0 fatal / 0 warning，并人工复核早期低频细节、近期主题锚点和 query 前未复述答案。 |
| 2026-05-16 | D_005_009 | semantic_distractor | distractor_D_gallery_frame_005; distractor_D_rehearsal_room_006; distractor_D_parent_camp_bus_007; distractor_D_laptop_charger_model_008; distractor_D_archival_box_shelf_009 | validated | 生成 D 类 5 个场景，文件位于 `benchmarks/statistics/D/`；全量 validator 通过 0 fatal / 0 warning，并人工复核当前正确实体、相似干扰实体和强排序 pairs。 |
| 2026-05-16 | E_005_009 | stability_check | stable_E_clinic_allergy_005; stable_E_invoice_title_006; stable_E_choir_position_007; stable_E_regular_coffee_008; stable_E_grandpa_followup_009 | validated | 生成 E 类 5 个场景，文件位于 `benchmarks/statistics/E/`；全量 validator 通过 0 fatal / 0 warning，并人工复核稳定事实低频出现、非覆盖干扰项和 query 前未泄露。 |
| 2026-05-17 | A_010_014 | update_override | override_A_flower_address_010; override_A_meeting_platform_011; override_A_camera_return_012; override_A_volunteer_meetup_013; override_A_assignment_email_014 | validated | 生成 A 类 5 个场景，文件位于 `benchmarks/statistics/A/`；全量 validator 通过 0 fatal / 0 warning，并人工复核当前事实覆盖旧事实、强排序 pairs、query_turn 为最后一轮和 query 前未泄露。 |
| 2026-05-17 | B_010_014 | decay_forget | decay_B_desk_lighting_010; decay_B_morning_brief_011; decay_B_market_bag_012; decay_B_handwritten_ideation_013; decay_B_photo_archive_014 | validated | 生成 B 类 5 个场景，文件位于 `benchmarks/statistics/B/`；全量 validator 通过 0 fatal / 0 warning，并人工复核长期偏好压过临时兴趣、query 自然询问默认选择且 query 前未泄露。 |
| 2026-05-17 | C_010_014 | reactivation | react_C_archive_reader_010; react_C_theatre_sash_011; react_C_field_quadrat_012; react_C_stamp_ink_013; react_C_apartment_valve_014 | validated | 生成 C 类 5 个场景，文件位于 `benchmarks/statistics/C/`；全量 validator 通过 0 fatal / 0 warning，并人工复核早期低频细节、近期主题锚点和 query 前未复述答案。 |
| 2026-05-17 | D_010_014 | semantic_distractor | distractor_D_scan_appointment_010; distractor_D_badge_print_011; distractor_D_apartment_viewing_012; distractor_D_lens_rental_013; distractor_D_service_window_014 | validated | 生成 D 类 5 个场景，文件位于 `benchmarks/statistics/D/`；全量 validator 通过 0 fatal / 0 warning，并人工复核当前正确目标、语义相近干扰项、强排序 pairs 和 query 前未泄露。 |
| 2026-05-17 | E_010_014 | stability_check | stable_E_label_tape_010; stable_E_racket_string_011; stable_E_audio_filename_012; stable_E_studio_access_013; stable_E_filter_cartridge_014 | validated | 生成 E 类 5 个场景，文件位于 `benchmarks/statistics/E/`；全量 validator 通过 0 fatal / 0 warning，并人工复核稳定事实低频出现、非覆盖干扰项和 query 前未泄露。 |
| 2026-05-17 | A_015_019 | update_override | override_A_poster_size_015; override_A_cake_flavor_016; override_A_parking_lot_017; override_A_bank_branch_018; override_A_freezer_shelf_019 | validated | 生成 A 类 5 个场景，文件位于 `benchmarks/statistics/A/`；全量 validator 通过 0 fatal / 0 warning，并人工复核当前事实覆盖旧事实、强排序 pairs、query_turn 为最后一轮和 query 前未泄露。 |
| 2026-05-17 | B_015_019 | decay_forget | decay_B_lunch_spice_015; decay_B_calendar_reminder_016; decay_B_fitness_class_017; decay_B_gift_wrap_018; decay_B_sleep_temperature_019 | validated | 生成 B 类 5 个场景，文件位于 `benchmarks/statistics/B/`；全量 validator 通过 0 fatal / 0 warning，并人工复核长期偏好压过临时兴趣、query 自然询问默认选择且 query 前未泄露。 |
| 2026-05-17 | C_015_019 | reactivation | react_C_seed_tin_015; react_C_bike_tail_light_016; react_C_recipe_card_017; react_C_badge_magnet_018; react_C_synth_patch_019 | validated | 生成 C 类 5 个场景，文件位于 `benchmarks/statistics/C/`；全量 validator 通过 0 fatal / 0 warning，并人工复核早期低频细节、近期主题锚点和 query 前未复述答案。 |
| 2026-05-17 | D_015_019 | semantic_distractor | distractor_D_ceramics_wheel_015; distractor_D_monitor_return_016; distractor_D_library_hold_017; distractor_D_poster_tube_018; distractor_D_dishwasher_pump_019 | validated | 生成 D 类 5 个场景，文件位于 `benchmarks/statistics/D/`；全量 validator 通过 0 fatal / 0 warning，并人工复核当前正确目标、语义相近干扰项、强排序 pairs 和 query 前未泄露。 |
| 2026-05-17 | E_015_019 | stability_check | stable_E_pharmacy_member_015; stable_E_property_account_016; stable_E_library_room_017; stable_E_print_paper_018; stable_E_yoga_mat_019 | validated | 生成 E 类 5 个场景，文件位于 `benchmarks/statistics/E/`；全量 validator 通过 0 fatal / 0 warning，并人工复核稳定事实低频出现、非覆盖干扰项和 query 前未泄露。 |

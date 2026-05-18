# LongMemEval-S 数据集统计

## 总体分布

LongMemEval-S 数据集共包含 **500** 个问题，分布如下：

| 问题类型 | 数量 |
|---------|------|
| multi-session | 133 |
| temporal-reasoning | 133 |
| knowledge-update | 78 |
| single-session-user | 70 |
| single-session-assistant | 56 |
| single-session-preference | 30 |

---

## 三种问题类型详细统计

### 1. multi-session

| 指标 | 数值 |
|------|------|
| 问题数量 | 133 |
| 总session数（平均） | 49.7 (范围: 39-58) |
| 目标session数（平均） | 2.6 (范围: 2-5) |
| session时间跨度（平均） | 8.21 天 |
| 相邻session平均间隔（平均） | 0.17 天 |
| 目标session时间跨度（平均） | 3.62 天 |
| 最早session到问题（平均） | 8.38 天 |
| 最晚session到问题（平均） | 0.17 天 |
| 最早目标session到问题（平均） | 5.99 天 |
| 答案session间隔天数（最晚目标到问题） | 2.37 天 |

### 2. temporal-reasoning

| 指标 | 数值 |
|------|------|
| 问题数量 | 133 |
| 总session数（平均） | 50.0 (范围: 42-65) |
| 目标session数（平均） | 2.2 (范围: 1-6) |
| session时间跨度（平均） | 28.14 天 |
| 相邻session平均间隔（平均） | 0.57 天 |
| 目标session时间跨度（平均） | 13.16 天 |
| 最早session到问题（平均） | 42.08 天 |
| 最晚session到问题（平均） | 13.95 天 |
| 最早目标session到问题（平均） | 27.48 天 |
| 答案session间隔天数（最晚目标到问题） | 14.32 天 |

### 3. knowledge-update

| 指标 | 数值 |
|------|------|
| 问题数量 | 78 |
| 总session数（平均） | 50.0 (范围: 41-59) |
| 目标session数（平均） | 2.0 (范围: 2-2) |
| session时间跨度（平均） | 96.55 天 |
| 相邻session平均间隔（平均） | 1.98 天 |
| 目标session时间跨度（平均） | 62.05 天 |
| 最早session到问题（平均） | 101.11 天 |
| 最晚session到问题（平均） | 4.56 天 |
| 最早目标session到问题（平均） | 77.12 天 |
| 答案session间隔天数（最晚目标到问题） | 15.07 天 |

---

## 对比汇总

| 指标 | multi-session | temporal-reasoning | knowledge-update |
|------|--------------|-------------------|------------------|
| 问题数量 | 133 | 133 | 78 |
| 总session数（平均） | 49.7 | 50.0 | 50.0 |
| 目标session数（平均） | 2.6 | 2.2 | 2.0 |
| session时间跨度（天） | 8.21 | 28.14 | 96.55 |
| 相邻session平均间隔（天） | 0.17 | 0.57 | 1.98 |
| 目标session时间跨度（天） | 3.62 | 13.16 | 62.05 |
| 最早session到问题（天） | 8.38 | 42.08 | 101.11 |
| 最晚session到问题（天） | 0.17 | 13.95 | 4.56 |
| 最早目标session到问题（天） | 5.99 | 27.48 | 77.12 |
| 答案session间隔天数 | 2.37 | 14.32 | 15.07 |

---

## 关键发现

### 1. 总session数
三种类型的平均总session数都在50左右，范围在39-65之间，说明数据集在session数量上保持了较好的一致性。

### 2. 目标session数
- **multi-session**：平均2.6个（2-5个），需要整合多个session的信息
- **temporal-reasoning**：平均2.2个（1-6个），需要时间推理能力
- **knowledge-update**：固定2个，需要追踪知识的更新变化

### 3. 时间跨度差异显著
三种问题类型的时间跨度呈现明显的梯度：

| 类型 | session时间跨度 | 目标session时间跨度 |
|------|---------------|-------------------|
| knowledge-update | 96.55天（最长） | 62.05天（最长） |
| temporal-reasoning | 28.14天 | 13.16天 |
| multi-session | 8.21天（最短） | 3.62天（最短） |

这表明：
- **knowledge-update** 类型的问题需要跨越最长时间的记忆，考验长期记忆能力
- **temporal-reasoning** 类型需要中等时间跨度的时间推理能力
- **multi-session** 类型主要考验跨session的信息整合能力，时间跨度相对较小

### 4. 答案session间隔天数
从最后一个证据session到问题的时间间隔：

| 类型 | 答案session间隔天数 |
|------|-------------------|
| knowledge-update | 15.07天（最长） |
| temporal-reasoning | 14.32天 |
| multi-session | 2.37天（最短） |

**knowledge-update** 和 **temporal-reasoning** 类型的问题对长期记忆的挑战更大，而 **multi-session** 更多是考验近期跨session的信息整合能力。

### 5. 最晚session到问题的时间
- **temporal-reasoning**：13.95天（最长）
- **knowledge-update**：4.56天
- **multi-session**：0.17天（最短，约4小时）

这说明 temporal-reasoning 类型的问题中，最后一个session往往距离问题时间较远，增加了记忆的难度。

---

## 术语解释

### session时间跨度
最早的session和最晚的session之间的时间差。

公式：`max(session_dates) - min(session_dates)`

### 相邻session平均间隔
按时间排序后，每两个相邻session之间的间隔的平均值。

公式：`avg(session_dates[i+1] - session_dates[i] for i in range(n-1))`

### 数学关系
对于单个问题，以下关系成立：
- `相邻间隔之和 = 时间跨度`
- `相邻平均间隔 × (总session数 - 1) = 时间跨度`

注意：n个session之间只有n-1个间隔，所以应该乘以 `(n-1)` 而不是 `n`。

### 答案session间隔天数
从最后一个证据session（目标session）到问题提出时间的天数间隔，反映了需要记住多久以前的信息。

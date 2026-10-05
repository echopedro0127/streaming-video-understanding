# Paired Failure Diagnosis and Recent-6 + Option-Memory-2 Design

状态：已获用户批准，待实现

日期：2026-10-05

## 1. 目标

本阶段回答核心问题：

> Under a fixed visual budget, when does historical semantic memory provide more useful evidence than simply retaining more recent frames?

本阶段包含两个互相依赖但边界清晰的子任务：

1. 对 `sample_51–sample_100` 做 Recent-8 与 Option-Semantic-4+4 的严格 QA-level paired diagnosis。
2. 在相同模型、数据、FPS、prompt、总帧预算和 retrieval 逻辑下，新增 `Recent-6 + Option-Semantic-Memory-2`，验证固定 4+4 分配是否过度牺牲 recent context。

不引入训练、不替换 VLM、不改变 CLIP 模型、不增加总视觉帧预算、不使用 ground-truth answer 做 retrieval。

## 2. 已核对的输入 artifact

主配对输入：

- Recent-8：`.worktrees/scale-100/results/scale-100/recent-8.jsonl`
  - 500 条记录，100 个视频
  - 只过滤 `sample_51–sample_100` 后参与本阶段
- Option-Semantic-4+4：
  - `results/option-semantic-validation-real-time-51-100.jsonl`
  - 500 条记录，包含两个 Option-aware policy
  - 只过滤 `policy_id=recent-4+option-semantic-memory-4`

当前只读配对审计已确认：

- paired QA：250
- Recent-8：95/250 = 38.0%
- Option 4+4：90/250 = 36.0%
- both correct：85
- memory helped：5
- memory hurt：10
- both wrong：150
- pairing missing：0
- net change：`5 - 10 = -5`，即 `-2.0 pp`

这些数字会由新脚本重新计算，不能只依赖本设计文档中的预审计结果。

## 3. Paired diagnosis 设计

### 3.1 Pair key

以 `sample_id` 作为 QA-level pair key。脚本必须：

- 每个输入文件先按 policy 过滤；
- 检查 pair key 唯一性；
- 检查两侧的 `question`、`ground_truth`、`query_time_s` 一致；
- 对缺失、重复或不一致直接报错；
- 不通过位置顺序配对。

### 3.2 四类分组

| Group | Recent-8 | Option 4+4 |
|---|---|---|
| `both_correct` | correct | correct |
| `memory_helped` | wrong | correct |
| `memory_hurt` | correct | wrong |
| `both_wrong` | wrong | wrong |

summary 必须包含：

- 每组 count 与 percentage；
- Recent accuracy；
- Option accuracy；
- `net_gain = helped - hurt`；
- `accuracy_delta == net_gain / n` 的校验结果。

### 3.3 诊断字段

paired CSV 至少包含：

- `sample_id`
- `video_id`
- `question`
- `options`
- `ground_truth`
- `query_time_s`
- `group`
- `recent_prediction`
- `option_prediction`
- `recent_timestamps`
- `recent_recent_timestamps`
- `option_recent_timestamps`
- `option_memory_timestamps`
- recent / memory frame 的 temporal distance
- Option semantic trace
- memory semantic score
- per-frame option scores
- option margin
- retrieval top option
- temporal-relation keyword proxy

`option_margin` 和 retrieval score 只能标记为 retrieval signal，不得称作 calibrated confidence。

### 3.4 Hypothesis tests

脚本输出下列统计，但不预先判定结论：

- H1 recent-context loss：
  - memory-hurt case 中，统计 Recent-8 独有 recent frame 数量；
  - 对比 Option memory 的时间距离及是否提供额外历史证据；
  - 该统计只作为 proxy，不能自动断言关键证据存在。
- H2 distractor option：
  - 统计 memory-hurt 中最终错误 prediction 是否与最高 retrieval option 一致；
  - 汇总 helped / hurt / both-wrong 的 option margin。
- H3 temporal relation：
  - 使用 before/after/first/then/currently/now/still/previously/earlier/later/continue/stop/change 等词做透明 keyword proxy；
  - 同时导出原问题和 case 供人工复核；
  - 不把关键词结果写成 temporal reasoning 的最终证明。
- H4 model bottleneck：
  - 统计 both-wrong 数量及其占 Option 错误的比例；
  - 作为判断继续优化 retrieval 是否值得的证据。

### 3.5 Case artifacts

输出：

- `results/analysis/51-100-recent8-vs-option4-paired.csv`
- `results/analysis/51-100-recent8-vs-option4-summary.json`
- `results/analysis/51-100-recent8-vs-option4-helped.jsonl`
- `results/analysis/51-100-recent8-vs-option4-hurt.jsonl`

helped 与 hurt 每类最多保留 20 个确定性代表 case；全量数据仍保存在 paired CSV，不生成大量图片。

## 4. Recent-6 + Option-Memory-2 设计

### 4.1 Policy contract

新增 policy ID：

```text
recent-6+option-semantic-memory-2
```

其唯一变量是 recent/memory allocation：

```text
recent_capacity = 6
memory_capacity = 2
total visual budget = 8
```

保持不变：

- SmolVLM2-500M-Video-Instruct；
- CLIP scorer；
- question + all options 的 Option-aware retrieval；
- semantic score aggregation；
- non-diverse selection；
- 1 FPS；
- query timestamp；
- prompt；
- answer parsing；
- causal boundary；
- resume/config hashing。

当前 `_policy_parts()` 已经支持通用的 `recent-K+option-semantic-memory-M` 格式，因此优先采用 policy ID 和 config 增量，不做大规模重构。

### 4.2 Config

新增配置：

```yaml
experiment:
  visual_budget: 8
  recent_capacity: 6
  policy_ids:
    - recent-6+option-semantic-memory-2
```

具体配置文件沿用当前 option-semantic validation 的 model、dataset、runtime、retrieval 设置，只改变 policy list 和输出 artifact 名称。

### 4.3 Evaluation scope

只在 `sample_51–sample_100` 的 250 QA 上运行 6+2：

- 复用已有 Recent-8 raw result；
- 复用已有 Option 4+4 raw result；
- 新增 6+2 独立 JSONL 和 profile；
- 不使用 `--overwrite` 覆盖旧结果；
- 支持 `--resume`。

## 5. 测试设计

先写测试再改生产代码。

必须覆盖：

1. policy parser 接受 `recent-6+option-semantic-memory-2`；
2. 6 recent + 2 memory 的选择容量不超过 8；
3. historical frame 数量不足时仍保持可运行；
4. 所有 selected timestamp 不超过 query timestamp；
5. selection deterministic；
6. Option scorer 收到 question 和全部 options，不收到 ground-truth；
7. 6+2 的 config ID 与 4+4 不同，resume 不会复用错误 policy 的记录；
8. paired analyzer 能检测缺失、重复和 question/GT mismatch；
9. paired accuracy delta 与 helped/hurt 差值一致。

## 6. 验证顺序

1. 运行 paired analyzer 的单元测试；
2. 运行 analyzer 读取真实 51–100 artifact，核对 n=250 和分组总数；
3. 写 policy/config failing tests；
4. 实现 6+2 最小 policy 增量；
5. 运行 focused tests；
6. 运行 full pytest；
7. 运行 6+2 mock smoke；
8. 运行 51–100 的 6+2 real-model evaluation；
9. 对 6+2 做 Recent-8 与 4+4 paired comparison；
10. 生成最终 summary，并更新项目状态文档。

## 7. 决策规则

- 若 6+2 同时超过 Recent-8 与 4+4：支持“固定 4+4 过度牺牲 recent context”，下一步只设计 adaptive allocation，不在本阶段实现 routing。
- 若 6+2 约等于 Recent-8 且超过 4+4：说明少量 memory 能减少 4+4 伤害，下一步研究 query-dependent gating。
- 若 6+2 低于 Recent-8：优先根据 paired diagnosis 判断 retrieval、temporal relation 或 VLM bottleneck，不进行大量 5+3/7+1 grid search。
- 若 both-wrong 占主导：提出固定 selected frames、更换 stronger VLM 的 20–30 case diagnostic；本阶段不自动下载大模型。

## 8. 非目标

- 不下载完整 203GB StreamingBench；
- 不运行 K=16、M=8、M=16 grid；
- 不训练 LoRA/SFT；
- 不增加 dependency；
- 不使用未来帧；
- 不使用 GT answer 做 retrieval；
- 不把 option margin 写成 calibrated confidence；
- 不把某个小样本 case 的观察写成普遍结论。

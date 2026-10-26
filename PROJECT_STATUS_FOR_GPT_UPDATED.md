# Streaming Video Understanding 当前项目状态

更新时间：2026-10-26（下文大规模基准表保留 2026-10-05 的已完成结果）

## 最新进展：可人工核验的单帧感知识别

- 已完成盲审材料与prepare/run/score工具，从先前3个预选案例定向抽24帧。
  类别为讲解者／插入片段／空白过渡／无法判断；没有预填真实标签。
- 同一SmolVLM2-500M、单图、两种选项顺序，24对／48次生成全部完成。
  48/48可解析，语义类别一致23/24（95.8%），显示字母一致0/24。
- 95.8%是选项重排后的语义一致性，不是识别准确率；人工标签尚无，准确率保持null。
  单帧任务、提示内容与输入数量均有变化，不能直接证明此前计数失败的内部原因。
- 图像和来源哈希、固定选帧、因果边界、提示词、映射、协议、运行错误均0；
  首顺序12/12，resume跳过24对且原文件哈希不变，526项测试通过，独立复核完成。
- 人与模型使用同一组JPEG；清单不含耗时字段，已公开固定帧ID及图像哈希，
  无需依赖上一轮含耗时的完整原始结果文件。真实JPEG和人工标签仅留本地。
- 本地盲审页：`results/perception-probe/review.html`；用户需逐帧核验并将
  `human-labels.json`保存到同目录。页面不显示逐帧模型输出；来源为用户自报人工核验，
  并非独立认证或双盲实验。Node已测试导出逻辑，实际浏览器布局与保存框尚待确认。
- 评分要求完整标签和匹配清单；排除人工“无法判断”帧并报告覆盖率、逐类召回，
  两个主类别都有人工样本时才报告平衡召回。之后再决定是否构建感知→事件累计状态。
- 报告：`experiments/2026-10-26/perception-probe/README.md`。

## 前一实验：固定选帧的视觉／选项敏感性诊断

- 完成20视频/20条计数QA、80次生成，锁定上一轮Change-4+4选帧，不调参。
- 原图、同尺寸灰色空白图、帧序反转均4/20（20%）；选项循环置换后8/20（40%）。
- 空白输入仍有17/20答案内容不变；反序18/20。
  选项置换后仅1/20答案内容保持，但18/20仍输出相同显示字母。
- 置换条件helped8/hurt4，差值+20pp，视频bootstrap区间[-15,+55]pp；
  恒答显示字母C也分别得到20%／40%，不能把40%当作视频理解提升。
- 原图条件的选帧、预测、原始输出、计数20/20复现旧结果；未来帧、帧预算、
  来源、提示词、选项映射、协议、运行错误均0，首位置轮换5/5/5/5。
- smoke/full resume分别跳过4/20条且哈希不变，513项测试通过，独立复核完成。
- 结论限于这20条已检查计数题：选项位置依赖明显，对这些视觉扰动敏感性有限。
  不能证明模型完全不看图；总次数可能天然对反序不变，不能据此判定时序能力缺陷。
- 本诊断依赖本机封存的精确来源JSONL，哈希含耗时字段，重新运行会变化；
  公开摘要不足以开箱复现该诊断。可移植选帧来源契约尚待补齐。
- 下一步优先人工核验片段/讲解画面识别与事件边界，再研究累计事件状态；
  不继续基于这20题调选帧阈值，也不把单题收益写入简历。
- 报告：`experiments/2026-10-26/visual-sensitivity/README.md`。

## 前一实验：画面变化代表帧记忆（Change-4+4）

- 同组20视频/20条计数QA，固定8帧、模型和MCQ提示词，共60次推理完成。
- 新方法：相邻RGB32x32缩略图的平均绝对像素差，固定阈值0.08、冷却2秒；
  最近4帧＋历史4个高分代表帧，不足时确定性补位。它检测画面变化，不是语义事件。
- Change-4+4为4/20（20%），Recent-8及全前缀分箱均3/20（15%）；
  helped1/hurt0，视频配对bootstrap区间[0,+15]pp，不能认定稳定提升。
- 关键负面对照：新方法19/20输出C；事后“恒答C”也4/20，逐题正确性向量完全一致。
  这不证明模型忽略图像，但不足以把单题改善归因于记忆机制，不可写成普遍+5pp收益。
- 未来帧/预算/历史结构/协议/运行错误均0；两种旧控制共40组预测与选帧完全复现。
  变化记忆自身峰值8张；联合诊断容量上限32张加到达帧，不含解码器内部缓冲，
  不是全进程RAM峰值。共享GPU分配峰值3537.18MiB。
- resume跳过20条且哈希不变，503项测试通过。代码复核发现的异常清理路径已在推理前修复。
- 事件人工标注仍待完成；已保存3案例核验队列，粗略区间不作官方标签或evidence recall。
  下一步优先视觉输入敏感性与选项置换对照，再决定是否扩大规模或累计事件状态。
- 报告：`experiments/2026-10-26/change-memory/README.md`。

## 前一实验：固定8帧预算的时间覆盖诊断

- 同一组20视频/20条计数QA，对比Recent-8、最近60秒分8箱、完整因果前缀分8箱。
- 60次推理全部完成，三组均3/20（15%）。平均选帧时间跨度为7.007/45.696/200.350秒。
- 两种分箱相对Recent-8均helped0/hurt0；本样本bootstrap区间退化为[0,0]，
  不能据此声称总体等效。扩大时间跨度本身没有带来本轮准确率收益。
- 未来帧/窗口范围/帧预算/重复/协议/运行错误均0，Recent-8选帧和预测20/20复现。
  5条短前缀控制两种分箱输入和输出一致，resume跳过20条且原文件哈希不变。
- 实际保留图像峰值23张，上限24张加1张到达帧（不含解码器内部缓冲）；
  三条件共享GPU分配峰值3537.18MiB，不是独立方法显存测量。全量测试492 passed。
- 3份预选案例的1FPS图表已辅助检查：全历史分箱仍可能漏掉两代表帧之间的短片段。
  观察尚待人工核验，不能当作官方事件标签或evidence recall。
- 该实验知道query时间后构造分箱，是因果前缀诊断，不是新的持久在线记忆方法；
  不覆盖原500 QA基准。下一步优先核验事件区间，再测试事件变化保留/累计状态。
- 报告：`experiments/2026-10-26/context-coverage/README.md`。

## 前一实验：固定选帧的直接计数读出

- 排除此前11案例涉及的所有视频后，固定哈希规则从20个计数视频各取一题。
  这些视频参与过早期benchmark，不能称为全新未见测试。
- 同组Recent-8图像比较MCQ与不展示选项的整数计数，20对/40次生成，先后顺序10/10。
- MCQ3/20（15%），直接计数2/20（10%）；差值-5 pp，视频CI95[-25,+15] pp，未证明收益。
- 两种输出均20/20可解析；直接计数17/20输出1，15/20在候选外，无最近选项修正。
- 缺失/重复/协议/运行错误均0，MCQ预测及选帧20/20复现旧结果，resume跳过20且文件不变。
- 后续优先检查证据可回答性、时间覆盖和事件累计；本实验不改写250 QA成绩，
  也不能证明某一种内部机制。全量测试474 passed。
- 报告：`experiments/2026-10-26/choice-free-count/README.md`。

## 最新实验：计数题选项顺序诊断

- 对此前 4 helped + 7 hurt 的 11 QA 完成 66 次推理：22 条同配置未置换对照，
  两种方法 × 两个种子（17/29）共 44 条选项置换；全部 profile 完整，运行和协议错误均为 0。
- 未置换对照的预测、选帧与旧记录 22/22 完全一致。44 次比较逐条验证配置 ID、
  实际提示词、选项映射、时间戳和选帧清单不变。
- 固定选帧后的答案内容一致率：Recent-8 6/22（27.3%），自适应路由 7/22（31.8%）。
  自适应路由 15/22 次改变了答案内容，但子集准确率仍为 4/11，正确与错误互换相抵消。
- 结论：这些计数案例存在明显的选项顺序敏感性，不能将所有答题变化归因于 memory。
  这是事后挑选案例的稳定性诊断，不替代 250 QA 准确率，也不是校准置信度。
- 下一步优先固定选帧做读出稳定性干预，并使用独立视频或新抽样计数集验证；
  证据覆盖、60 秒截断与事件计数仍需独立视觉标注。
- 全量测试 450 passed；完整报告：`experiments/2026-10-26/README.md`。

## 最新实施进展：选择性时序记忆

- 已实现 `adaptive-semantic-v1`：仅根据问题文字，在 8+0 / 6+2 / 4+4
  之间路由，使用 Q-only 检索；输出包含规则原因、版本和实际分配。
- 已新增按视频聚类的配对 bootstrap、关键词代理分组和 60 QA 盲化证据标注队列。
  人工证据标签仍为空，不能据此报告 evidence recall。
- 后 50 视频：Recent-8 95/250，Option-aware 4+4 90/250；helped/hurt=5/10；
  差值 -2.0 pp，95% 视频聚类 bootstrap 区间 [-5.2, +1.2] pp，未证明稳定优劣。
- 6+2 已完成后 50 视频的 250 QA：93 正确（37.2%），缺失/重复/未来帧/预算超限/运行错误均为 0。
  对比 Recent-8：helped/hurt=4/6，-0.8 pp，CI95 [-3.2, +1.6] pp；
  对比 Option-aware 4+4：4/1，+1.2 pp，CI95 [-0.4, +2.8] pp。两者区间均跨零。
  新路由初始 5 QA、独立 3 分支真实检查均跑通；resume 跳过 3/3 且文件校验值不变。
  新路由已完成 250/250：92 正确（36.8%），250 条 profile，审计和运行错误均为 0。
  相对 Recent-8：helped/hurt=4/7，-1.2 pp，视频聚类 CI95 [-4.0, +1.6] pp。
  当前路由 98 题的预测和选帧均与 Recent-8 完全一致；计数组为 31/150，对照 34/150。
  未证明整体准确率提升，后续应优先审查这 11 条配对变化的证据，而非继续增加规则。
  6+2 与新路由的小型公开结果及源文件校验值见 `experiments/2026-10-25/`。
- 当前代码全量测试 **474 passed**，包含新路由、旧基线兼容性、resume 和诊断工具。
- 当前候选范围为最近 60 秒，尚未实现持久增量摄取；本次新增结果会记录该配置范围。
- 详细实施、测量和后续步骤见 `docs/SELECTIVE_MEMORY_STAGE1.md`。

本文档用于与 GPT 讨论下一步研究方案。所有数字均来自当前本地结果文件；100 视频结果已经完成，但不是所有策略都在 100 视频上运行。

## 1. 项目目标与当前 setting

项目研究问题：

> Can a lightweight temporal memory improve streaming video understanding without repeatedly encoding the entire video history?

当前实现的是严格因果、training-free 的 Streaming Video QA evaluator：

- 模型：`SmolVLM2-500M-Video-Instruct`
- 数据：StreamingBench Real-Time Visual Understanding
- 采样：1 FPS
- 查询时刻只能访问 `timestamp <= query_time`
- 默认视觉预算：8 帧
- 策略：Recent Window、Uniform History、Uniform/Temporal Memory、Q-only Semantic Memory、Option-aware Memory
- GPU：当前运行环境可见显存约 6GB；没有使用大型模型或完整 203GB 数据集

## 2. 已完成的工程能力

- 严格 causal frame selection 与 future-frame leakage audit
- Recent / Uniform / bounded temporal memory policy
- Q-only CLIP semantic retrieval
- Option-aware semantic retrieval：使用问题与四个候选答案的联合表征，不使用 ground-truth
- temporal diversity memory selection
- deterministic seed、YAML config、JSONL 输出、resume
- P50/P95 latency、throughput、peak GPU memory、input/retained frames
- failure JSONL 与 selected-frame / timestamp trace
- bounded decoded-frame cache、逐 query image release、CUDA cleanup
- Windows/6GB 显存环境下的长跑内存回归测试

内存修复后的测试结果：

- focused tests：36 passed
- full pytest：307 passed
- 100 视频 Option-aware 补跑：0 runtime error、0 causal violation、0 frame-budget violation

## 3. 50 视频 / 250 QA 结果

以下结果覆盖 `real-time-1-50` 的 250 个 QA。每个策略 250 条记录；Option-aware 两个策略合计 500 条记录。

| Policy | Correct / 250 | Accuracy |
|---|---:|---:|
| Recent-8 | 163 | 65.2% |
| Uniform-8 | 157 | 62.8% |
| Recent-4 + Uniform-Memory-4 | 160 | 64.0% |
| Recent-4 + Q-only Semantic-Memory-4 | 159 | 63.6% |
| Recent-4 + Q-only Semantic-Diverse-Memory-4 | 157 | 62.8% |
| Recent-4 + Option-Semantic-Memory-4 | 164 | 65.6% |
| Recent-4 + Option-Semantic-Diverse-Memory-4 | 164 | 65.6% |

Option-aware 的配对结果：

- 相对 Recent-8：两种策略均为 `+0.4 pp`
- 相对 Q-only Semantic-Memory：两种策略均为 `+2.0 pp`
- Option-aware semantic：helped / hurt / unchanged = `10 / 9 / 231`
- Option-aware diverse：helped / hurt / unchanged = `12 / 11 / 227`
- 0 runtime error、0 causal violation、0 budget violation、0 option-schema violation
- 峰值显存约 `3834 MB`

该结果只观测到前 50 视频上的微小准确率差异，尚不能证明稳定收益或推广到完整 100 视频。

## 4. 100 视频 / 500 QA 扩展结果

### 4.1 后 50 视频补跑

`sample_51–sample_100` 共 250 QA，在两个 Option-aware 策略下产生 500 条记录：

| Policy | Correct / 250 | Accuracy |
|---|---:|---:|
| Option-Semantic-Memory-4 | 90 | 36.0% |
| Option-Semantic-Diverse-Memory-4 | 89 | 35.6% |

后 50 视频结果中：

- 合并两策略准确率：`35.8%`
- mean latency：`20945.34 ms`
- P50 latency：`19095.17 ms`
- P95 latency：`48150.63 ms`
- peak GPU memory：`3834.31 MB`
- average input frames：`7.98`
- 321 条 failure 记录全部是模型回答错误，不是程序错误

### 4.2 完整 100 视频合并结果

将前 50 与后 50 的两个 Option-aware 结果拼接后，共 1000 条记录：

| Policy | Correct / 500 | Accuracy | Mean latency | P50 | P95 | Peak GPU |
|---|---:|---:|---:|---:|---:|---:|
| Option-Semantic-Memory-4 | 254 | 50.8% | 24.89 s | 23.68 s | 58.22 s | 3.83 GB |
| Option-Semantic-Diverse-Memory-4 | 253 | 50.6% | 21.91 s | 19.12 s | 58.50 s | 3.83 GB |

已有完整 100 视频基线结果：

| Baseline | Accuracy |
|---|---:|
| Recent-8 | 51.6% |
| Recent-4 | 51.8% |
| Recent-4 + Timestamp-aware-Memory-4 | 48.6% |

当前结论：

1. Option-aware 在前 50 视频上相对 Q-only semantic memory 有 +2.0 pp。
2. 在完整 100 视频上，Option-aware 为 50.8% / 50.6%，没有超过 Recent-4 的 51.8%。
3. 后 50 视频从前 50 的约 65.6% 降至约 35.8%，存在明显 long-tail / split difficulty 问题。
4. 当前最可靠的工程性结论是：有限帧预算、严格因果约束、长跑 resume 和显存控制已经稳定；准确率提升尚未在完整规模上泛化。

100 视频结果文件：

- `results/option-semantic-validation-real-time-51-100.jsonl`
- `results/option-semantic-validation-real-time-51-100-summary.json`
- `results/option-semantic-validation-real-time-1-100.jsonl`：当前包含前 50 视频的 Option-aware 结果，文件名中的 `1-100` 不能视为完整 100 视频单文件结果

## 5. 当前尚未执行或仅部分执行的方案

### 完整 100 视频矩阵

尚未在完整 100 视频上完成：

- Uniform-8
- Q-only Semantic-Memory-4
- Q-only Semantic-Diverse-Memory-4
- Recent-4 + Uniform-Memory-4

因此当前不能对所有策略做严格的 100 视频公平排名。

### Memory / Window ablation

- `M=0` 可由 Recent-only 近似覆盖
- `M=4` 已有主要结果
- `M=8`、`M=16` 尚未完成
- `K=16` 只做过硬件探测，因 6GB 显存和上下文长度边界停止，未作为正式结果

### Ambiguity-aware query

以下设计仍未正式实现：

- retrieval confidence calibration
- option permutation stability 的完整评测
- answer confidence + retrieval confidence 融合
- 高置信度直接回答、低置信度请求澄清的 routing
- precise / vague paired query 数据集
- clarification rate、clarification precision、risk-coverage、post-clarification accuracy

当前记录了 option score、option margin 和 semantic trace，但它们还不是经过校准的 answer confidence。

### Failure analysis 与展示

- 已有 50 视频子集的 30 个 failure cases 人工分析
- 100 视频扩展后的后 50 failure cases 尚未重新抽样、分类和制作 contact sheet
- 尚未形成最终 demo / architecture figure

### Optional model extension

尚未执行，也暂不建议优先执行：

- Qwen2.5-VL-3B-AWQ
- Cosmos3-Nano hosted cross-model check
- 额外训练或 LoRA

## 6. 建议与 GPT 重点讨论的问题

下一步需要在以下两条路线中选择：

### 路线 A：先补齐实验矩阵

完成缺失的 100 视频 baseline，得到同一规模下的公平比较，再做 per-video accuracy、difficulty bucket 和 failure analysis。

### 路线 B：优先做 ambiguity-aware system

基于已有 option score / retrieval trace，实现 permutation stability、confidence calibration 和 answer/clarify routing，形成“在线视频理解 + 不确定性处理”的新贡献。

建议 GPT 重点判断：

1. 在完整 100 视频上没有准确率增益时，项目的主贡献应定位为效率/因果性/有界记忆，还是继续优化 frame selection？
2. 是否值得先补齐完整 100 视频矩阵，再实现模糊 query routing？
3. 如何利用后 50 视频的失败分布设计下一版 retrieval，而不是盲目扩大 memory？
4. 是否应把 Option-aware 定义为 benchmark-specific ablation，把 Q-only semantic memory 作为更通用的主方法？
5. 在 6GB 显存条件下，下一轮最值得做的最小实验是什么？

## 7. 真实性边界

- 不把前 50 视频 65.6% 写成完整 100 视频结果
- 不把 Option-aware 写成完整规模上的 accuracy improvement
- 不把 option margin 写成 calibrated confidence
- 不声称已经实现 clarification system
- 不声称使用 COT 进行了置信度判断
- 不虚构尚未完成的 K=16 / M=8 / M=16 / 大模型结果

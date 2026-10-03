# Ambiguity-Aware Query Memory 设计

日期：2026-10-03

分支：`feature/optimization-stage1`

状态：设计已确认，尚未实施

## 1. 背景与问题

当前项目在 StreamingBench 的实时多选视频问答上实现了严格因果的流式评测：模型在查询时刻只能访问该时刻及以前的帧，并在固定帧预算下比较 Recent Window、Uniform History 和 bounded temporal memory。现有 1 FPS、100 QA 验证实验中，`Recent-4 + Uniform-Memory-4` 达到 50% 准确率，而已有 timestamp-aware 策略为 44%。这表明“加入时间分数”本身不足以保证有效记忆，下一步更有价值的问题是：在不增加视觉上下文预算的情况下，能否依据用户问题保留真正相关、时间上互补的历史证据。

同时，真实在线用户的问题可能不完整、模糊，甚至对应多个时间片段。系统不应在证据不足时盲目输出答案，而应估计不确定性：高置信度时直接回答，低置信度时请求澄清。

本设计把这两个问题统一为：

> 在严格因果、固定内存预算下，使用 query-aware semantic retrieval 与 temporal diversity 选择历史帧，并融合答案、检索和稳定性信号，决定直接回答还是请求澄清。

## 2. 目标与非目标

### 2.1 目标

- 保留现有 SmolVLM、评测协议、基线和全部历史结果。
- 在主实验的 8 帧总预算下，将局部 recent context 与全局 semantic memory 结合。
- 对多选 QA 区分通用的 Q-only 检索和 benchmark-specific 的 Q+A hypothesis 检索。
- 对模糊 query 显式处理多时刻匹配和证据不确定性。
- 建立可校准的“回答或澄清”机制，并报告 selective prediction 指标。
- 保持 RTX 3060 12GB 可运行，不引入大型模型或复杂训练流程。

### 2.2 非目标

- 不训练神经记忆模块。
- MVP 不引入 Agent、外部 LLM planner 或复杂编排。
- 不替换当前 SmolVLM 推理后端。
- 不复制 Beyond Caption-Based Queries 项目的代码或训练结构。
- 第一轮研究实验不把开放式生成质量作为主指标；开放式问答仅用于 demo。
- 在实验完成前不宣称 SOTA、准确率提升或方法新颖性。

## 3. 现有系统边界

当前多选 prompt 要求模型仅输出 `A`、`B`、`C` 或 `D`，SmolVLM 使用 `do_sample=False`，默认最多生成 8 个新 token。系统目前没有显式生成 chain-of-thought，也没有返回选项 token 的概率或候选答案 log-probability。

因此，第一版不会要求模型生成长 CoT，也不会用“CoT 与最终答案的语义相似度”作为主要置信度。错误答案可以拥有语言上自洽的事后解释，所以这种相似度主要测量内部一致性，不能可靠反映视觉证据是否充分。若后续加入解释，只生成可审计的短 evidence trace，并检查它是否与选中帧一致；它是辅助信号，不是主要置信度。

## 4. 总体架构

```text
streaming frames
      |
      v
lightweight visual feature cache
      |
      +------------------------------+
                                     v
user query --> Q-only embedding --> semantic retrieval
      |
      +--> optional per-option Q+A hypotheses (MC-only ablation)
                                     |
                                     v
                        temporal clustering / diversity selection
                                     |
                                     v
                       recent frames + bounded memory frames
                                     |
                                     v
                              SmolVLM answer
                                     |
                                     v
                    confidence fusion and routing
                         /                     \
                direct answer              clarification
```

研究评测继续使用多选 QA，以便准确计算 accuracy、候选概率和 option permutation stability。面向展示的 demo 可以接受开放式自然语言问题，但与 benchmark 指标分开报告。

## 5. 严格因果与固定预算

任意查询时刻 `t_q`，候选帧必须满足：

```text
frame_timestamp <= t_q
```

线索区间、答案标注和未来帧只能用于离线评测，绝不能进入模型输入、检索特征、记忆更新或置信度计算。Evaluator 必须记录并断言所有选中帧满足因果截止条件；发现未来帧时立即失败，而不是静默过滤后继续。

主实验保持总视觉预算 `B=8`：

- recent buffer：4 帧；
- historical memory：4 帧；
- 对照组 Recent-8 和 Uniform-8 同样最多使用 8 帧。

扩展实验可沿用既有 `K/M` 配置，但不得通过增加总帧数为新方法取得不公平优势。

## 6. Query 表征设计

### 6.1 Q-only：主方法

主检索特征只编码问题文本 `q`。这是最接近真实在线场景的设定，因为用户通常不会提供候选答案，也使研究方法可迁移到开放式问答。

### 6.2 Q+A hypotheses：多选 benchmark 消融

对于候选答案 `a_j`，构造四个独立假设：

```text
h_j = [question; candidate option j]
```

逐个编码 `h_j`，分别计算帧相关度，再使用与选项标签无关的公平聚合得到 `s_qa(t)`。第一版优先比较 `max` 与 `mean` 聚合，不将全部选项拼接成一个长文本，因为简单拼接会稀释语义，并容易偏向更长或措辞更具体的选项。

正确答案/ground truth 永远不能用于 query 表征。Q+A 分支仅使用推理时本来就可见的全部候选项，并作为 benchmark-specific ablation，不替代 Q-only 主方法。

需要增加测试，证明修改任意 ground-truth 字段不会改变检索结果。

## 7. 历史帧评分与选择

对因果候选历史帧 `f_t`，基础信号包括：

- `s_q(t)`：帧与 Q-only query 的跨模态语义相似度；
- `s_qa(t)`：帧与全部 Q+A hypotheses 的公平聚合相似度，仅用于多选消融；
- `retrieval margin`：最相关候选与次相关候选或次优时间簇的差距；
- `temporal coverage`：候选对尚未覆盖时间区域的贡献；
- `redundancy`：候选与已选帧在视觉或时间上的重复程度；
- `recency`：可选的弱先验，不允许压过语义证据。

候选综合分数采用可解释的线性形式：

```text
S(t) = alpha * s_q(t)
     + beta  * s_qa(t)
     + gamma * temporal_coverage(t)
     + rho   * recency(t)
     - lambda * redundancy(t)
```

Q-only 主实验设 `beta=0`。所有分量必须在组合前归一化，权重只在 validation split 上选择并固定。若轻量视觉—文本特征模型不可用，系统回退到现有 uniform memory，而不是加载更大的模型或破坏评测。

### 7.1 Temporal clustering 与 diversity

模糊问题可能匹配多个不相邻时刻。直接选相似度最高的 top-K 帧，容易把名额浪费在同一短片段的相邻帧。第一版采用轻量、确定性的时间聚类与多样性选择：

1. 按语义分数获得候选池；
2. 将时间相近的候选合并为 temporal clusters；
3. 先从不同高分簇中各取代表帧；
4. 剩余名额使用 MMR 风格的相关性减冗余规则补齐；
5. 分数相同时按时间戳和稳定索引确定性排序。

这比直接复制训练式 temporal grounding 模型更适合当前无训练、单卡、固定帧预算的目标。

## 8. 从 Beyond Caption-Based Queries 借鉴的内容

本地论文与代码位于：

- `beyond.pdf`
- `Beyond_Caption-Based_Queries_for_Video_Moment_Retrieval/`

可借鉴的研究洞察：

- language gap：自然用户查询与视频 caption 风格描述存在差距；
- multi-moment gap：模糊查询可能对应多个合理时刻；
- 构建 precise、mild vague、strong vague 的配对查询；
- 用时间簇而不是单个 top-1 时刻表示多解性；
- 自动改写后必须验证语义保真，并进行人工抽查或修正。

不直接采用的部分：

- CG-DETR 的 decoder self-attention removal 与 query dropout 属于训练式 temporal grounding 改造，不适用于当前 SmolVLM 帧选择器；
- 本地改写脚本使用 Gemma-3-12B，超出当前轻量 MVP 的资源目标；
- 本地匹配代码使用 `stsb-roberta-large` 和 0.85 cosine threshold，主要处理文本—文本匹配，不能直接充当帧—问题跨模态检索器；
- 本地仓库根目录未发现统一 LICENSE，因此只重实现公开论文思想并引用来源，不复制其代码。

相关资料：

- 论文：https://arxiv.org/html/2603.02363v1
- 项目页：https://davidpujol.github.io/beyond-vmr/
- TVQA+ grounding 数据说明：https://aclanthology.org/2020.acl-main.730/

## 9. 置信度与回答/澄清路由

### 9.1 第一版置信度信号

按以下顺序实现：

1. `C_answer`：最终 A/B/C/D 候选的归一化 log-probability、概率 margin 或 entropy；
2. `C_retrieval`：检索 top-1/top-2 margin、检索 entropy，以及高分 temporal clusters 是否竞争；
3. `C_stability`：对候选项进行确定性 permutation 后，映射回原答案语义，检查预测是否稳定。

融合形式：

```text
C = w1 * C_answer + w2 * C_retrieval + w3 * C_stability
```

各信号转换为方向一致、范围一致的分数。权重和阈值仅在 validation split 上校准，锁定后再运行最终 test，防止用测试答案调阈值。

可选后续消融加入：

```text
C = w1 * C_answer
  + w2 * C_retrieval
  + w3 * C_stability
  + w4 * C_grounding
```

其中 `C_grounding` 来自短 evidence trace 与所选帧的可验证一致性。第一版不生成长 CoT，也不把解释相似度当作真实性证明。

### 9.2 路由策略

- 高置信度：直接回答；
- 低置信度：请求澄清；
- 多个相近高分时间簇：澄清问题应描述候选事件或大致时间区域，让用户选择；
- 非交互 benchmark：请求澄清等价为 `abstain`，不得把 abstain 计为普通错误后隐藏 selective performance。

澄清示例：

```text
你指的是视频前半段第一次出现该动作，还是后半段再次发生的那次？
```

## 10. 最小实验矩阵

为避免组合爆炸，实验分三层进行。

### 10.1 主方法比较

在同一 8 帧预算、同一数据和同一生成配置下比较：

1. Recent-8；
2. Uniform-8；
3. Recent-4 + Uniform-Memory-4（当前最强可解释基线）；
4. Recent-4 + Q-only Semantic-Memory-4；
5. Recent-4 + Q-only Semantic-and-Temporal-Diversity-Memory-4（目标方法）。

现有 timestamp-aware 结果完整保留并在结果表中作为历史对照，不要求重新命名或覆盖。

### 10.2 Query 表征消融

只在 validation 子集比较：

- Q-only；
- per-option Q+A hypotheses with mean aggregation；
- per-option Q+A hypotheses with max aggregation。

选定规则后固定，不在最终测试集继续搜索。

### 10.3 置信度消融

- answer confidence only；
- answer + retrieval confidence；
- answer + retrieval + option permutation stability。

短 evidence grounding 作为 optional extension，不进入第一轮完成标准。

## 11. 模糊 Query 数据与评测

第一轮建立 50–100 个 paired queries，每个原始 precise query 对应：

- mild vague rewrite：省略部分实体、动作或时间限定，但仍保持原意；
- strong vague rewrite：显著欠指定，可能对应多个时间片段。

每个改写需保存 source question、rewrite level、语义保真检查结果，以及是否存在多个合理时间簇。自动生成只能作为草稿，必须进行规则验证和人工抽查；无法确认语义一致的样本不进入主结果。

线索时间区间只用于离线评测：

- evidence recall@M；
- multi-moment / temporal-cluster coverage；
- 所选帧到线索区间的时间距离。

它们永远不能输入检索器或回答模型。

## 12. 指标

### 12.1 理解与检索

- QA accuracy；
- evidence recall@M；
- temporal diversity；
- multi-moment coverage；
- option permutation consistency。

### 12.2 Selective prediction

- auto-answer coverage；
- selective accuracy；
- risk-coverage curve；
- clarification rate；
- clarification precision；
- post-clarification accuracy，在模拟或真实澄清回复可用时报告。

### 12.3 效率与安全

- 平均、P50、P95 latency；
- throughput；
- peak GPU memory；
- feature-cache disk/memory usage；
- 总输入帧数与 memory slots；
- confidence 与 permutation 的额外延迟；
- future-frame leakage 数量，必须为 0。

## 13. 错误处理与回退

- 模型输出无法解析为选项：记录原始输出，标记 malformed prediction，并进入低置信度路由；
- embedding 缺失或损坏：记录明确错误，单样本回退到 uniform memory；
- 无历史候选帧：只使用可用 recent frames；
- 分数相同：按时间戳、帧索引确定性打破平局；
- 多个时间簇无法区分：请求针对候选事件的澄清；
- 用户引用对象不清：请求补充人物、动作或时间信息；
- 任意候选帧越过 query timestamp：立即中止该样本并报告 leakage，不允许回退掩盖错误；
- OOM：沿用现有安全处理，释放缓存后以更小推理单元重试，不改变帧选择策略。

## 14. 测试要求

### 14.1 单元测试

- ground truth 字段变化不影响 query embedding 和检索结果；
- 所有选中帧满足 causal cutoff；
- score normalization 与权重组合数值正确；
- 相邻高分帧会被 temporal diversity 约束；
- 同分候选产生确定性结果；
- option permutation 后答案映射回原语义保持一致；
- 高/低置信度阈值产生正确路由；
- 多簇竞争触发时间定向澄清；
- 缺失 embedding 时正确回退。

### 14.2 集成测试

使用 mock backend 和小型合成视频时间轴，端到端验证：

```text
load sample
-> apply query cutoff
-> retrieve causal frames
-> select diverse memory
-> predict option
-> compute confidence
-> answer or abstain/clarify
-> save trace
```

测试输出必须包含所选 recent/memory 帧、时间戳、各评分分量、置信度分量和路由决定，便于 failure analysis。

## 15. 实施阶段与交付物

### 15.1 MVP

- 定义统一的 frame feature cache 接口；
- 实现 Q-only semantic memory；
- 实现 temporal clustering 与 diversity selection；
- 接入现有 causal evaluator 和结果记录；
- 在小规模 mock/real smoke test 上验证无 future leakage。

### 15.2 Core Experiment

- 完成 5 个主方法的同预算比较；
- 完成 Q-only 与 Q+A hypotheses 消融；
- 提取答案候选 log-probability；
- 实现 retrieval confidence 与 option permutation stability；
- 校准回答/澄清阈值并生成 risk-coverage 曲线；
- 自动保存 failure cases 和检索 trace。

### 15.3 Optional Extension

- precise/mild/strong vague paired-query benchmark；
- 基于候选时间簇生成针对性澄清问题；
- 短 evidence trace 与视觉 grounding 辅助置信度；
- 开放式自然语言在线 demo。

## 16. 验收标准

设计落地后至少满足：

- 现有测试、基线配置和历史结果不被覆盖；
- 主方法始终使用固定 8 帧预算；
- 自动 causal assertion 证明 future leakage 为 0；
- ground truth 不参与任何检索或置信度输入；
- 至少完成 Q-only semantic memory 与 semantic + temporal diversity memory；
- 报告 accuracy、evidence recall、latency、peak VRAM 和 memory usage；
- 报告 risk-coverage 与 selective accuracy，而不是只给一个人工阈值结果；
- 置信度阈值只在 validation 上校准；
- 失败案例可追溯到具体帧、时间簇、评分和路由决定；
- RTX 3060 12GB 上可复现，无大型新依赖或模型下载。

## 17. 实施顺序

1. 先增加测试与统一 trace schema，锁定因果和无 GT 泄漏约束；
2. 增加轻量 frame/query feature 接口与离线缓存；
3. 实现 Q-only semantic retrieval；
4. 实现 temporal clustering 和 diversity selection；
5. 接入现有 evaluator，完成同预算 smoke test；
6. 在 validation 上做 Q+A hypotheses 消融；
7. 修改 SmolVLM backend，提取候选答案分数；
8. 实现 retrieval confidence 与 option permutation stability；
9. 校准并评测 answer/clarify 路由；
10. 最后再扩展 vague-query paired set 和 demo。

该顺序优先验证帧选择是否真正改善证据覆盖，再增加置信度与交互层，避免把“检索不准”和“回答模型不确定”混为同一个问题。

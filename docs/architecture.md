# 架构与能力边界

| 模型 | 职责 |
|---|---|
| BAAI/bge-base-zh-v1.5 | 查询/法条编码、语义原型路由、离线候选挖掘 |
| BAAI/bge-reranker-base | CrossEncoder 候选精排 |
| Qwen/Qwen2.5-1.5B-Instruct | 本地短提示生成、完整提示 token 计数 |
| DeepSeek | QU、争点规划、证据核验、工具选择、云端生成及离线样本判别 |

```text
API / CLI → 查询理解与争点计划 → DeepSeek Tool Calls
    search_evidence: BM25 + BGE/FAISS → RRF → BGE-Reranker
                      → 同条/引用图扩展 → 原文核验 → 证据加权
    read_article: 整条原文定位 → 核验及加权
    ask_user: 关键事实缺口 → 保存并暂停
    finish: 逐争点完成检查 → 选证并保留配套来源
        → 完整提示 ≤1000 tokens 用 Qwen，超出用 DeepSeek
        → 引用编号检查 → 答案 / 来源 / 争点 / 图 / 轨迹
```

当前 Agent 对一般法律问题启用争点计划，会调用 DeepSeek；单轮 `LegalRAGPipeline.answer()` 保留简单语义原型直接检索分支。本地生成不代表整条链路离线。纯问候短路；纯法条查阅保留目标；复杂案件中的引用不跳过规划。

默认两路各取 50 条，RRF 权重为向量 0.7、BM25 0.3、K=60，融合 20 条并精排保留 8 条，再做预算为 8 的图扩展。每争点最多 3 条主证据，同时保留同条与配套原文。FAISS 在 CPU；编码与重排可在 GPU。

核验记录 supports/contradicts/conditional/irrelevant，检查逐字法条、法定条件、用户事实及配套引用。缺失事实保持未知；更正事实后复核本案证据。同一来源按争点分别判断，不把模型计划当成事实。

静态图是条文成员与引用关系；案件图包含命题、证据、条件、例外及联合证据超边。旧 `graph/graph.py` 相似构图函数独立保留，不参与当前距离计算。`retrieval/practice.py` 是独立 GPU kernel 练习，不是业务加速依据。

证据加权借鉴 CHAIN，使用法律依赖距离、给定日期下的适用性与来源/依赖路径去重，保留权重分解。它不是因果效应或答案正确概率。未知时效明确保留未知；法律不会仅因发布时间较早而降权。

SupportCalibrator 只读取外部 Platt 参数，默认是 uncalibrated_support_score，没有拟合或校准收益结果。逐字核验不单独证明语义判断正确，引用编号检查也没有逐句验证答案蕴含。

默认每轮最多 6 次业务操作、10 次决策，重复查询不叠加支持。SQLite 保存快照和 revision，避免过时写入覆盖；续办处理既定争点。轨迹 token 用量不是完整费用统计。API 对共享模型串行调用。

可选 Triton Ensemble 编排 BM25、CPU FAISS 和 RRF，没有托管 Qwen、BGE 编码器或 Reranker，未做本次部署验收。默认依赖锁不包含 Triton 客户端。

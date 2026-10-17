# 数据与模型资源

`graph/Chinese-laws` 包含项目原有的 177 份法律文本，供结构化、检索和引用定位使用。材料没有完整来源 URL 和有效期信息，不构成已确认的现行法律汇编，也不增加许可声明。

`python -m law_rag.corpus` 按条文生成稳定 ID、原文区间和显式引用关系。`python graph/embedding.py` 用固定 BGE 版本生成向量及 manifest。prepared 产物本地生成，不提交 Git。

当前静态统计为 14,620 个片段、14,415 个条文快照、913 条可解析引用边，另有 6 条未解决引用。语料更新后需重建结构化数据与向量；这些不是准确率指标。

已有 `legal_counsel_v2.json`（8,000 条）和 `final_triplets.json`（2,469 条）留在本地，公开仓库不分发原始问答或训练数据。自己的有授权数据可放入 `graph/`：问答字段为 instruction/input/output，三元组字段为 query/positive/hard_negative。

`graph/triple.py` 保留 BGE 候选召回、GPU 批量匹配、异步 DeepSeek 正负样本判别流程；它会产生模型费用，不在启动或 CI 时自动运行。

`EvidenceStore.link_dataset()` 通过精确原文或显式引用添加证据/条文 ID 和依赖边。当前本地三元组 2,467/2,469 条至少有一个字段关联来源；问答仅 1/8,000 条；问题/输入字段均为 0。关联不是核验金标，不能宣传完成全部 QA 的语义图融合。

原有 MNRL 对比学习脚本保存在本机旧版私有归档中。本次没有重新训练、拟合校准参数或发布微调权重；在线默认使用基础 BGE。接入已有领域模型时，建库与查询必须使用相同模型并重建索引。

本地模型仓库与 revision 见 `models.lock.json`。`python tools/download_models.py` 显式下载固定版本模型，权重与缓存均写 D 盘，不进入 Git。云端 DeepSeek 是可配置的服务商别名，不是不可变权重快照。

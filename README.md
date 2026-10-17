# 律据 · 四模型协同的法律证据 Agent

[![CI](https://github.com/hbk520520/law_rag/actions/workflows/ci.yml/badge.svg)](https://github.com/hbk520520/law_rag/actions/workflows/ci.yml)

基于现有法律 RAG，加入按争点检索、原文核验、证据反馈补查和多轮事实追问。保留原有混合检索、独立重排、本地/云端级联生成；使用原生 Function Calling，不依赖额外 Agent 框架。

## 四个模型的分工

| 模型 | 作用 |
|---|---|
| **BGE-base-zh-v1.5** | 768 维法条/查询编码、原型路由、离线候选挖掘 |
| **BGE-reranker-base** | CrossEncoder 问题—法条精排 |
| **Qwen2.5-1.5B-Instruct** | 本地短提示答案生成，tokenizer 计算完整提示长度 |
| **DeepSeek** | 查询理解、争点规划、核验、工具决策、长提示生成及离线样本判别 |

本地模型固定仓库 revision，Qwen 同时校验已有权重/配置/tokenizer 哈希。云端默认别名 `deepseek-flash` 可由 `LEGAL_API_MODEL` 覆盖，不是固定权重快照。

## 执行流程

```text
用户案情 → 争点规划 → Agent 工具选择
    ├─ 检索：BM25 + BGE/FAISS → 加权 RRF → BGE-Reranker
    │        → 引用图扩展 → 原文/适用条件核验 → 去重加权
    ├─ 法条全文读取 → 核验
    ├─ 事实不足 → 追问、保存并暂停
    └─ 逐争点完成检查 → 选证 → Qwen / DeepSeek 生成 → 引用检查
```

支持 `search_evidence`、`read_article`、`ask_user`、`finish`，默认每轮最多 6 次业务操作、10 次决策。同源证据去重、按争点分别核验；更正事实后复核本案已收集证据。SQLite 支持案件恢复，页面展示来源、争点与行动轨迹。

## 本机安装与准备

正式运行锁针对 **Windows x64 / CPython 3.13 / PyTorch 2.10.0 + CUDA 12.8**。先把项目放在 D 盘；以下命令在项目根目录执行。存储脚本只设置当前终端，下载、pip/Hugging Face 缓存和临时文件均落 D 盘。

```powershell
. ./tools/use-d-storage.ps1
python -m venv .venv-agent
.\.venv-agent\Scripts\python.exe -m pip install --require-hashes --no-deps -r requirements-bootstrap.lock
.\.venv-agent\Scripts\python.exe -m pip install --no-build-isolation --no-deps -r requirements.txt
.\.venv-agent\Scripts\python.exe -m pip check

# 显式下载固定版本模型；已有模型会复用，本地 Qwen 权重约 3.09 GB
.\.venv-agent\Scripts\python.exe tools/download_models.py
.\.venv-agent\Scripts\python.exe -m law_rag.corpus
.\.venv-agent\Scripts\python.exe graph/embedding.py
```

原有 177 份法律文本在 `graph/Chinese-laws`；索引本地生成。权重、原始问答、训练三元组、案件库、缓存、备份和个人报告不提交 Git。来源与授权边界见 [数据说明](docs/data.md)。项目代码尚未指定额外开源许可；模型与语料遵守各自许可。

## 启动与续办

```powershell
$env:DEEPSEEK_API_KEY = "你的有效密钥"
$env:LEGAL_DEVICE = "cuda"
.\.venv-agent\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 6006
```

打开 http://127.0.0.1:6006，接口文档在 `/docs`。`.env.example` 是配置说明，程序不自动加载 `.env`。一般法律问题会调用云端争点规划；本地 Qwen 生成不代表整条链路离线。

`POST /chat` 新建案件：

```json
{"query":"公司欠薪且未签合同，分别如何处理？","case_date":"2026-10-01"}
```

返回 `case_id`、`status`、`answer`、`issues`、`docs`、`evidence_graph` 和 `trace`。需要补充事实时，携带同一个 `case_id` 再发 `query`；`GET /cases/{case_id}` 恢复案件。续办针对既定争点，新诉求请新建案件。

```powershell
.\.venv-agent\Scripts\python.exe main.py "法律问题" --json
.\.venv-agent\Scripts\python.exe main.py "补充事实" --case-id "案件编号" --json
.\.venv-agent\Scripts\python.exe main.py --show-case "案件编号"
```

## 开发与检查

```powershell
git config core.hooksPath .githooks
./tools/check.ps1
```

提交检查阻止常见凭证、权重/运行数据、LFS 指针、无效语法和文本格式问题。现有软件回归不调用真实模型或云端，不等于法律任务效果评测。GitHub CI 使用相同正式版本的 Windows CPU 依赖锁，不下载模型。

- [架构、证据权重与能力边界](docs/architecture.md)
- [数据与离线构建流程](docs/data.md)
- [验证记录](docs/validation.md)
- [贡献流程](CONTRIBUTING.md) · [安全说明](SECURITY.md) · [回滚](ROLLBACK.md)

当前没有训练或效果实验结果；证据权重不是正确概率或胜诉概率，法条有效期尚未完整核实。可选 Triton 检索配置保留，但未包含在本次部署验收中。服务默认面向本机演示，公网部署需另做鉴权与隔离。

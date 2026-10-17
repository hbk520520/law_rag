from pathlib import Path
import sys
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from law_rag.config import API_KEY, API_BASE, API_MODEL, API_TIMEOUT, DEVICE, EMBEDDING_MODEL, EMBEDDING_REVISION, MODEL_CACHE, DATA_DIR, PREPARED_DIR
from law_rag.corpus import EvidenceStore

import json
import asyncio
import torch
from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm # 用于异步的高级进度条

# =======================================================
# 1. 基础配置
# =======================================================
# 【优化点三】换用 AsyncOpenAI，开启异步协程模式
aclient = None

LAWS_DIR = DATA_DIR / "Chinese-laws"
QA_FILE = DATA_DIR / "legal_counsel_v2.json"
OUTPUT_FILE = DATA_DIR / "final_triplets_verified.jsonl"

# 设置并发安全锁 (控制向 DeepSeek 发送请求的并发量，防止 429 报错)
CONCURRENCY_LIMIT = 20
sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

# Models and clients are initialized only when this data-generation command is run.
# 强制将模型加载到 cuda (GPU) 上
model = None

# =======================================================
# 2. Prompt 定义
# =======================================================

JUDGE_NEGATIVE_PROMPT = """
你是一位严谨的法官。请评估【候选法条】是否能作为对比学习的“高质量困难负样本(Hard Negative)”。
拒绝条件：它其实也能回答问题，或者它和问题毫无关系。
通过条件：它和正样本字面相似，但因为适用条件不同，在当前情境下绝对不能适用。
输出 JSON: {"reason": "...", "is_valid_hard_negative": true/false}
"""

# =======================================================
# 3. 异步大模型调用函数 (Asyncio 并发)
# =======================================================
async def get_positive_law_async(query: str, answer: str, indices):
    async with sem:
        # Propose IDs from actual retrieved text; never ask the LLM to recreate statutory wording.
        candidates = [{"evidence_id": evidence_store.nodes[i]["evidence_id"],
                       "text": positive_texts[i]} for i in indices]
        response = await aclient.chat.completions.create(
            model=API_MODEL, temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content":
                       '选择能在问题案情下支持参考回答的一个候选法条，检查适用条件和例外。参考回答可能错误。'
                       '禁止生成法条。仅输出JSON: {"evidence_id": "候选ID或null", "quote":"逐字候选原文", "reason":"理由"}。'},
                      {"role": "user", "content": json.dumps({"query": query, "answer": answer, "candidates": candidates}, ensure_ascii=False)}])
        result = json.loads(response.choices[0].message.content)
        eid = result.get("evidence_id")
        if eid is None:
            return None
        allowed = {item["evidence_id"] for item in candidates}
        if eid not in allowed:
            raise ValueError("Positive selector returned an unknown evidence ID")
        text = evidence_store.by_id[eid]["search_text"]
        quote = result.get("quote")
        if not isinstance(quote, str) or not quote.strip() or quote not in text:
            raise ValueError("Positive selector quotation is absent from its source")
        return {"text": text, "evidence_id": eid, "quote": quote, "reason": result.get("reason", "")}


async def verify_negative_with_llm_async(query: str, positive: str, candidate: str):
    async with sem:
        response = await aclient.chat.completions.create(
            model=API_MODEL, messages=[{"role": "system", "content": JUDGE_NEGATIVE_PROMPT},
                                      {"role": "user", "content": f"Query: {query}\nPositive: {positive}\nCandidate: {candidate}"}],
            response_format={"type": "json_object"}, temperature=0)
        result = json.loads(response.choices[0].message.content)
        if not isinstance(result, dict) or type(result.get("is_valid_hard_negative")) is not bool:
            raise ValueError("Negative judge returned an invalid result")
        return result["is_valid_hard_negative"]


# =======================================================
# 4. PyTorch 显存建库模块 (替换掉 Numpy)
# =======================================================
def build_law_index_gpu(laws_dir):
    global evidence_store
    evidence_store = EvidenceStore(PREPARED_DIR)
    texts = [node["search_text"] for node in evidence_store.nodes]
    if not texts:
        raise ValueError("Prepared evidence corpus is empty")
    matrix = model.encode(texts, show_progress_bar=True, normalize_embeddings=True, convert_to_tensor=True)
    return texts, matrix


# =======================================================
# 5. 三段式流水线 (终极架构)
# =======================================================
async def process_alpaca_data_pipeline(law_texts, law_matrix_gpu):
    # ---------------------------------------------------
    # 准备数据
    # ---------------------------------------------------
    with open(QA_FILE, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    parsed_data = []
    for item in qa_data:
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text = item.get("output", "")
        query = f"{instruction}\n补充信息：{input_text}" if input_text else instruction
        parsed_data.append({"query": query, "answer": output_text})

    print(f"\n🚀 [阶段一] 启动 Asyncio 异步网络集群，提取正样本...")

    # 构建所有提取正样本的异步任务
    candidate_rows = []
    for offset in range(0, len(parsed_data), 128):
        batch = [d["query"] for d in parsed_data[offset:offset + 128]]
        encoded = model.encode(batch, batch_size=128, normalize_embeddings=True, convert_to_tensor=True)
        scores = torch.matmul(encoded, law_matrix_gpu.T)
        candidate_rows.extend(torch.topk(scores, k=min(20, len(law_texts)), dim=1).indices.cpu().tolist())
    tasks_stage_1 = [get_positive_law_async(d["query"], d["answer"], indices)
                     for d, indices in zip(parsed_data, candidate_rows)]
    # 并发执行并显示进度条
    positive_laws = await tqdm.gather(*tasks_stage_1, desc="阶段一：溯源正样本")

    # 过滤掉没有找到法条的数据
    valid_items = []
    for i, p_law in enumerate(positive_laws):
        if p_law:
            valid_items.append({"query": parsed_data[i]["query"], "positive": p_law["text"],
                                "positive_evidence_id": p_law["evidence_id"], "positive_verification": p_law})

    print(f"阶段一完成！成功找到正样本法条的数据量：{len(valid_items)}")

    if not valid_items:
        print("没有找到任何正样本，程序退出。")
        return []

    # ---------------------------------------------------
    # 【优化点一】批处理计算 (Batching) + GPU 矩阵乘法
    # ---------------------------------------------------
    print(f"\n🔥 [阶段二] GPU 满载运作：批处理查询与显存级相似度匹配...")
    queries = [item["query"] for item in valid_items]

    # 彻底告别 Batch Size = 1！一次性把所有 Query 塞进 GPU，batch_size 开到 128
    query_tensors = model.encode(queries, batch_size=128, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True)

    # 瞬间完成几千乘几万的浮点矩阵乘法 (Tensor Core 发力)
    print("正在 GPU 显存内执行全量点积与 Top-K 排序...")
    scores = torch.matmul(query_tensors, law_matrix_gpu.T)
    # 直接在 GPU 上取出每个 Query 对应的 Top 10 最相似法条索引
    top_k_indices = torch.topk(scores, k=min(10, len(law_texts)), dim=1).indices.cpu().numpy()

    # 将候选法条挂载回字典中
    for i, item in enumerate(valid_items):
        item["candidates"] = [law_texts[idx] for idx in top_k_indices[i]]

    # ---------------------------------------------------
    # 阶段三：异步验证困难负样本
    # ---------------------------------------------------
    print(f"\n🚀 [阶段三] 再次唤醒 Asyncio，裁判验证高质量负样本...")

    async def find_hard_negative(item):
        for candidate in item["candidates"]:
            if item["positive"] in candidate or candidate in item["positive"]:
                continue
            # 发起并发 API 调用
            is_valid = await verify_negative_with_llm_async(item["query"], item["positive"], candidate)
            if is_valid:
                item["hard_negative"] = candidate
                return item # 只要找到一个就立刻返回
        return None

    tasks_stage_3 = [find_hard_negative(item) for item in valid_items]
    results_stage_3 = await tqdm.gather(*tasks_stage_3, desc="阶段三：挖掘困难负样本")

    # 过滤掉返回为 None 的数据
    final_triplets = [res for res in results_stage_3 if res is not None]
    return final_triplets

# =======================================================
# 6. 执行入口
# =======================================================
async def main():
    global model, aclient, positive_texts
    if not API_KEY:
        raise ValueError("Set DEEPSEEK_API_KEY before generating source-linked triplets")
    model = SentenceTransformer(EMBEDDING_MODEL, revision=EMBEDDING_REVISION,
                                cache_folder=str(MODEL_CACHE), device=DEVICE)
    async with AsyncOpenAI(api_key=API_KEY, base_url=API_BASE, timeout=API_TIMEOUT, max_retries=1) as client:
        aclient = client
        laws, matrix = build_law_index_gpu(LAWS_DIR)
        positive_texts = laws
        triplets = await process_alpaca_data_pipeline(laws, matrix)
    ids_by_text = {}
    for node in evidence_store.nodes:
        ids_by_text.setdefault(node["search_text"], []).append(node["evidence_id"])
    with open(OUTPUT_FILE, "w", encoding="utf-8") as handle:
        for row in triplets:
            negative_ids = ids_by_text[row["hard_negative"]]
            clean = {"query": row["query"], "positive": row["positive"], "hard_negative": row["hard_negative"],
                     "positive_evidence_id": row["positive_evidence_id"], "hard_negative_evidence_ids": negative_ids,
                     "positive_verification": row["positive_verification"], "graph_file": "prepared/evidence_graph.json"}
            handle.write(json.dumps(clean, ensure_ascii=False) + "\n")
    print(f"Saved {len(triplets)} source-linked triplets to {OUTPUT_FILE}")


if __name__ == "__main__":
    # Windows/Linux 兼容的 Asyncio 启动方式
    asyncio.run(main())

import numpy as np
import faiss
import igraph as ig
import logging

logger = logging.getLogger("GraphBatchBuilder")

def build_initial_graph_batch(self, nodes_data: list, embeddings: np.ndarray, search_batch_size: int = 10000):
    """
    匹配算子：将离线的结构化节点和向量矩阵，全量批量注入图系统
    """
    total_nodes = len(nodes_data)
    if total_nodes != embeddings.shape[0]:
        raise ValueError("节点数量与矩阵维度不匹配！")

    logger.info(f"开始批量构建底层图谱，总节点数: {total_nodes}")

    # 1. 批量写入节点元数据到 igraph
    self.graph.add_vertices(total_nodes)
    self.graph.vs["name"] = [n["id"] for n in nodes_data]     # 设置唯一标识
    self.graph.vs["content"] = [n["content"] for n in nodes_data]
    self.graph.vs["type"] = [n["type"] for n in nodes_data]
    self.graph.vs["metadata"] = [n["metadata"] for n in nodes_data]

    # 2. 批量将向量注入 FAISS (极速完成)
    all_ids = np.array([n["id"] for n in nodes_data], dtype=np.int64)
    self.index.add_with_ids(embeddings.astype(np.float32), all_ids)
    logger.info("FAISS 索引全量注入完成。")

    # 3. 分块并行检索，建立稠密边 (防止内存打满)
    all_edges = []
    all_weights = []

    logger.info("开始进行拓扑连边推演...")
    # 切片分批进行 FAISS Search，防止巨型结果矩阵占满系统 RAM
    for i in range(0, total_nodes, search_batch_size):
        end_idx = min(i + search_batch_size, total_nodes)
        batch_emb = embeddings[i:end_idx].astype(np.float32)
        batch_ids = all_ids[i:end_idx]

        # 检索 Top-K (包含自己)
        sims, n_ids = self.index.search(batch_emb, self.top_k)

        # 矩阵遍历过滤
        for row_idx, query_id in enumerate(batch_ids):
            for sim, target_id in zip(sims[row_idx], n_ids[row_idx]):
                # 触发阈值 0.85，拦截重复阈值 0.98，跳过自己，跳过无效节点
                if self.threshold <= sim < self.dedup_threshold and target_id != query_id and target_id != -1:
                    # igraph 要求传入内部顶点序号，而不是 name，需要映射一下，此处简化处理直接记录 ID
                    # 在实际 igraph 中，最好转换为内部连边
                    all_edges.append((query_id, int(target_id)))
                    all_weights.append(float(sim))

    # 4. 批量写入边 (igraph 的极致性能体现)
    logger.info(f"推演结束，共计算出 {len(all_edges)} 条合法边，准备批量落图。")
    # 为了防止无向图重复连边 (A->B, B->A)，采用 set 去重
    unique_edges = {}
    for edge, weight in zip(all_edges, all_weights):
        # 排序确保 (1, 2) 和 (2, 1) 被视为同一个 key
        sorted_edge = tuple(sorted(edge))
        if sorted_edge not in unique_edges:
            unique_edges[sorted_edge] = weight

    edges_list = list(unique_edges.keys())
    weights_list = list(unique_edges.values())

    # igraph 批量加边
    # 这里需要将 node_name(id) 映射为 igraph 的内部索引
    name_to_index = {v["name"]: v.index for v in self.graph.vs}
    igraph_edges = [(name_to_index[e[0]], name_to_index[e[1]]) for e in edges_list]

    self.graph.add_edges(igraph_edges)
    self.graph.es["weight"] = weights_list
    self.graph.es["relation"] = ["similar_to"] * len(weights_list)

    logger.info("初始稠密图谱装配彻底完毕！")

# ==========================================
# 外部调用入口：将两部分打通
# ==========================================
# import json, numpy as np
#
# with open("corpus_nodes.json", "r") as f:
#     nodes_data = json.load(f)
# embeddings = np.load("corpus_embeddings.npy")
#
# builder = LegalDenseGraphBuilder(...)
# builder.build_initial_graph_batch(nodes_data, embeddings)

def load_directed_evidence_graph(directory=None):
    """Expose the source-linked citation graph through the existing igraph dependency."""
    from law_rag.config import PREPARED_DIR
    from law_rag.corpus import EvidenceStore
    store = EvidenceStore(directory or PREPARED_DIR)
    graph = ig.Graph(directed=True)
    names = list(store.by_id) + ["article:" + aid for aid in store.graph["article_members"]]
    graph.add_vertices(names)
    edges, relations = [], []
    for aid, members in store.graph["article_members"].items():
        for eid in members:
            edges.append(("article:" + aid, eid))
            relations.append("contains")
    for edge in store.graph["edges"]:
        edges.append((edge["source_id"], "article:" + edge["target_article_id"]))
        relations.append("references")
    graph.add_edges(edges)
    graph.es["relation"] = relations
    return graph

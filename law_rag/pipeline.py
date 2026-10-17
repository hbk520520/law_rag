"""Existing legal RAG operations, shared by the single-pass and Agent entry points."""
from .config import CONTEXT_K, DEVICE, GRAPH_BUDGET, RECALL_K, RERANKER_MODEL, RERANKER_REVISION, MODEL_CACHE, RETRIEVAL_BACKEND, VERIFY_K, MAX_DEPENDENCY_HOPS
from .evidence import EvidenceVerifier, question_graph, select_evidence
from .weighting import EvidenceWeighter, parse_date


class LegalRAGPipeline:
    def __init__(self):
        from sentence_transformers import CrossEncoder
        from query import LegalQueryRouter
        from retriever import HybridRetriever, TritonRetriever
        from generate import RAGGenerator
        if RETRIEVAL_BACKEND not in {"local", "triton"}:
            raise ValueError("LEGAL_RETRIEVAL_BACKEND must be local or triton")
        if min(CONTEXT_K, RECALL_K, VERIFY_K) < 1 or GRAPH_BUDGET < 0:
            raise ValueError("Invalid retrieval/context limits")
        self.verifier = EvidenceVerifier()
        self.router = LegalQueryRouter()
        self.retriever = HybridRetriever() if RETRIEVAL_BACKEND == "local" else TritonRetriever()
        self.weighter = EvidenceWeighter(self.retriever.store)
        self.reranker = CrossEncoder(RERANKER_MODEL, revision=RERANKER_REVISION,
                                    cache_folder=str(MODEL_CACHE), device=DEVICE)
        self.generator = RAGGenerator(max_local_tokens=1000)

    def search(self, search_query, *, vector=None, rerank_query=None):
        if vector is None:
            vector = self.router.embedder.encode(search_query, normalize_embeddings=True)
        docs = self.retriever.retrieve(search_query, vector, RECALL_K)
        if docs:
            scores = self.reranker.predict([[rerank_query or search_query, d["search_text"]] for d in docs])
            docs = sorted([{**d, "rerank_score": float(score)} for d, score in zip(docs, scores)],
                          key=lambda d: d["rerank_score"], reverse=True)[:VERIFY_K]
            docs = self.retriever.store.expand(docs, GRAPH_BUDGET, MAX_DEPENDENCY_HOPS)
        return docs

    def read_article(self, reference):
        store = self.retriever.store
        ids = store.find_articles(reference)
        candidates = [{"article_id": aid, "article_label": store.by_id[store.graph["article_members"][aid][0]]["article_label"]}
                      for aid in ids]
        return {"documents": store.read_article(ids[0]) if len(ids) == 1 else [], "candidates": candidates,
                "lookup_status": "found" if len(ids) == 1 else "ambiguous" if ids else "not_found"}

    def verify_evidence(self, query, docs, case_date=None, *, claim=None, user_turns=None):
        day = parse_date(case_date, "case_date")
        date_text = day.isoformat() if day else None
        docs = self.weighter.metadata.apply(docs)
        verified = self.verifier.verify(query, docs, case_date=date_text, claim=claim, user_turns=user_turns)
        present = {}
        for doc in docs:
            present[doc["article_id"]] = present.get(doc["article_id"], 0) + 1
        for doc in verified:
            verdict = doc["verification"]
            if verdict["relation"] != "irrelevant" and present[doc["article_id"]] < doc["metadata"]["part_count"]:
                verdict["relation"] = "conditional"
                message = "本条其余原文片段未完整提供，尚不能排除其他适用条件或例外。"
                verdict["missing_facts"].append(message)
                verdict.setdefault("gaps", []).append({"kind": "article", "description": message,
                                                      "reference": doc["article_id"]})
        return self.weighter.apply(verified, day)

    def answer(self, query, case_date=None):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be empty")
        day = parse_date(case_date, "case_date")
        date_text = day.isoformat() if day else None
        route = self.router.process(query.strip())
        if route["route"] == "chitchat":
            return {"answer": route["response"], "route": "chitchat", "docs": [],
                    "case_date": date_text, "evidence_graph": {"schema_version": 2, "nodes": [], "edges": [], "hyperedges": []}}
        docs = self.search(route["search_query"], vector=route["vector"], rerank_query=query)
        weighted = self.verify_evidence(query, docs, day)
        selected = select_evidence(weighted, CONTEXT_K)
        answer = self.generator.generate(query, route, selected) if selected else "现有资料中没有经核验可用于回答该问题的依据，请补充相关事实或法条信息。"
        return {"answer": answer, "route": route["route"], "search_query": route["search_query"],
                "case_date": date_text, "docs": selected,
                "evidence_graph": question_graph(weighted, self.retriever.store.graph),
                "evidence_weight_method": "legal_dependency_validity_recurrence_v1",
                "relation_support_method": self.verifier.calibrator.method}

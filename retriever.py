"""Existing BM25 + FAISS + weighted RRF, now retaining evidence provenance."""
import json
import numpy as np

from law_rag.config import EMBEDDING_MODEL, EMBEDDING_REVISION, PREPARED_DIR, TRITON_URL
from law_rag.corpus import EvidenceStore, digest


class HybridRetriever:
    def __init__(self, directory=PREPARED_DIR):
        import faiss
        import jieba
        from rank_bm25 import BM25Okapi
        from pathlib import Path
        directory = Path(directory)
        self.store = EvidenceStore(directory)
        self.corpus_texts = [n["search_text"] for n in self.store.nodes]
        manifest = json.loads((directory / "embedding_manifest.json").read_text(encoding="utf-8"))
        if manifest["embedding_model"] != EMBEDDING_MODEL:
            raise ValueError("Index embedding model differs from query model; rebuild using graph/embedding.py")
        if manifest.get("embedding_revision") != EMBEDDING_REVISION:
            raise ValueError("Index and query encoder revisions differ; rebuild using graph/embedding.py")
        if manifest["texts_sha256"] != digest(json.dumps(self.corpus_texts, ensure_ascii=False)):
            raise ValueError("Corpus changed after encoding; rebuild embeddings before retrieval")
        matrix = np.load(directory / "corpus_embeddings.npy", allow_pickle=False)
        if matrix.ndim != 2 or matrix.shape[0] != len(self.store.nodes) or not np.isfinite(matrix).all():
            raise ValueError("Invalid or misaligned corpus embeddings")
        if matrix.shape[1] != manifest["dimensions"]:
            raise ValueError("Embedding manifest dimension mismatch")
        self.index = faiss.IndexFlatIP(matrix.shape[1])
        self.index.add(np.ascontiguousarray(matrix, dtype=np.float32))
        self._cut = jieba.cut
        self.bm25 = BM25Okapi([list(self._cut(text)) for text in self.corpus_texts])
        self.RRF_K, self.WEIGHT_VECTOR, self.WEIGHT_BM25 = 60, 0.7, 0.3

    def retrieve(self, query_text, query_vector, top_k=20):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        vector = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        if vector.shape[1] != self.index.d or not np.isfinite(vector).all():
            raise ValueError("Invalid query embedding")
        count = min(50, len(self.store.nodes))
        bm25_ranks = np.argsort(self.bm25.get_scores(list(self._cut(query_text))))[::-1][:count]
        _, neighbors = self.index.search(vector, count)
        scores, ranks = {}, {}
        for kind, weight, indices in [("vec_rank", self.WEIGHT_VECTOR, neighbors[0]),
                                      ("bm25_rank", self.WEIGHT_BM25, bm25_ranks)]:
            for rank, idx in enumerate(indices, 1):
                idx = int(idx)
                if not 0 <= idx < len(self.store.nodes):
                    continue
                scores[idx] = scores.get(idx, 0.0) + weight / (self.RRF_K + rank)
                ranks.setdefault(idx, {})[kind] = rank
        return [{**self.store.nodes[idx], "rrf_score": score, **ranks[idx], "retrieval_origin": "hybrid"}
                for idx, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]]

    def search(self, query_text, query_vector, top_k=20):
        # Retain the original text-only convenience API for existing callers.
        return [doc["search_text"] for doc in self.retrieve(query_text, query_vector, top_k)]


class TritonRetriever:
    def __init__(self, directory=PREPARED_DIR):
        import tritonclient.http as httpclient
        self.http = httpclient
        self.client = httpclient.InferenceServerClient(url=TRITON_URL)
        self.store = EvidenceStore(directory)

    def retrieve(self, query_text, query_vector, top_k=20):
        text = np.array([[query_text]], dtype=object)
        vector = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        inputs = [self.http.InferInput("QUERY_TEXT", text.shape, "BYTES"),
                  self.http.InferInput("QUERY_VECTOR", vector.shape, "FP32")]
        inputs[0].set_data_from_numpy(text)
        inputs[1].set_data_from_numpy(vector)
        result = self.client.infer("legal_ensemble", inputs=inputs)
        ids = result.as_numpy("FINAL_TOP_IDS")[0]
        docs = []
        for raw in ids[:top_k]:
            eid = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
            if eid not in self.store.by_id:
                raise ValueError("Triton and local evidence corpus differ; publish the same prepared directory to both")
            docs.append({**self.store.by_id[eid], "retrieval_origin": "triton"})
        return docs

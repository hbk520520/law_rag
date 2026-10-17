import json
import os
from pathlib import Path
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        root = Path(__file__).resolve().parents[3]
        directory = Path(os.environ.get("LEGAL_PREPARED_DIR", root / "graph" / "prepared" / "agent-index-v1"))
        self.nodes = json.loads((directory / "corpus_nodes.json").read_text(encoding="utf-8"))
        self.top_k = min(int(os.environ.get("LEGAL_RECALL_K", "20")), len(self.nodes))
        if self.top_k < 1:
            raise ValueError("Recall limit and corpus must be nonempty")

    def execute(self, requests):
        responses = []
        for request in requests:
            vectors = pb_utils.get_input_tensor_by_name(request, "VECTOR_RANKS_IN").as_numpy()
            sparse = pb_utils.get_input_tensor_by_name(request, "BM25_RANKS_IN").as_numpy()
            if vectors.shape != sparse.shape:
                raise ValueError("Rank batch shapes differ")
            texts, ids = [], []
            for vranks, branks in zip(vectors, sparse):
                scores = {}
                for ranks, weight in [(vranks, 0.7), (branks, 0.3)]:
                    for rank, idx in enumerate(ranks, 1):
                        idx = int(idx)
                        if 0 <= idx < len(self.nodes):
                            scores[idx] = scores.get(idx, 0.0) + weight / (60 + rank)
                indices = [idx for idx, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:self.top_k]]
                # Every row must have the same output width, even for overlapping short lists.
                if len(indices) != self.top_k:
                    raise ValueError("Insufficient valid retrieved IDs")
                texts.append([self.nodes[idx]["search_text"].encode("utf-8") for idx in indices])
                ids.append([self.nodes[idx]["evidence_id"].encode("utf-8") for idx in indices])
            responses.append(pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("FINAL_RESULTS", np.array(texts, dtype=object)),
                pb_utils.Tensor("FINAL_IDS", np.array(ids, dtype=object))]))
        return responses

import json
import os
from pathlib import Path
import numpy as np
import jieba
from rank_bm25 import BM25Okapi
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        root = Path(__file__).resolve().parents[3]
        directory = Path(os.environ.get("LEGAL_PREPARED_DIR", root / "graph" / "prepared" / "agent-index-v1"))
        texts = json.loads((directory / "corpus_texts.json").read_text(encoding="utf-8"))
        if not texts:
            raise ValueError("Corpus is empty")
        self.bm25 = BM25Okapi([list(jieba.cut(text)) for text in texts])

    def execute(self, requests):
        responses = []
        for request in requests:
            queries = pb_utils.get_input_tensor_by_name(request, "TEXT_IN").as_numpy()
            ranks = np.full((len(queries), 50), -1, dtype=np.int32)
            for i, row in enumerate(queries):
                raw = row[0]
                text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                scores = self.bm25.get_scores(list(jieba.cut(text)))
                indices = np.argsort(scores)[::-1][:50]
                ranks[i, :len(indices)] = indices
            tensor = pb_utils.Tensor("BM25_RANKS_OUT", ranks)
            responses.append(pb_utils.InferenceResponse(output_tensors=[tensor]))
        return responses

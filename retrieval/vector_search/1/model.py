import os
from pathlib import Path
import faiss
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        # Keep Triton independent of the web application's Python import path.
        root = Path(__file__).resolve().parents[3]
        directory = Path(os.environ.get("LEGAL_PREPARED_DIR", root / "graph" / "prepared" / "agent-index-v1"))
        matrix = np.load(directory / "corpus_embeddings.npy", allow_pickle=False)
        if matrix.ndim != 2 or not len(matrix) or not np.isfinite(matrix).all():
            raise ValueError("Invalid corpus embeddings")
        self.index = faiss.IndexFlatIP(matrix.shape[1])
        self.index.add(np.ascontiguousarray(matrix, dtype=np.float32))

    def execute(self, requests):
        responses = []
        for request in requests:
            vector = pb_utils.get_input_tensor_by_name(request, "VECTOR_IN").as_numpy()
            if vector.ndim != 2 or vector.shape[1] != self.index.d or not np.isfinite(vector).all():
                raise ValueError("Invalid query vectors")
            _, ids = self.index.search(np.ascontiguousarray(vector, dtype=np.float32), min(50, self.index.ntotal))
            # Fixed-width rank tensors use -1 exclusively as padding.
            padded = np.full((len(vector), 50), -1, dtype=np.int32)
            padded[:, :ids.shape[1]] = ids
            tensor = pb_utils.Tensor("VECTOR_RANKS_OUT", padded)
            responses.append(pb_utils.InferenceResponse(output_tensors=[tensor]))
        return responses

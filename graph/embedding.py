"""Prepare source-linked corpus and encode it using the existing BGE pipeline."""
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from law_rag.config import DEVICE, EMBEDDING_MODEL, EMBEDDING_REVISION, MODEL_CACHE, PREPARED_DIR
from law_rag.corpus import digest


def main():
    import numpy as np
    from sentence_transformers import SentenceTransformer
    nodes = json.loads((PREPARED_DIR / "corpus_nodes.json").read_text(encoding="utf-8"))
    texts = [node["search_text"] for node in nodes]
    if not texts:
        raise ValueError("Prepare the corpus using python -m law_rag.corpus first")
    model = SentenceTransformer(EMBEDDING_MODEL, revision=EMBEDDING_REVISION,
                               cache_folder=str(MODEL_CACHE), device=DEVICE)
    # Respect the encoder's actual positional limit; do not force BERT to 1024 tokens.
    for start in range(0, len(texts), 256):
        token_ids = model.tokenizer(texts[start:start + 256], truncation=False, add_special_tokens=True)["input_ids"]
        if any(len(ids) > model.max_seq_length for ids in token_ids):
            raise ValueError("Evidence exceeds this encoder's token limit. Reprepare with a smaller --chunk-size; no evidence is silently truncated.")
    embeddings = model.encode(texts, batch_size=64, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=True)
    np.save(PREPARED_DIR / "corpus_embeddings.npy", embeddings)
    manifest = {"embedding_model": EMBEDDING_MODEL, "embedding_revision": EMBEDDING_REVISION,
                "dimensions": int(embeddings.shape[1]),
                "rows": len(nodes), "texts_sha256": digest(json.dumps(texts, ensure_ascii=False))}
    (PREPARED_DIR / "embedding_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Encoded {len(nodes)} source-linked chunks; no training.")


if __name__ == "__main__":
    main()

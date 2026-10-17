"""Download explicitly pinned model snapshots to project storage on D:."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["all", "embedding", "reranker", "local_generator"], default="all")
    args = parser.parse_args()
    if ROOT.drive.upper() != "D:":
        parser.error("Place the local project on D: before downloading models.")
    storage = ROOT / "output" / "local"
    for name, directory in {"HF_HOME": storage / "huggingface", "TORCH_HOME": storage / "torch", "TEMP": storage / "tmp", "TMP": storage / "tmp"}.items():
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(directory)
    from huggingface_hub import snapshot_download
    lock = json.loads((ROOT / "models.lock.json").read_text(encoding="utf-8"))
    targets = ["embedding", "reranker", "local_generator"] if args.model == "all" else [args.model]
    for name in targets:
        entry = lock[name]
        options = {"repo_id": entry["repository"], "revision": entry["revision"]}
        if name == "local_generator":
            options.update(local_dir=str(ROOT / "generate" / "model"), allow_patterns=entry["files"])
        else:
            options["cache_dir"] = str(ROOT / "output" / "agent" / "model-cache")
        snapshot_download(**options)
        if name == "local_generator":
            for relative, expected in entry["sha256"].items():
                with (ROOT / relative).open("rb") as handle:
                    actual = hashlib.file_digest(handle, "sha256").hexdigest()
                if actual != expected:
                    raise ValueError(f"Model checksum mismatch: {relative}")
        print(f"Prepared {name} at pinned revision {entry['revision']} on D:.")


if __name__ == "__main__":
    main()

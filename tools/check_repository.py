"""Check the published Git tree without loading models or contacting services."""
import argparse
import ast
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

SECRET = re.compile(rb"(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)")
PRIVATE_PARTS = {".venv", ".venv-agent", "__pycache__", "output", "tmp", "backups", "Program Files (x86)", "jiancha", "优化前", "RAG_data"}
PRIVATE_SUFFIXES = {".safetensors", ".pt", ".pth", ".bin", ".dll", ".exe", ".npy", ".npz", ".faiss", ".db", ".pem", ".key", ".zip", ".7z"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".jsonl", ".html", ".yml", ".yaml", ".ps1", ".pbtxt", ".lock", ".in"}


def git(*arguments):
    return subprocess.run(["git", *arguments], check=True, capture_output=True).stdout


def inspect_file(name, content):
    path = PurePosixPath(name)
    problems = []
    if (set(path.parts) & PRIVATE_PARTS or path.suffix in PRIVATE_SUFFIXES
            or name.startswith(("generate/model/", "graph/prepared/"))
            or path.name in {"legal_counsel_v2.json", "final_triplets.json", "final_triplets_verified.jsonl"}
            or path.name.startswith(".env") and path.name != ".env.example"
            or ".sqlite" in path.name):
        problems.append("private/generated/binary file must not be tracked")
    if len(content) > 10 * 1024 * 1024:
        problems.append("file exceeds the repository's 10 MiB limit")
    if SECRET.search(content):
        problems.append("possible credential or private key (value redacted)")
    if content.startswith(b"version https://git-lfs.github.com/spec/v1"):
        problems.append("LFS pointer in the source-only release")
    if path.suffix in TEXT_SUFFIXES or path.name.startswith(".") or name.startswith(".githooks/"):
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            problems.append("text file must use UTF-8")
        else:
            source_data = name.startswith("graph/Chinese-laws/")
            if "\r" in text and not source_data:
                problems.append("text file must use LF line endings")
            if text and not text.endswith("\n") and not source_data:
                problems.append("text file must end with a newline")
            if path.suffix == ".py":
                try:
                    ast.parse(text, filename=name)
                except SyntaxError as error:
                    problems.append(f"Python syntax error on line {error.lineno}")
            elif path.suffix == ".json":
                try:
                    json.loads(text)
                except json.JSONDecodeError as error:
                    problems.append(f"JSON syntax error on line {error.lineno}")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true", help="inspect only staged additions and changes")
    args = parser.parse_args()
    root = Path(git("rev-parse", "--show-toplevel").decode().strip())
    if args.staged:
        names = git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z").split(b"\0")
    else:
        names = git("ls-files", "-z").split(b"\0")
    failures = []
    count = 0
    for raw in names:
        if not raw:
            continue
        name = raw.decode("utf-8")
        content = git("show", ":" + name) if args.staged else (root / name).read_bytes()
        failures.extend(f"{name}: {problem}" for problem in inspect_file(name, content))
        count += 1
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"Repository checks passed: {count} files; no models or network calls.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

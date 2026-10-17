"""Lossless article boundaries, stable evidence IDs and explicit citation edges."""
import argparse
import hashlib
import json
import re
from collections import defaultdict, deque
from pathlib import Path

from .config import LAWS_DIR, PREPARED_DIR

NUM = r"[零〇一二三四五六七八九十百千万两0-9]+"
ARTICLE = re.compile(rf"(?m)^[ \t]*(?:《(?P<law>[^》\n]+)》)?[ \t]*第(?P<num>{NUM})条(?:之(?P<suffix>{NUM}))?")
REFERENCE = re.compile(rf"(?P<law>《[^》\n]+》|本法|本条例)第(?P<num>{NUM})条(?:之(?P<suffix>{NUM}))?")
HEADING = re.compile(rf"(?m)^[ \t]*第{NUM}[编章节][^\n]*$")


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def number(value):
    if value.isdecimal():
        return str(int(value))
    digits = dict(zip("零〇一二两三四五六七八九", [0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9]))
    units = {"十": 10, "百": 100, "千": 1000, "万": 10000}
    total = section = current = 0
    for char in value:
        if char in digits:
            current = digits[char]
        elif units[char] == 10000:
            total += (section + current) * 10000
            section = current = 0
        else:
            section += (current or 1) * units[char]
            current = 0
    return str(total + section + current)


def article_key(law, num, suffix=None):
    return f"{law}:{number(num)}" + (f":{number(suffix)}" if suffix else "")


def split_spans(text, start, end, limit):
    """Cut only within one article; retain exact document offsets for every slice."""
    while start < end:
        stop = min(start + limit, end)
        if stop < end:
            boundary = max(text.rfind(mark, start + limit // 2, stop) for mark in ["\n", "。", "；"])
            if boundary >= 0:
                stop = boundary + 1
        raw = text[start:stop]
        left = len(raw) - len(raw.lstrip())
        right = len(raw.rstrip())
        if right > left:
            yield start + left, start + right
        start = stop


def build_corpus(source_dir, limit=400):
    if limit < 100:
        raise ValueError("chunk size must be at least 100 characters")
    paths = sorted(Path(source_dir).glob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"No legal text files in {source_dir}")
    nodes, articles = [], defaultdict(list)
    snapshots = {}
    for path in paths:
        text = path.read_text(encoding="utf-8-sig")
        snapshot = digest(text)
        snapshots[path.name] = snapshot
        matches = list(ARTICLE.finditer(text))
        if not matches:
            raise ValueError(f"No article boundary recognized in {path.name}")
        for idx, match in enumerate(matches):
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            heading = HEADING.search(text, match.end(), end)
            if heading:
                end = heading.start()
            law = match.group("law") or path.stem.strip("《》")
            key = article_key(law, match.group("num"), match.group("suffix"))
            article_id = digest(f"{key}:{snapshot}")[:24]
            label = f"《{law}》第{match.group('num')}条" + (f"之{match.group('suffix')}" if match.group("suffix") else "")
            spans = list(split_spans(text, match.start(), end, limit))
            for part, (start, stop) in enumerate(spans):
                content = text[start:stop]
                evidence_id = digest(f"{article_id}:{start}:{stop}")[:24]
                node = {
                    "id": len(nodes), "evidence_id": evidence_id, "article_id": article_id,
                    "article_key": key, "article_label": label, "law_name": law,
                    "type": "ArticleChunk", "text": content, "content": content,
                    "search_text": content if content.startswith(label) else f"{label}\n{content}",
                    "metadata": {"source_file": path.name, "source_sha256": snapshot,
                                 "source_start": start, "source_end": stop, "part": part, "part_count": len(spans),
                                 "effective_from": None, "effective_to": None,
                                 "version_status": "not_supplied"},
                }
                nodes.append(node)
                articles[article_id].append(evidence_id)
    by_key = defaultdict(set)
    for node in nodes:
        by_key[node["article_key"]].add(node["article_id"])
    edges, unresolved = [], []
    for node in nodes:
        for ref in REFERENCE.finditer(node["text"]):
            law = ref.group("law")
            law = node["law_name"] if law in {"本法", "本条例"} else law.strip("《》")
            key = article_key(law, ref.group("num"), ref.group("suffix"))
            # A book-prefixed article heading names the current node; it is not a citation edge.
            if key == node["article_key"]:
                continue
            candidates = sorted(by_key.get(key, set()))
            edge = {"source_id": node["evidence_id"], "relation": "references",
                    "target_article_key": key, "quote": ref.group(),
                    "source_start": node["metadata"]["source_start"] + ref.start(),
                    "source_end": node["metadata"]["source_start"] + ref.end()}
            # An ambiguous cross-version reference must not silently select a version.
            if len(candidates) == 1:
                edges.append({**edge, "target_article_id": candidates[0]})
            else:
                unresolved.append({**edge, "status": "missing" if not candidates else "ambiguous"})
    graph = {"schema_version": 1, "article_members": dict(articles), "edges": edges,
             "unresolved_references": unresolved, "source_snapshots": snapshots}
    return nodes, graph


def write_corpus(source_dir=LAWS_DIR, output_dir=PREPARED_DIR, limit=400):
    nodes, graph = build_corpus(source_dir, limit)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    for name, data in [("corpus_nodes.json", nodes), ("corpus_texts.json", [n["search_text"] for n in nodes]),
                       ("evidence_graph.json", graph)]:
        (target / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return nodes, graph


class EvidenceStore:
    def __init__(self, directory=PREPARED_DIR):
        directory = Path(directory)
        self.nodes = json.loads((directory / "corpus_nodes.json").read_text(encoding="utf-8"))
        self.graph = json.loads((directory / "evidence_graph.json").read_text(encoding="utf-8"))
        self.by_id = {n["evidence_id"]: n for n in self.nodes}
        self.articles_by_key = defaultdict(set)
        for node in self.nodes:
            key = node["article_key"].removeprefix("中华人民共和国")
            self.articles_by_key[key].add(node["article_id"])
        self.references = defaultdict(list)
        self.article_references = defaultdict(list)
        for edge in self.graph["edges"]:
            self.references[edge["source_id"]].append(edge)
            self.article_references[self.by_id[edge["source_id"]]["article_id"]].append(edge)

    def find_articles(self, reference):
        """Resolve a full citation, article key or snapshot-specific article ID."""
        if reference in self.graph["article_members"]:
            return [reference]
        match = REFERENCE.fullmatch(reference.strip())
        key = article_key(match.group("law").strip("《》"), match.group("num"), match.group("suffix")) if match else reference
        return sorted(self.articles_by_key.get(key.removeprefix("中华人民共和国"), set()))

    def read_article(self, article_id):
        return [{**self.by_id[eid], "retrieval_origin": "article_lookup"}
                for eid in self.graph["article_members"][article_id]]

    def expand(self, docs, budget=8, max_hops=8):
        expanded = {d["evidence_id"]: dict(d) for d in docs}
        added = 0
        pending = deque((d["article_id"], [d["article_id"]]) for d in docs)
        seen = set()
        while pending:
            article_id, path = pending.popleft()
            if article_id in seen:
                continue
            seen.add(article_id)
            for eid in self.graph["article_members"][article_id]:
                if eid not in expanded and added < budget:
                    expanded[eid] = {**self.by_id[eid], "retrieval_origin": "graph", "rrf_score": 0.0,
                                     "retrieval_dependency_path": path}
                    added += 1
            if added >= budget:
                break
            if len(path) - 1 < max_hops:
                for edge in self.article_references[article_id]:
                    target = edge["target_article_id"]
                    if target not in seen:
                        pending.append((target, path + [target]))
        return list(expanded.values())

    def link_dataset(self, input_file, output_file):
        """Link existing triplets to source IDs without promoting labels to verified edges."""
        source = Path(input_file)
        text = source.read_text(encoding="utf-8-sig")
        rows = json.loads(text) if text.lstrip().startswith("[") else [json.loads(line) for line in text.splitlines() if line.strip()]
        normalize = lambda s: re.sub(r"\s+", "", s)
        by_text, by_article = defaultdict(list), defaultdict(list)
        for node in self.nodes:
            by_text[normalize(node["text"])].append(node["evidence_id"])
            by_article[node["article_key"]].append(node["evidence_id"])
        target = Path(output_file)
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() == target.resolve():
            raise ValueError("Dataset output must not overwrite the original")
        with target.open("w", encoding="utf-8") as handle:
            for row in rows:
                links = []
                for field in ("query", "instruction", "input", "positive", "hard_negative", "output"):
                    if field not in row:
                        continue
                    value = row.get(field, "")
                    ids = by_text.get(normalize(value), []) if value else []
                    method = "exact_text" if ids else "unresolved"
                    refs = list(REFERENCE.finditer(value)) if value else []
                    if not ids:
                        linked = set()
                        for ref in refs:
                            if ref.group("law").startswith("《"):
                                linked.update(by_article.get(article_key(ref.group("law").strip("《》"), ref.group("num"), ref.group("suffix")), []))
                        ids = sorted(linked)
                        method = "citation_only" if ids else "unresolved"
                    article_ids = sorted({self.by_id[eid]["article_id"] for eid in ids})
                    dependencies = [edge for aid in article_ids for edge in self.article_references[aid]]
                    links.append({"field": field, "evidence_ids": ids, "article_ids": article_ids,
                                  "dependency_edges": dependencies, "match_method": method,
                                  "verification_status": "not_verified",
                                  "usage": "dataset_annotation_only" if field == "output" else "source_link"})
                sample_id = digest(normalize(row.get("query", row.get("instruction", ""))) + "\n" + normalize(row.get("input", "")))[:24]
                input_ids = sorted({eid for link in links if link["field"] in {"query", "instruction", "input"}
                                    for eid in link["evidence_ids"]})
                handle.write(json.dumps({**row, "sample_id": sample_id, "evidence_links": links,
                                         "query_evidence_ids": input_ids, "graph_schema_version": 2,
                                         "graph_file": "evidence_graph.json"}, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(LAWS_DIR))
    parser.add_argument("--output", default=str(PREPARED_DIR))
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--dataset")
    parser.add_argument("--dataset-output")
    args = parser.parse_args()
    if bool(args.dataset) != bool(args.dataset_output):
        parser.error("--dataset and --dataset-output must be provided together")
    nodes, graph = write_corpus(args.source, args.output, args.chunk_size)
    if args.dataset:
        EvidenceStore(args.output).link_dataset(args.dataset, args.dataset_output)
    print(f"Prepared {len(nodes)} evidence chunks and {len(graph['edges'])} explicit citation edges; no model calls.")


if __name__ == "__main__":
    main()

"""CHAIN-inspired legal evidence priorities, not event or causal-effect probabilities."""
from collections import defaultdict, deque
from datetime import date
import json
import math
from pathlib import Path

from .config import DISTANCE_SCALE, MAX_DEPENDENCY_HOPS, RECURRENCE_GAIN, SOURCE_METADATA_FILE
from .corpus import digest


def canonical_claim(text):
    return "".join(c for c in text.casefold() if c.isalnum())


def parse_date(value, field):
    if value is None:
        return None
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO date")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field} must use YYYY-MM-DD") from error


class SourceMetadata:
    """Optional supplied provenance and legal validity; never infer dates from filenames."""
    def __init__(self, filename=SOURCE_METADATA_FILE):
        self.rows = {}
        if filename:
            rows = json.loads(Path(filename).read_text(encoding="utf-8"))
            if not isinstance(rows, dict):
                raise ValueError("Source metadata must be an object keyed by source filename")
            allowed = {"source_url", "source_group", "effective_from", "effective_to", "jurisdiction"}
            for name, values in rows.items():
                if not isinstance(values, dict) or set(values) - allowed:
                    raise ValueError(f"Unsupported source metadata for {name}")
                for field, value in values.items():
                    if value is not None and (not isinstance(value, str) or not value.strip()):
                        raise ValueError(f"Invalid {field} for {name}")
                lower = parse_date(values.get("effective_from"), "effective_from")
                upper = parse_date(values.get("effective_to"), "effective_to")
                if lower and upper and lower > upper:
                    raise ValueError(f"Reversed legal validity dates for {name}")
                self.rows[name] = values

    def apply(self, docs):
        result = []
        for doc in docs:
            metadata = {**doc["metadata"], **self.rows.get(doc["metadata"]["source_file"], {})}
            if metadata.get("effective_from") or metadata.get("effective_to"):
                metadata["version_status"] = "validity_metadata_supplied"
            result.append({**doc, "metadata": metadata})
        return result


def temporal_applicability(metadata, case_date):
    """Inclusive validity interval; open/missing bounds remain explicit uncertainty."""
    lower = parse_date(metadata.get("effective_from"), "effective_from")
    upper = parse_date(metadata.get("effective_to"), "effective_to")
    if lower and upper and lower > upper:
        raise ValueError("Legal validity interval is reversed")
    if case_date is None:
        return "case_date_missing", 1.0
    if lower and case_date < lower:
        return "not_yet_effective", 0.0
    if upper and case_date > upper:
        return "no_longer_effective", 0.0
    if lower is None or upper is None:
        return "partially_known" if lower or upper else "validity_unknown", 1.0
    return "within_supplied_interval", 1.0


def dependency_paths(graph, targets, max_hops):
    """Forward article-reference paths: a queried article depends on its cited articles."""
    adjacency = defaultdict(set)
    for edge in graph["edges"]:
        source = graph["evidence_articles"][edge["source_id"]]
        adjacency[source].add(edge["target_article_id"])
    paths = {target: [target] for target in sorted(targets)}
    queue = deque(paths)
    while queue:
        source = queue.popleft()
        if len(paths[source]) - 1 >= max_hops:
            continue
        for target in sorted(adjacency[source]):
            if target not in paths:
                paths[target] = paths[source] + [target]
                queue.append(target)
    return paths


class EvidenceWeighter:
    def __init__(self, store, distance_scale=DISTANCE_SCALE, recurrence_gain=RECURRENCE_GAIN,
                 max_hops=MAX_DEPENDENCY_HOPS, source_metadata=None):
        if not math.isfinite(distance_scale) or distance_scale <= 0:
            raise ValueError("Distance scale must be finite and positive")
        if not math.isfinite(recurrence_gain) or not 0 <= recurrence_gain <= 1 or max_hops < 1:
            raise ValueError("Invalid recurrence gain or dependency horizon")
        self.store = store
        self.scale, self.gain, self.max_hops = distance_scale, recurrence_gain, max_hops
        self.metadata = source_metadata or SourceMetadata()
        unknown_sources = set(self.metadata.rows) - {n["metadata"]["source_file"] for n in store.nodes}
        if unknown_sources:
            raise ValueError("Metadata contains unknown source filenames: " + ", ".join(sorted(unknown_sources)))
        self.graph = {**store.graph, "evidence_articles": {n["evidence_id"]: n["article_id"] for n in store.nodes}}
        self.article_source_keys = {
            n["article_id"]: self.source_key({**n, "metadata": {
                **n["metadata"], **self.metadata.rows.get(n["metadata"]["source_file"], {})}})
            for n in store.nodes}

    def apply(self, docs, case_date=None):
        case_date = parse_date(case_date, "case_date")
        docs = self.metadata.apply(docs)
        by_id = {d["evidence_id"]: d for d in docs}
        eligible = set()
        temporal = {}
        for doc in docs:
            status, factor = temporal_applicability(doc["metadata"], case_date)
            temporal[doc["evidence_id"]] = (status, factor)
            if factor and doc["verification"]["relation"] != "irrelevant":
                eligible.add(doc["evidence_id"])
        # A judgment depending on a rejected or date-ineligible companion cannot survive.
        changed = True
        while changed:
            changed = False
            for eid in list(eligible):
                if any(cid not in eligible for cid in by_id[eid]["verification"]["companion_ids"]):
                    eligible.remove(eid)
                    changed = True
        anchors = {d["article_id"] for d in docs if d["evidence_id"] in eligible and d.get("retrieval_origin") != "graph"}
        basis = "verified_retrieval_anchors"
        if not anchors:
            anchors = {by_id[eid]["article_id"] for eid in eligible}
            basis = "verified_evidence_anchors"
        # Joint evidence identified by the verifier is also a grounded legal dependency.
        query_edges = list(self.graph["edges"])
        for eid in eligible:
            for cid in by_id[eid]["verification"]["companion_ids"]:
                query_edges.append({"source_id": eid, "target_article_id": by_id[cid]["article_id"],
                                    "relation": "requires_evidence"})
        paths = dependency_paths({**self.graph, "edges": query_edges}, anchors, self.max_hops)
        weighted, groups = [], defaultdict(list)
        for doc in docs:
            eid, verdict = doc["evidence_id"], doc["verification"]
            path = paths.get(doc["article_id"])
            distance = len(path) - 1 if path else None
            proximity = 1 / (1 + distance / self.scale) if distance is not None else 0.0
            status, time_factor = temporal[eid]
            support = float(doc["evidence_weight"])
            if not math.isfinite(support) or not 0 <= support <= 1:
                raise ValueError("Invalid input relation support weight")
            base = support * proximity * time_factor if eid in eligible else 0.0
            origins = {self.source_key(doc)}
            origins.update(self.article_source_keys[aid] for aid in path or [])
            for cid in verdict["companion_ids"]:
                origins.add(self.source_key(by_id[cid]))
            claim = canonical_claim(verdict["claim"])
            if not claim:
                raise ValueError("Claim must contain meaningful characters")
            group_id = digest(claim + ":" + verdict["relation"])[:24]
            details = {"relation_support": support, "support_method": doc["weight_method"],
                       "dependency_distance": distance, "distance_factor": proximity,
                       "distance_basis": basis, "dependency_path": path or [],
                       "temporal_status": status, "temporal_factor": time_factor,
                       "case_date": case_date.isoformat() if case_date else None,
                       "source_keys": sorted(origins), "support_group": group_id,
                       "eligible": eid in eligible and path is not None,
                       "base_weight": base, "method": "legal_dependency_validity_recurrence_v1"}
            row = {**doc, "weighting": details, "evidence_weight": base}
            weighted.append(row)
            if details["eligible"] and base > 0:
                groups[group_id].append(row)
        usable = {d["evidence_id"] for d in weighted if d["weighting"]["eligible"]}
        changed = True
        while changed:
            changed = False
            for eid in list(usable):
                if any(cid not in usable for cid in by_id[eid]["verification"]["companion_ids"]):
                    usable.remove(eid)
                    changed = True
        groups.clear()
        for doc in weighted:
            if doc["evidence_id"] not in usable:
                doc["weighting"]["eligible"] = False
                doc["weighting"]["base_weight"] = 0.0
                doc["evidence_weight"] = 0.0
            elif doc["evidence_weight"] > 0:
                groups[doc["weighting"]["support_group"]].append(doc)
        # Overlapping source/dependency sets count as one support group. No Noisy-OR
        # probability claim is made for potentially dependent legal provisions.
        for members in groups.values():
            accepted = []
            for doc in sorted(members, key=lambda d: d["evidence_weight"], reverse=True):
                origins = set(doc["weighting"]["source_keys"])
                if not any(origins & previous for previous in accepted):
                    accepted.append(origins)
            count = len(accepted)
            peak = max(d["evidence_weight"] for d in members)
            bonus = (1 - peak) * self.gain * (1 - math.exp(-(count - 1)))
            for doc in members:
                doc["weighting"].update({"distinct_support_groups": count, "recurrence_bonus": bonus,
                                          "group_weight": min(1.0, peak + bonus)})
                doc["evidence_weight"] = min(1.0, doc["evidence_weight"] + bonus)
                doc["weight_method"] = "legal_dependency_validity_recurrence_v1"
        for doc in weighted:
            doc["weighting"].setdefault("distinct_support_groups", 0)
            doc["weighting"].setdefault("recurrence_bonus", 0.0)
        return weighted

    @staticmethod
    def source_key(doc):
        # Split chunks, reposts and revisions of the same numbered article must not inflate support.
        title, _, rest = doc["article_key"].partition(":")
        canonical_article = title.removeprefix("中华人民共和国") + ":" + rest
        group = doc["metadata"].get("source_group")
        return "source_group:" + group if group else "article:" + canonical_article

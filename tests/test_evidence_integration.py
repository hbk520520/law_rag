"""Deterministic software regression checks; no model calls or task-performance evaluation."""
import copy
from datetime import date
import json
from types import SimpleNamespace
import unittest

from law_rag.evidence import EvidenceVerifier, EvidenceVerificationError, question_graph, select_evidence
from law_rag.pipeline import LegalRAGPipeline
from law_rag.weighting import EvidenceWeighter, temporal_applicability


def document(eid="a", aid="A", article_key="示例:1", relation="supports", score=0.8, companions=(), origin="hybrid"):
    return {"id": 0, "evidence_id": eid, "article_id": aid, "article_key": article_key,
            "article_label": article_key, "law_name": "示例", "text": "条件甲成立时，命题乙适用。例外丙除外。",
            "search_text": "条件甲成立时，命题乙适用。例外丙除外。", "retrieval_origin": origin,
            "metadata": {"source_file": "fixture.txt", "part_count": 1, "part": 0,
                         "effective_from": None, "effective_to": None},
            "verification": {"evidence_id": eid, "claim": "命题乙适用", "relation": relation,
                             "relation_support_score": score, "quotes": ["命题乙适用"], "conditions": [],
                             "exceptions": [], "missing_facts": [], "companion_ids": list(companions),
                             "companion_quotes": [], "reason": "软件接口夹具", "status": "source_quotes_checked"},
            "evidence_weight": score if relation != "irrelevant" else 0,
            "weight_method": "uncalibrated_support_score"}


def store_for(docs, references=()):
    members = {}
    for doc in docs:
        members.setdefault(doc["article_id"], []).append(doc["evidence_id"])
    edges = [{"source_id": source, "target_article_id": target, "relation": "references", "quote": "配套引用"}
             for source, target in references]
    return SimpleNamespace(nodes=docs, graph={"article_members": members, "edges": edges})


def fake_client(items):
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"items": items}, ensure_ascii=False)))])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)))


class WeightingChecks(unittest.TestCase):
    def test_valid_old_law_is_not_decayed_by_age(self):
        status, factor = temporal_applicability({"effective_from": "1980-01-01", "effective_to": "2099-01-01"}, date(2026, 1, 1))
        self.assertEqual((status, factor), ("within_supplied_interval", 1.0))

    def test_known_outside_dates_are_excluded(self):
        self.assertEqual(temporal_applicability({"effective_from": "2030-01-01"}, date(2026, 1, 1))[1], 0.0)
        self.assertEqual(temporal_applicability({"effective_to": "2020-01-01"}, date(2026, 1, 1))[1], 0.0)

    def test_unknown_dates_are_not_fabricated(self):
        self.assertEqual(temporal_applicability({}, date(2026, 1, 1)), ("validity_unknown", 1.0))
        self.assertEqual(temporal_applicability({}, None), ("case_date_missing", 1.0))

    def test_reversed_interval_raises(self):
        with self.assertRaises(ValueError):
            temporal_applicability({"effective_from": "2030-01-01", "effective_to": "2020-01-01"}, None)

    def test_dependency_direction_and_cycle_are_respected(self):
        docs = [document(), document("b", "B", "示例:2", origin="graph"), document("c", "C", "示例:3", origin="graph")]
        store = store_for(docs, [("a", "B"), ("b", "C"), ("c", "A")])
        rows = EvidenceWeighter(store, recurrence_gain=0).apply(docs)
        self.assertEqual([r["weighting"]["dependency_distance"] for r in rows], [0, 1, 2])
        self.assertGreater(rows[0]["evidence_weight"], rows[1]["evidence_weight"])
        self.assertGreater(rows[1]["evidence_weight"], rows[2]["evidence_weight"])

    def test_repeat_slices_do_not_inflate_support_or_fill_primary_slots(self):
        docs = [document(), document("a2")]
        rows = EvidenceWeighter(store_for(docs)).apply(docs)
        self.assertEqual([r["weighting"]["distinct_support_groups"] for r in rows], [1, 1])
        self.assertEqual([r["evidence_weight"] for r in rows], [0.8, 0.8])
        selected = select_evidence(rows)
        self.assertEqual(sum(d.get("context_role") != "companion_source" for d in selected), 1)

    def test_nonoverlapping_sources_receive_bounded_recurrence_bonus(self):
        docs = [document(), document("b", "B", "示例:2")]
        rows = EvidenceWeighter(store_for(docs)).apply(docs)
        self.assertEqual(rows[0]["weighting"]["distinct_support_groups"], 2)
        self.assertGreater(rows[0]["evidence_weight"], 0.8)
        self.assertLessEqual(rows[0]["evidence_weight"], 1)

    def test_paths_with_shared_legal_dependency_do_not_count_twice(self):
        docs = [document(), document("b", "B", "示例:2", origin="graph"), document("c", "C", "示例:3", origin="graph")]
        rows = EvidenceWeighter(store_for(docs, [("a", "B"), ("a", "C")])).apply(docs)
        self.assertEqual([r["weighting"]["distinct_support_groups"] for r in rows], [1, 1, 1])
        self.assertEqual([r["weighting"]["recurrence_bonus"] for r in rows], [0, 0, 0])

    def test_overlapping_companion_sources_are_not_counted_twice(self):
        docs = [document(companions=["b"]), document("b", "B", "示例:2")]
        rows = EvidenceWeighter(store_for(docs)).apply(docs)
        self.assertEqual(rows[0]["weighting"]["distinct_support_groups"], 1)

    def test_zero_support_does_not_create_evidence_via_recurrence(self):
        docs = [document(score=0), document("b", "B", "示例:2", score=0)]
        rows = EvidenceWeighter(store_for(docs)).apply(docs)
        self.assertEqual([r["evidence_weight"] for r in rows], [0, 0])
        self.assertEqual(select_evidence(rows), [])

    def test_invalid_companion_invalidates_primary_and_blocks_selection(self):
        docs = [document(companions=["b"]), document("b", "B", "示例:2")]
        docs[1]["metadata"]["effective_to"] = "2020-01-01"
        rows = EvidenceWeighter(store_for(docs)).apply(docs, "2026-01-01")
        self.assertFalse(rows[0]["weighting"]["eligible"])
        self.assertEqual(select_evidence(rows), [])

    def test_verified_companion_forms_dependency_without_static_citation(self):
        docs = [document(companions=["b"]), document("b", "B", "示例:2", origin="graph")]
        rows = EvidenceWeighter(store_for(docs)).apply(docs)
        self.assertEqual(rows[1]["weighting"]["dependency_path"], ["A", "B"])
        self.assertTrue(rows[1]["weighting"]["eligible"])

    def test_required_evidence_beyond_horizon_blocks_dependent_claims(self):
        docs = [document(companions=["b"]), document("b", "B", "示例:2", companions=["c"], origin="graph"),
                document("c", "C", "示例:3", origin="graph")]
        rows = EvidenceWeighter(store_for(docs), max_hops=1).apply(docs)
        self.assertEqual(select_evidence(rows), [])


class VerificationChecks(unittest.TestCase):
    def test_hallucinated_source_quote_is_rejected(self):
        doc = document()
        item = copy.deepcopy(doc["verification"])
        item["quotes"] = ["不存在于原文"]
        with self.assertRaises(EvidenceVerificationError):
            EvidenceVerifier(client=fake_client([item])).verify("条件甲成立", [doc])

    def test_condition_without_law_quote_is_rejected(self):
        doc = document()
        item = copy.deepcopy(doc["verification"])
        item["conditions"] = [{"condition": "条件甲", "status": "met", "fact_quote": "条件甲成立",
                               "source_evidence_id": "a", "source_quote": "原文没有此条件"}]
        with self.assertRaises(EvidenceVerificationError):
            EvidenceVerifier(client=fake_client([item])).verify("条件甲成立", [doc])

    def test_missing_case_fact_becomes_conditional(self):
        doc = document()
        item = copy.deepcopy(doc["verification"])
        item["conditions"] = [{"condition": "条件甲", "status": "unknown", "fact_quote": "",
                               "source_evidence_id": "a", "source_quote": "条件甲成立时"}]
        result = EvidenceVerifier(client=fake_client([item])).verify("请问命题乙是否适用", [doc])
        self.assertEqual(result[0]["verification"]["relation"], "conditional")

    def test_rejected_condition_does_not_support_claim(self):
        doc = document()
        item = copy.deepcopy(doc["verification"])
        item["conditions"] = [{"condition": "条件甲", "status": "unmet", "fact_quote": "条件甲不成立",
                               "source_evidence_id": "a", "source_quote": "条件甲成立时"}]
        result = EvidenceVerifier(client=fake_client([item])).verify("条件甲不成立", [doc])
        self.assertEqual(result[0]["verification"]["relation"], "irrelevant")
        self.assertEqual(result[0]["evidence_weight"], 0)

    def test_missing_judgment_is_rejected(self):
        with self.assertRaises(EvidenceVerificationError):
            EvidenceVerifier(client=fake_client([])).verify("条件甲成立", [document()])

    def test_condition_exception_hyperedge_endpoints_exist(self):
        doc = document()
        doc["verification"]["conditions"] = [{"condition": "条件甲", "status": "met", "fact_quote": "条件甲成立",
                                              "source_evidence_id": "a", "source_quote": "条件甲成立时"}]
        doc["verification"]["exceptions"] = ["例外丙除外"]
        graph = question_graph([doc])
        ids = {node["id"] for node in graph["nodes"]}
        self.assertTrue(all(edge["source_id"] in ids and edge["target_id"] in ids for edge in graph["edges"]))
        self.assertEqual({edge["relation"] for edge in graph["edges"]}, {"contains", "supports", "requires", "exception_to"})
        self.assertEqual(len(graph["hyperedges"]), 1)

    def test_pipeline_passes_case_date_and_weighted_evidence_to_generator(self):
        doc = document()
        store = store_for([doc])
        store.expand = lambda docs, budget, hops: docs
        captured = {}
        pipeline = object.__new__(LegalRAGPipeline)
        pipeline.router = SimpleNamespace(process=lambda query: {"route": "simple_search", "search_query": query, "vector": "opaque"})
        pipeline.retriever = SimpleNamespace(store=store, retrieve=lambda query, vector, limit: [doc])
        pipeline.reranker = SimpleNamespace(predict=lambda pairs: [0.9])
        pipeline.verifier = EvidenceVerifier(client=fake_client([doc["verification"]]))
        pipeline.weighter = EvidenceWeighter(store)
        def generate(query, route, docs):
            captured["docs"] = docs
            return "software-contract-ok"
        pipeline.generator = SimpleNamespace(generate=generate)
        result = pipeline.answer("条件甲成立", case_date="2026-01-01")
        self.assertEqual(result["answer"], "software-contract-ok")
        self.assertEqual(captured["docs"][0]["weighting"]["case_date"], "2026-01-01")
        self.assertIn("hyperedges", result["evidence_graph"])


if __name__ == "__main__":
    unittest.main()

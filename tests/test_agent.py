"""Agent behavior checks with scripted model responses; no network or task experiments."""
import copy
from collections import defaultdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from openai.types.chat import ChatCompletion

from api import create_app
from law_rag.agent import AgentProtocolError, LegalAgent
from law_rag.corpus import EvidenceStore
from law_rag.evidence import EvidenceVerifier, EvidenceVerificationError
from law_rag.session import SessionConflict, SessionStore
from test_evidence_integration import document, fake_client, store_for


def call(name, **arguments):
    return name, {"reason": "根据核验结果执行本步骤", **arguments}


def finish(*statuses):
    return call("finish", results=[{"issue_id": f"issue_{index}", "status": status,
                "summary": "按当前证据说明结论与适用前提。"} for index, status in enumerate(statuses, 1)])


class ScriptedClient:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        name, arguments = next(self.actions)
        return ChatCompletion.model_validate({"id": f"response-{len(self.requests)}", "object": "chat.completion",
            "created": 0, "model": "scripted-software-check", "usage": {"prompt_tokens": 4, "completion_tokens": 4, "total_tokens": 8},
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {"role": "assistant", "content": None,
                "tool_calls": [{"id": f"call-{len(self.requests)}", "type": "function",
                                "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}]}}]})


class FixturePipeline:
    def __init__(self, questions=("命题乙是否适用",), mode="support"):
        self.mode, self.searches, self.reads, self.verifications, self.generated = mode, [], [], [], []
        self.a = document(article_key="中华人民共和国示例法:1")
        self.b = document("b", "A", "中华人民共和国示例法:1")
        self.a["metadata"]["part_count"] = self.b["metadata"]["part_count"] = 2 if mode == "article" else 1
        store = store_for([self.a, self.b])
        store.article_references = defaultdict(list)
        self.retriever = SimpleNamespace(store=store)
        self.verifier = SimpleNamespace(calibrator=SimpleNamespace(method="uncalibrated_support_score"))
        self.router = SimpleNamespace(process=lambda query, plan=False: {"route": "complex_rag", "search_query": query,
            "entities": {}, "intent": "consultation", "vector": None,
            "issues": [{"question": question, "search_query": question} for question in questions]})
        self.generator = SimpleNamespace(generate=self.generate)

    def search(self, query, *, rerank_query):
        self.searches.append((query, rerank_query))
        return [copy.deepcopy(self.a)]

    def read_article(self, reference):
        self.reads.append(reference)
        return {"documents": [copy.deepcopy(self.a), copy.deepcopy(self.b)], "candidates": [], "lookup_status": "found"}

    def verify_evidence(self, query, docs, case_date, *, claim, user_turns):
        self.verifications.append({"query": query, "claim": claim, "user_turns": list(user_turns), "date": case_date})
        result = copy.deepcopy(docs)
        for doc in result:
            verdict = doc["verification"]
            verdict.update(claim=claim, gaps=[], conditions=[], relation="supports", missing_facts=[])
            if self.mode == "article" and len(docs) < 2:
                verdict.update(relation="conditional", gaps=[{"kind": "article", "description": "缺少同条其余原文", "reference": "A"}])
            if self.mode == "fact":
                last = user_turns[-1]
                if "条件甲不成立" in last:
                    verdict["relation"] = "irrelevant"
                    doc["evidence_weight"] = 0
                elif "条件甲成立" not in last:
                    verdict.update(relation="conditional", gaps=[{"kind": "case_fact", "description": "条件甲是否成立", "reference": "a"}])
                    verdict["conditions"] = [{"condition": "条件甲成立", "status": "unknown", "fact_quote": "",
                        "source_evidence_id": "a", "source_quote": "条件甲成立时"}]
        return result

    def generate(self, query, route, docs):
        self.generated.append({"query": query, "route": copy.deepcopy(route), "docs": copy.deepcopy(docs)})
        return "按已提供的原文及适用前提说明。[参考资料 1]"


class AgentChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.database = Path(self.temp.name) / "cases.sqlite3"
        self.sessions = SessionStore(self.database)

    def tearDown(self):
        self.temp.cleanup()

    def agent(self, pipeline, actions, **kwargs):
        return LegalAgent(pipeline, self.sessions, client=ScriptedClient(actions), **kwargs)

    def test_two_issues_share_source_but_keep_separate_judgments(self):
        pipeline = FixturePipeline(("争点甲", "争点乙"))
        agent = self.agent(pipeline, [call("search_evidence", issue_id="issue_1", search_query="甲"),
            call("search_evidence", issue_id="issue_2", search_query="乙"), finish("resolved", "resolved")])
        result = agent.run("处理两个争点")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(result["issues"]), 2)
        self.assertEqual(len(result["docs"]), 1)
        self.assertEqual(len(result["docs"][0]["issue_judgments"]), 2)
        self.assertEqual(len(pipeline.generated), 1)
        self.assertEqual(len(pipeline.generated[0]["route"]["issues"]), 2)
        second = agent.client.requests[1]["messages"]
        self.assertEqual(second[-1]["role"], "tool")
        self.assertEqual(second[-1]["tool_call_id"], second[-2]["tool_calls"][0]["id"])

    def test_missing_article_drives_targeted_read_and_reverification(self):
        pipeline = FixturePipeline(mode="article")
        result = self.agent(pipeline, [call("search_evidence", issue_id="issue_1", search_query="命题乙"),
            call("read_article", issue_id="issue_1", reference="A"), finish("resolved")]).run("处理命题乙")
        self.assertEqual(pipeline.reads, ["A"])
        self.assertEqual(len(result["docs"]), 2)
        self.assertEqual(result["issues"][0]["gaps"], [])
        read_step = next(step for step in result["trace"] if step["action"] == "read_article")
        self.assertEqual(read_step["new_evidence_ids"], ["b"])

    def test_clarification_survives_new_agent_and_sqlite_connection(self):
        pipeline = FixturePipeline(mode="fact")
        first = self.agent(pipeline, [call("search_evidence", issue_id="issue_1", search_query="命题乙"),
            call("ask_user", issue_ids=["issue_1"], question="条件甲是否成立？")]).run("命题乙是否适用")
        self.assertEqual(first["status"], "needs_input")
        resumed = LegalAgent(pipeline, SessionStore(self.database), client=ScriptedClient([finish("resolved")]))
        second = resumed.run("条件甲成立", case_id=first["case_id"])
        self.assertEqual(second["case_id"], first["case_id"])
        self.assertEqual(second["status"], "completed")
        self.assertEqual(len(pipeline.searches), 1)
        self.assertEqual(pipeline.verifications[-1]["user_turns"], ["命题乙是否适用", "条件甲成立"])
        self.assertTrue(any(t["action"] == "reverify" for t in second["trace"]))
        self.assertEqual(second["conversation"][-1]["citations"], ["a"])

    def test_fact_correction_does_not_reuse_old_support_verdict(self):
        pipeline = FixturePipeline(mode="fact")
        first = self.agent(pipeline, [call("search_evidence", issue_id="issue_1", search_query="命题乙"), finish("resolved")]).run("条件甲成立")
        second = self.agent(pipeline, [finish("insufficient")]).run("更正：条件甲不成立", case_id=first["case_id"])
        self.assertEqual(second["docs"], [])
        self.assertEqual(second["status"], "insufficient")
        self.assertEqual(len(pipeline.generated), 1)

    def test_budget_produces_explicit_partial_result(self):
        pipeline = FixturePipeline(mode="article")
        result = self.agent(pipeline, [call("search_evidence", issue_id="issue_1", search_query="命题乙"),
            call("read_article", issue_id="issue_1", reference="A")], max_tools=1).run("处理命题乙")
        self.assertEqual(result["stop_reason"], "tool_budget_exhausted")
        self.assertEqual(result["status"], "insufficient")
        self.assertEqual(pipeline.reads, [])

    def test_repeat_search_does_not_duplicate_source_or_call_retriever(self):
        pipeline = FixturePipeline()
        search = call("search_evidence", issue_id="issue_1", search_query="命题乙")
        result = self.agent(pipeline, [search, search, finish("resolved")]).run("处理命题乙")
        self.assertEqual(len(pipeline.searches), 1)
        self.assertEqual(len(result["docs"]), 1)
        self.assertEqual(result["docs"][0]["evidence_weight"], 0.8)

    def test_finish_cannot_hide_an_unsearched_issue(self):
        pipeline = FixturePipeline(("争点甲", "争点乙"))
        agent = self.agent(pipeline, [call("search_evidence", issue_id="issue_1", search_query="甲"),
            finish("resolved", "insufficient"), call("search_evidence", issue_id="issue_2", search_query="乙"),
            finish("resolved", "resolved")])
        result = agent.run("处理两个争点")
        self.assertEqual(result["status"], "completed")
        rejected = json.loads(agent.client.requests[2]["messages"][-1]["content"])
        self.assertEqual(rejected["status"], "not_ready")
        self.assertEqual(len(pipeline.generated), 1)

    def test_unknown_tool_is_an_error_not_a_silent_answer(self):
        pipeline = FixturePipeline()
        with self.assertRaises(AgentProtocolError):
            self.agent(pipeline, [call("invent_legal_answer")]).run("处理命题乙")
        self.assertEqual(pipeline.generated, [])

    def test_snapshot_revision_rejects_lost_updates(self):
        first = {"case_id": "case", "revision": 0, "value": "initial"}
        first["revision"] = self.sessions.save(first)
        other = copy.deepcopy(first)
        first["value"] = "updated"
        self.sessions.save(first)
        with self.assertRaises(SessionConflict):
            self.sessions.save(other)
        self.assertEqual(self.sessions.load("case")["value"], "updated")

    def test_api_page_resume_and_missing_case(self):
        pipeline = FixturePipeline(mode="fact")
        agent = self.agent(pipeline, [call("search_evidence", issue_id="issue_1", search_query="命题乙"),
            call("ask_user", issue_ids=["issue_1"], question="条件甲是否成立？"), finish("resolved")])
        with TestClient(create_app(agent)) as client:
            self.assertIn("律据", client.get("/").text)
            self.assertEqual(client.post("/chat", json={"query": "   "}).status_code, 422)
            first = client.post("/chat", json={"query": "命题乙是否适用"}).json()
            self.assertEqual(first["status"], "needs_input")
            loaded = client.get("/cases/" + first["case_id"])
            self.assertEqual(loaded.json()["clarification"], "条件甲是否成立？")
            second = client.post("/chat", json={"query": "条件甲成立", "case_id": first["case_id"]})
            self.assertEqual(second.status_code, 200)
            self.assertEqual(second.json()["status"], "completed")
            self.assertEqual(client.get("/cases/" + "0" * 32).status_code, 404)

    def test_api_reports_invalid_controller_output_without_generating(self):
        pipeline = FixturePipeline()
        agent = self.agent(pipeline, [call("unknown_action")])
        with TestClient(create_app(agent)) as client:
            response = client.post("/chat", json={"query": "处理命题乙"})
            self.assertEqual(response.status_code, 502)
        self.assertEqual(pipeline.generated, [])


class SourceContractChecks(unittest.TestCase):
    def test_full_article_lookup_and_ambiguous_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = document(article_key="中华人民共和国示例法:1")
            b = document("b", "A", "中华人民共和国示例法:1")
            store = store_for([a, b])
            (root / "corpus_nodes.json").write_text(json.dumps([a, b]), encoding="utf-8")
            (root / "evidence_graph.json").write_text(json.dumps(store.graph), encoding="utf-8")
            loaded = EvidenceStore(root)
            self.assertEqual(loaded.find_articles("《示例法》第一条"), ["A"])
            self.assertEqual(len(loaded.read_article("A")), 2)
            c = document("c", "C", "中华人民共和国示例法:1")
            (root / "corpus_nodes.json").write_text(json.dumps([a, b, c]), encoding="utf-8")
            (root / "evidence_graph.json").write_text(json.dumps(store_for([a, b, c]).graph), encoding="utf-8")
            self.assertEqual(EvidenceStore(root).find_articles("《示例法》第1条"), ["A", "C"])

    def test_model_claim_cannot_supply_a_user_fact(self):
        doc = document()
        item = copy.deepcopy(doc["verification"])
        item["gaps"] = []
        item["conditions"] = [{"condition": "条件甲", "status": "met", "fact_quote": "条件甲成立",
                               "source_evidence_id": "a", "source_quote": "条件甲成立时"}]
        with self.assertRaises(EvidenceVerificationError):
            EvidenceVerifier(client=fake_client([item])).verify("条件甲成立（模型摘要）", [doc],
                claim=item["claim"], user_turns=["用户没有提供该事实"])

    def test_verifier_cannot_replace_the_target_issue(self):
        doc = document()
        item = copy.deepcopy(doc["verification"])
        item["gaps"] = []
        with self.assertRaises(EvidenceVerificationError):
            EvidenceVerifier(client=fake_client([item])).verify("用户原话", [doc], claim="另一个待判断命题", user_turns=["用户原话"])

    def test_citation_numbers_must_map_to_actual_context(self):
        from generate import RAGGenerator
        self.assertEqual(RAGGenerator._check_citations("结论[参考资料 1]", 1), "结论[参考资料 1]")
        with self.assertRaises(EvidenceVerificationError):
            RAGGenerator._check_citations("结论[参考资料 2]", 1)

    def test_citation_inside_complex_case_does_not_short_circuit_planning(self):
        import numpy as np
        from query import LegalQueryRouter
        embedder = SimpleNamespace(encode=lambda text, **kw: np.ones((len(text), 4)) if isinstance(text, list) else np.ones(4))
        plan = {"intent": "consultation", "rewritten_query": "两个争点", "entities": {},
                "issues": [{"question": "争点甲", "search_query": "甲"}, {"question": "争点乙", "search_query": "乙"}]}
        engine = SimpleNamespace(analyze=lambda text: copy.deepcopy(plan))
        with patch("query.SentenceTransformer", return_value=embedder), patch("query.DeepSeekQUEngine", return_value=engine):
            router = LegalQueryRouter()
            direct = router.process("请查看《示例法》第1条全文", plan=True)
            self.assertEqual(direct["route"], "article_lookup")
            complex_case = router.process("《示例法》第1条是否适用于我？另外还有争点乙。", plan=True)
            self.assertEqual(len(complex_case["issues"]), 2)
            self.assertEqual(complex_case["intent"], "consultation")


if __name__ == "__main__":
    unittest.main()

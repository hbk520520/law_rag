import argparse
import json
from datetime import date

from law_rag.agent import LegalAgent
from law_rag.config import SESSION_DB
from law_rag.pipeline import LegalRAGPipeline
from law_rag.session import SessionStore


def main():
    parser = argparse.ArgumentParser(description="基于证据反馈的法律问答 Agent")
    parser.add_argument("query", nargs="?", help="法律问题或同一案件的补充事实")
    parser.add_argument("--json", action="store_true", help="输出争点、工具轨迹和证据图")
    parser.add_argument("--case-date", type=date.fromisoformat, help="案件发生日期 YYYY-MM-DD")
    parser.add_argument("--case-id", help="续办已有案件")
    parser.add_argument("--show-case", help="只读取已保存的案件，无模型加载或调用")
    args = parser.parse_args()
    sessions = SessionStore(SESSION_DB)
    if args.show_case:
        print(json.dumps(sessions.load(args.show_case), ensure_ascii=False, indent=2))
        return
    if not args.query:
        parser.error("需要 query，或使用 --show-case")
    result = LegalAgent(LegalRAGPipeline(), sessions).run(args.query, case_id=args.case_id, case_date=args.case_date)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result["answer"])
        if result["case_id"]:
            print(f"\n案件编号：{result['case_id']}\n状态：{result['status']}")


if __name__ == "__main__":
    main()

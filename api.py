from contextlib import asynccontextmanager
from datetime import date
import logging
from threading import Lock
from typing import Annotated

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from openai import APIError
from pydantic import BaseModel, Field, StringConstraints, ValidationError

from law_rag.agent import AgentProtocolError, LegalAgent
from law_rag.config import ROOT, SESSION_DB
from law_rag.evidence import EvidenceVerificationError
from law_rag.pipeline import LegalRAGPipeline
from law_rag.session import CaseNotFound, SessionConflict, SessionStore

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=10000)]
    case_date: date | None = None
    case_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")


def create_app(agent=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.agent = agent if agent is not None else LegalAgent(LegalRAGPipeline(), SessionStore(SESSION_DB))
        app.state.run_lock = Lock()
        yield

    app = FastAPI(title="法律证据 Agent", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(ROOT / "web" / "index.html")

    @app.post("/chat")
    def chat_endpoint(req: ChatRequest, request: Request):
        # A single local GPU serves requests sequentially; model instances are shared.
        with request.app.state.run_lock:
            return request.app.state.agent.run(req.query, case_id=req.case_id, case_date=req.case_date)

    @app.get("/cases/{case_id}")
    def get_case(case_id: str, request: Request):
        return request.app.state.agent.get_case(case_id)

    @app.exception_handler(CaseNotFound)
    async def not_found(request, error):
        return JSONResponse(status_code=404, content={"detail": "案件不存在，请新建案件。"})

    @app.exception_handler(SessionConflict)
    async def conflict(request, error):
        return JSONResponse(status_code=409, content={"detail": str(error)})

    async def invalid_model_response(request, error):
        logger.error("Model response rejected (%s)", type(error).__name__)
        return JSONResponse(status_code=502, content={"detail": "模型返回的证据或行动格式不合格，本次未生成未经核验的答案。"})

    for error_type in (EvidenceVerificationError, AgentProtocolError, ValidationError):
        app.add_exception_handler(error_type, invalid_model_response)

    @app.exception_handler(APIError)
    async def provider_error(request, error):
        logger.error("Model provider failed (%s)", type(error).__name__)
        return JSONResponse(status_code=502, content={"detail": "模型服务请求失败，请检查 API 配置及服务状态。"})

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=6006)

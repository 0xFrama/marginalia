import os
import tempfile
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from api.schemas import AskRequest, AskResponse, IndexRequest
from index import QdrantStore
from qa import Answerer, OpenAIService
from retrieval import CrossEncoderReranker, Retriever
from models import IndexingResult
from index.pipeline import index_pdf
from orchestrator.graph import GraphOrchestrator
from patient.db import create_all
from patient.seed import seed_database
from patient.views import create_views

load_dotenv()

app = FastAPI()

STATIC_DIR = Path(__file__).parent / "static"


@lru_cache(maxsize=1)
def get_answerer() -> Answerer:
    store = get_store()
    retriever = Retriever(store)
    llm_client = OpenAIService()
    return Answerer(retriever=retriever, llm_client=llm_client)


def get_store() -> QdrantStore:
    return QdrantStore()


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()


@app.post("/index", response_model=IndexingResult)
async def index_document(
    request: IndexRequest, store: QdrantStore = Depends(get_store)
) -> IndexingResult:
    return index_pdf(request.pdf_path, store=store)


MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB


@app.post("/upload", response_model=IndexingResult)
async def upload_and_index(
    file: UploadFile = File(...),
    store: QdrantStore = Depends(get_store),
) -> IndexingResult:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF.")
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 50 MB limit.")
    # Use a temp file with a fixed .pdf suffix so index_pdf receives a safe path
    # that is independent of the (untrusted) upload filename.
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    try:
        return index_pdf(str(tmp_path), store=store, source_name=file.filename)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not process PDF: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/ask", response_model=AskResponse)
async def ask(
    request: AskRequest,
    answerer: Answerer = Depends(get_answerer),
) -> AskResponse:
    result = answerer.answer(
        request.question,
        candidate_k=request.candidate_k,
        top_k=request.top_k,
        min_score=request.min_score,
        reranker=get_reranker() if request.use_reranker else None,
        chat_history=request.chat_history,
    )
    return AskResponse(
        question=result.question,
        rewritten_query=result.rewritten_query,
        answer=result.answer,
        retrieved_evidence=result.sources,
        evidence_context=result.evidence,
        cited_sources=result.cited_sources,
    )


class ClinicalAskRequest(BaseModel):
    question: str
    patient_id: int


class ApproveRequest(BaseModel):
    thread_id: str
    decision: str


@lru_cache(maxsize=1)
def get_clinical() -> dict:
    """Build a seeded demo patient DB and one reusable orchestrator."""
    db_path = os.path.join(tempfile.gettempdir(), "marginalia_demo.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    create_all(engine)
    with Session(engine) as session:
        anchors = seed_database(session)
        session.commit()
        patients = [
            {"patient_id": anchor.patient_id, "name": anchor.name, "key": key}
            for key, anchor in anchors.items()
        ]
    create_views(engine)
    orch = GraphOrchestrator(engine, OpenAIService(), Retriever(QdrantStore()))
    patients.sort(key=lambda p: p["patient_id"])
    return {"orch": orch, "patients": patients}


def _serialize_source(source) -> dict:
    return {
        "citation_id": source.citation_id,
        "kind": source.kind,
        "text": source.text,
        "source_file": getattr(source, "source_file", None),
        "page_start": getattr(source, "page_start", None),
        "sql": getattr(source, "sql", None),
    }


def _serialize_step(step: dict) -> dict:
    if step["status"] == "awaiting_approval":
        return {
            "status": "awaiting_approval",
            "thread_id": step["thread_id"],
            "sql": step["sql"],
        }
    result = step["result"]
    return {
        "status": "done",
        "thread_id": step["thread_id"],
        "answer": result.answer,
        "planes": result.planes,
        "sources": [_serialize_source(s) for s in result.sources],
        "cited": [s.citation_id for s in result.cited_sources],
    }


@app.get("/")
async def index_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/patients")
async def list_patients() -> list[dict]:
    return get_clinical()["patients"]


@app.post("/clinical/ask")
async def clinical_ask(request: ClinicalAskRequest) -> dict:
    orchestrator = get_clinical()["orch"]
    step = orchestrator.start(request.question, request.patient_id)
    return _serialize_step(step)


@app.post("/clinical/approve")
async def clinical_approve(request: ApproveRequest) -> dict:
    orchestrator = get_clinical()["orch"]
    step = orchestrator.resume(request.thread_id, request.decision)
    return _serialize_step(step)

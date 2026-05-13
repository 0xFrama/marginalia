from functools import lru_cache

from dotenv import load_dotenv
from fastapi import Depends, FastAPI

from api.schemas import AskRequest, AskResponse, IndexRequest
from index import QdrantStore
from qa import Answerer, OpenAIService
from retrieval import CrossEncoderReranker, Retriever
from models import IndexingResult
from index.pipeline import index_pdf

load_dotenv()

app = FastAPI()


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
        answer=result.answer,
        retrieved_evidence=result.sources,
        evidence_context=result.evidence,
        cited_sources=result.cited_sources,
    )

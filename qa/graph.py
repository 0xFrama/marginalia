from typing import Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from models import AnswerResult, ChatMessage, EvidenceBlock, RetrievalHit
from qa.citations import extract_citation_ids, filter_cited_sources
from qa.prompt import SYSTEM_PROMPT, build_qa_messages, build_qa_prompt
from retrieval import CrossEncoderReranker, build_evidence_blocks, format_evidence
from retrieval.retriever import Retriever


class QAState(TypedDict):
    question: str
    chat_history: list[ChatMessage]
    candidate_k: int
    top_k: int
    min_score: float | None
    reranker: CrossEncoderReranker | None
    hits: list[RetrievalHit]
    evidence_blocks: list[EvidenceBlock]
    evidence_context: str
    answer: str
    cited_sources: list[EvidenceBlock]


class GraphAnswerer:
    def __init__(self, retriever: Retriever, llm_client: Any):
        self.retriever = retriever
        self.llm_client = llm_client
        self.graph = self._build_graph()

    def _retrieve_node(self, state: QAState) -> dict:
        hits = self.retriever.retrieve(
            question=state["question"],
            candidate_k=state["candidate_k"],
            min_score=state["min_score"],
            reranker=state["reranker"],
            top_k=state["top_k"],
        )
        return {"hits": hits}

    def _evidence_node(self, state: QAState) -> dict:
        blocks = build_evidence_blocks(state["hits"])
        evidence_context = format_evidence(blocks)
        if not evidence_context.strip():
            evidence_context = "No evidence was retrieved."

        return {
            "evidence_blocks": blocks,
            "evidence_context": evidence_context,
        }

    def _answer_node(self, state: QAState) -> dict:
        messages = build_qa_messages(
            question=state["question"],
            evidence_context=state["evidence_context"],
            chat_history=state["chat_history"],
        )

        if hasattr(self.llm_client, "generate_messages"):
            answer_text = self.llm_client.generate_messages(messages)
        else:
            user_prompt = build_qa_prompt(
                question=state["question"],
                evidence_context=state["evidence_context"],
                chat_history=state["chat_history"],
            )
            answer_text = self.llm_client.generate(SYSTEM_PROMPT, user_prompt)

        return {"answer": answer_text}

    def _citations_node(self, state: QAState) -> dict:
        citation_ids = extract_citation_ids(state["answer"])
        cited_sources = filter_cited_sources(citation_ids, state["evidence_blocks"])
        return {"cited_sources": cited_sources}

    def _build_graph(self):
        builder = StateGraph(QAState)
        builder.add_node("retrieve", self._retrieve_node)
        builder.add_node("evidence", self._evidence_node)
        builder.add_node("answer", self._answer_node)
        builder.add_node("citations", self._citations_node)

        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "evidence")
        builder.add_edge("evidence", "answer")
        builder.add_edge("answer", "citations")
        builder.add_edge("citations", END)

        return builder.compile()

    def answer(
        self,
        question: str,
        candidate_k: int = 10,
        min_score: float | None = None,
        reranker: CrossEncoderReranker | None = None,
        top_k: int = 3,
        chat_history: list[ChatMessage] | None = None,
    ) -> AnswerResult:
        final_state = self.graph.invoke(
            {
                "question": question,
                "chat_history": chat_history or [],
                "candidate_k": candidate_k,
                "top_k": top_k,
                "min_score": min_score,
                "reranker": reranker,
            }
        )

        return AnswerResult(
            question=question,
            answer=final_state["answer"],
            sources=final_state["evidence_blocks"],
            cited_sources=final_state["cited_sources"],
            evidence=final_state["evidence_context"],
        )

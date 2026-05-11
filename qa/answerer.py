from models.answer import AnswerResult
from qa.citations import extract_citation_ids, filter_cited_sources
from qa.prompt import SYSTEM_PROMPT, build_qa_prompt
from retrieval import (
    CrossEncoderReranker,
    Retriever,
    build_evidence_blocks,
    format_evidence,
)


class Answerer:
    def __init__(self, retriever: Retriever, llm_client):
        self.retriever = retriever
        self.llm_client = llm_client

    def answer(
        self,
        question: str,
        candidate_k: int = 10,
        min_score: float | None = None,
        reranker: CrossEncoderReranker | None = None,
        top_k: int = 3,
    ) -> AnswerResult:
        hits = self.retriever.retrieve(
            question,
            candidate_k=candidate_k,
            min_score=min_score,
            reranker=reranker,
            top_k=top_k,
        )
        blocks = build_evidence_blocks(hits)
        evidence_context = format_evidence(blocks)
        if not evidence_context.strip():
            evidence_context = "No evidence was retrieved."
        user_prompt = build_qa_prompt(question, evidence_context)
        answer_text = self.llm_client.generate(SYSTEM_PROMPT, user_prompt)
        citation_ids = extract_citation_ids(answer=answer_text)
        cited_sources = filter_cited_sources(citation_ids, blocks)
        return AnswerResult(
            question=question,
            answer=answer_text,
            sources=blocks,
            cited_sources=cited_sources,
            evidence=evidence_context,
        )

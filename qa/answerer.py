from models.answer import AnswerResult
from models.chat import ChatMessage
from qa.citations import extract_citation_ids, filter_cited_sources
from qa.prompt import SYSTEM_PROMPT, build_qa_messages, build_qa_prompt
from qa.rewrite import rewrite_query
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
        chat_history: list[ChatMessage] | None = None,
    ) -> AnswerResult:
        rewritten_query = rewrite_query(question, chat_history, self.llm_client)
        hits = self.retriever.retrieve(
            rewritten_query,
            candidate_k=candidate_k,
            min_score=min_score,
            reranker=reranker,
            top_k=top_k,
        )
        blocks = build_evidence_blocks(hits)
        evidence_context = format_evidence(blocks)
        if not evidence_context.strip():
            evidence_context = "No evidence was retrieved."
        messages = build_qa_messages(question, evidence_context, chat_history)

        if hasattr(self.llm_client, "generate_messages"):
            answer_text = self.llm_client.generate_messages(messages)
        else:
            user_prompt = build_qa_prompt(question, evidence_context, chat_history)
            answer_text = self.llm_client.generate(SYSTEM_PROMPT, user_prompt)
        citation_ids = extract_citation_ids(answer=answer_text)
        cited_sources = filter_cited_sources(citation_ids, blocks)
        return AnswerResult(
            question=question,
            rewritten_query=rewritten_query,
            answer=answer_text,
            sources=blocks,
            cited_sources=cited_sources,
            evidence=evidence_context,
        )

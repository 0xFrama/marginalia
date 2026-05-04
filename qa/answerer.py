from models.answer import AnswerResult
from qa.prompt import SYSTEM_PROMPT, build_qa_prompt
from retrieval import Retriever, build_evidence_blocks, format_evidence


class Answerer:
    def __init__(self, retriever: Retriever, llm_client):
        self.retriever = retriever
        self.llm_client = llm_client

    def answer(self, question: str, top_k: int = 3) -> AnswerResult:
        hits = self.retriever.retrieve(question, top_k=top_k)
        blocks = build_evidence_blocks(hits)
        evidence_context = format_evidence(blocks)
        if not evidence_context.strip():
            evidence_context = "No evidence was retrieved."
        user_prompt = build_qa_prompt(question, evidence_context)
        answer_text = self.llm_client.generate(SYSTEM_PROMPT, user_prompt)
        return AnswerResult(
            question=question,
            answer=answer_text,
            sources=blocks,
            evidence=evidence_context,
        )

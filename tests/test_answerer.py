from qa import Answerer
from models import AnswerResult, ChatMessage, Chunk, ChunkType, RetrievalHit


class FakeRetriever:
    def __init__(self, hits: list[RetrievalHit]) -> None:
        self.hits = hits
        self.calls = []

    def retrieve(
        self,
        question: str,
        candidate_k: int = 10,
        min_score: float | None = None,
        reranker=None,
        top_k: int = 5,
    ) -> list[RetrievalHit]:
        self.calls.append(
            {
                "question": question,
                "candidate_k": candidate_k,
                "top_k": top_k,
                "min_score": min_score,
                "reranker": reranker,
            }
        )
        return self.hits


class FakeLLMClient:
    def __init__(self, answer: str | list[str]) -> None:
        self.answer = answer
        self.calls = []

    def generate_messages(self, messages) -> str:
        self.calls.append({"messages": messages})
        if isinstance(self.answer, list):
            return self.answer.pop(0)
        return self.answer

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        return self.answer


def make_hit() -> RetrievalHit:
    return RetrievalHit(
        query_text="How should tomatoes be irrigated?",
        chunk=Chunk(
            doc_id="agri-guide.pdf",
            chunk_id="agri-guide.pdf:000001",
            text="Tomatoes need regular irrigation during hot periods.",
            source_file="agri-guide.pdf",
            page_start=1,
            page_end=1,
            section_title="Irrigation",
            chunk_type=ChunkType.BODY,
            chunk_index=1,
        ),
        score=0.9,
        rank=1,
    )


def test_answerer_returns_answer_result():
    question = "How should tomatoes be irrigated?"
    fake_answer = "Tomatoes need regular irrigation during hot periods [1]."
    answerer = Answerer(
        retriever=FakeRetriever([make_hit()]),
        llm_client=FakeLLMClient(fake_answer),
    )

    result = answerer.answer(question)

    assert isinstance(result, AnswerResult)
    assert result.question == question
    assert result.rewritten_query == question
    assert result.answer == fake_answer
    assert len(result.sources) == 1
    assert result.sources[0].citation_id == 1
    assert result.cited_sources == result.sources
    assert "Tomatoes need regular irrigation" in result.evidence


def test_answerer_calls_retriever_with_question():
    question = "How should tomatoes be irrigated?"
    retriever = FakeRetriever([make_hit()])
    answerer = Answerer(
        retriever=retriever,
        llm_client=FakeLLMClient("Tomatoes need irrigation [1]."),
    )

    answerer.answer(question)

    assert retriever.calls == [
        {
            "question": question,
            "candidate_k": 10,
            "top_k": 3,
            "min_score": None,
            "reranker": None,
        }
    ]


def test_answerer_passes_custom_top_k_to_retriever():
    question = "How should tomatoes be irrigated?"
    retriever = FakeRetriever([make_hit()])
    answerer = Answerer(
        retriever=retriever,
        llm_client=FakeLLMClient("Tomatoes need irrigation [1]."),
    )

    answerer.answer(question, top_k=2)

    assert retriever.calls == [
        {
            "question": question,
            "candidate_k": 10,
            "top_k": 2,
            "min_score": None,
            "reranker": None,
        }
    ]


def test_answerer_passes_candidate_k_and_reranker_to_retriever():
    question = "How should tomatoes be irrigated?"
    retriever = FakeRetriever([make_hit()])
    reranker = object()
    answerer = Answerer(
        retriever=retriever,
        llm_client=FakeLLMClient("Tomatoes need irrigation [1]."),
    )

    answerer.answer(question, candidate_k=8, top_k=2, reranker=reranker)

    assert retriever.calls == [
        {
            "question": question,
            "candidate_k": 8,
            "top_k": 2,
            "min_score": None,
            "reranker": reranker,
        }
    ]


def test_answerer_passes_min_score_to_retriever():
    question = "How should tomatoes be irrigated?"
    retriever = FakeRetriever([make_hit()])
    answerer = Answerer(
        retriever=retriever,
        llm_client=FakeLLMClient("Tomatoes need irrigation [1]."),
    )

    answerer.answer(question, min_score=0.55)

    assert retriever.calls == [
        {
            "question": question,
            "candidate_k": 10,
            "top_k": 3,
            "min_score": 0.55,
            "reranker": None,
        }
    ]


def test_answerer_sends_grounded_prompt_to_llm():
    question = "How should tomatoes be irrigated?"
    llm_client = FakeLLMClient("Tomatoes need irrigation [1].")
    answerer = Answerer(
        retriever=FakeRetriever([make_hit()]),
        llm_client=llm_client,
    )

    answerer.answer(question)

    assert len(llm_client.calls) == 1
    call = llm_client.calls[0]
    assert len(call["messages"]) == 2
    assert "provided evidence" in call["messages"][0].content
    assert question in call["messages"][1].content
    assert "Tomatoes need regular irrigation" in call["messages"][1].content
    assert "[1]" in call["messages"][1].content


def test_answerer_handles_no_evidence():
    question = "What does the document say about tomatoes?"
    llm_client = FakeLLMClient("The provided documents do not contain enough information.")
    answerer = Answerer(
        retriever=FakeRetriever([]),
        llm_client=llm_client,
    )

    result = answerer.answer(question)

    assert result.sources == []
    assert result.cited_sources == []
    assert "No evidence was retrieved." in result.evidence
    assert "No evidence was retrieved." in llm_client.calls[0]["messages"][1].content


def test_answerer_includes_chat_history_in_prompt():
    question = "Can you explain it more simply?"
    history = [
        ChatMessage(role="user", content="What is attention?"),
        ChatMessage(role="assistant", content="Attention maps queries to outputs [1]."),
    ]
    llm_client = FakeLLMClient(
        ["Explain attention in simpler terms.", "A simpler explanation [1]."]
    )
    answerer = Answerer(
        retriever=FakeRetriever([make_hit()]),
        llm_client=llm_client,
    )

    result = answerer.answer(question, chat_history=history)

    rewrite_messages = llm_client.calls[0]["messages"]
    answer_messages = llm_client.calls[1]["messages"]

    assert result.rewritten_query == "Explain attention in simpler terms."
    assert len(rewrite_messages) == 4
    assert rewrite_messages[1].content == "What is attention?"
    assert rewrite_messages[2].content == "Attention maps queries to outputs [1]."
    assert question in rewrite_messages[3].content

    assert len(answer_messages) == 4
    assert answer_messages[1].content == "What is attention?"
    assert answer_messages[2].content == "Attention maps queries to outputs [1]."
    assert question in answer_messages[3].content

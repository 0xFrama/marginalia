from typing import Any
import requests

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from models import AnswerResult, ChatMessage, EvidenceBlock, RetrievalHit
from qa.citations import extract_citation_ids, filter_cited_sources
from qa.prompt import SYSTEM_PROMPT, build_qa_messages, build_qa_prompt
from qa.rewrite import rewrite_query
from retrieval import CrossEncoderReranker, build_evidence_blocks, format_evidence
from retrieval.retriever import Retriever


class QAState(TypedDict):
    question: str
    rewritten_query: str
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
    needs_tool: bool
    tool_result: str | None


NO_EVIDENCE_ANSWER = "The provided documents do not contain enough information."
WEATHER_PROMPT = "Does this question require weather data? Answer YES or NO."


class GraphAnswerer:
    def __init__(self, retriever: Retriever, llm_client: Any):
        self.retriever = retriever
        self.llm_client = llm_client
        self.graph = self._build_graph()

    def _rewrite_node(self, state: QAState) -> dict:
        rewritten_query = rewrite_query(
            state["question"], state["chat_history"], self.llm_client
        )
        return {"rewritten_query": rewritten_query}

    def _retrieve_node(self, state: QAState) -> dict:
        hits = self.retriever.retrieve(
            question=state["rewritten_query"],
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
        tool_prefix = (
            f"[Live weather data]\n{state['tool_result']}\n\n"
            if state.get("tool_result")
            else ""
        )
        messages = build_qa_messages(
            question=state["question"],
            evidence_context=tool_prefix + state["evidence_context"],
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

    def _tool_router_node(self, state: QAState) -> dict:
        answer_text = self.llm_client.generate(
            "You are a routing assistant. Answer only YES or NO.",
            f"{WEATHER_PROMPT}\n\nQuestion: {state['question']}",
        )
        return {"needs_tool": "YES" in answer_text}

    def _weather_tool_node(self, state: QAState) -> dict:
        location = self.llm_client.generate(
            "You extract location names from text. Reply with only the location name, or NO LOCATION if none is found.",
            f"Extract only the location name from: {state['question']}.",
        )

        lat, lon = 46.06, 11.12

        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&daily=temperature_2m_max,precipitation_sum&forecast_days=7&timezone=auto"
        )
        resp = requests.get(url)
        data = resp.json()
        daily = data["daily"]
        lines = [
            f"Day {i + 1}: max {daily['temperature_2m_max'][i]}°C, rain {daily['precipitation_sum'][i]}mm"
            for i in range(7)
        ]

        tool_result = f"7-day forecast ({location}):\n" + "\n".join(lines)

        return {"tool_result": tool_result}

    def _citations_node(self, state: QAState) -> dict:
        citation_ids = extract_citation_ids(state["answer"])
        cited_sources = filter_cited_sources(citation_ids, state["evidence_blocks"])
        return {"cited_sources": cited_sources}

    def _fallback_node(self, state: QAState) -> dict:
        return {
            "answer": NO_EVIDENCE_ANSWER,
            "cited_sources": [],
        }

    def _route_after_evidence(self, state: QAState) -> str:
        if state["evidence_blocks"]:
            return "answer"
        return "fallback"

    def _build_graph(self):
        builder = StateGraph(QAState)
        builder.add_node("rewrite", self._rewrite_node)
        builder.add_node("tool_router", self._tool_router_node)
        builder.add_node("weather_tool", self._weather_tool_node)
        builder.add_node("retrieve", self._retrieve_node)
        builder.add_node("evidence", self._evidence_node)
        builder.add_node("answer", self._answer_node)
        builder.add_node("citations", self._citations_node)
        builder.add_node("fallback", self._fallback_node)

        builder.add_edge(START, "rewrite")
        builder.add_conditional_edges(
            "tool_router",
            lambda state: "weather_tool" if state["needs_tool"] else "retrieve",
            {"weather_tool": "weather_tool", "retrieve": "retrieve"},
        )
        builder.add_edge("weather_tool", "retrieve")
        builder.add_edge("rewrite", "tool_router")
        builder.add_edge("retrieve", "evidence")
        builder.add_conditional_edges(
            "evidence",
            self._route_after_evidence,
            {
                "answer": "answer",
                "fallback": "fallback",
            },
        )
        builder.add_edge("answer", "citations")
        builder.add_edge("citations", END)
        builder.add_edge("fallback", END)

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
                "rewritten_query": question,
                "chat_history": chat_history or [],
                "candidate_k": candidate_k,
                "top_k": top_k,
                "min_score": min_score,
                "reranker": reranker,
                "needs_tool": False,
                "tool_result": None,
            }
        )

        return AnswerResult(
            question=question,
            rewritten_query=final_state["rewritten_query"],
            answer=final_state["answer"],
            sources=final_state["evidence_blocks"],
            cited_sources=final_state["cited_sources"],
            evidence=final_state["evidence_context"],
        )

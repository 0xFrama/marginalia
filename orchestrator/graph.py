from dataclasses import dataclass, field
from typing import Callable
from typing_extensions import TypedDict

from langgraph.graph import START, END, StateGraph

from models.evidence import Evidence
from masking.identifiers import get_patient_identifiers
from masking.mask import mask
from masking.llm import MaskingLLM
from orchestrator.guideline_adapter import get_guideline_evidence
from orchestrator.router import route
from orchestrator.patient_adapter import get_patient_evidence
from orchestrator.fuse import fuse_evidence
from retrieval import format_evidence
from qa.prompt import build_qa_prompt
from qa.citations import extract_citation_ids, filter_cited_sources

SYSTEM_PROMPT = """You are a clinical assistant for doctors.

You answer using only two kinds of evidence, provided to you below:
- Patient records: facts about THIS specific patient (their own labs, medications, conditions, visits).
- Clinical guidelines: general public medical guidance, not tied to any patient.

Rules:
- Answer ONLY using the provided evidence. Do not use outside knowledge.
- Cite every claim with its evidence ID, like [1] or [2].
- When you state a patient fact, cite the patient-record evidence; when you state
  general guidance, cite the guideline evidence.
- If the evidence is insufficient to answer, say clearly that the provided evidence
  does not contain enough information. Do not guess and do not invent facts.
"""


# total = False --> not every key has to be present at all times.
class QAState(TypedDict, total=False):
    question: str
    patient_id: int
    identifiers: dict
    masked_llm: object
    masked_question: str
    planes: list[str]
    patient_evidence: list
    guideline_evidence: list
    fused: list
    answer: str
    cited_sources: list
    refused: bool


@dataclass
class OrchestratorResult:
    answer: str
    planes: list
    sources: list = field(default_factory=list)
    cited_sources: list = field(default_factory=list)


class GraphOrchestrator:
    def __init__(self, engine, llm, retriever, reranker=None):
        self.engine = engine
        self.llm = llm
        self.retriever = retriever
        self.reranker = reranker
        self.graph = self._build_graph()

    def _build_graph(self):
        b = StateGraph(QAState)
        for name in [
            "mask",
            "route",
            "patient",
            "guideline",
            "fuse",
            "answer",
            "citations",
            "fallback",
        ]:
            b.add_node(name, getattr(self, f"_{name}"))
        b.add_edge(START, "mask")
        b.add_edge("mask", "route")
        b.add_conditional_edges(
            "route", self._after_route, {"patient": "patient", "guideline": "guideline"}
        )
        b.add_conditional_edges(
            "patient", self._after_patient, {"guideline": "guideline", "fuse": "fuse"}
        )
        b.add_edge("guideline", "fuse")
        b.add_conditional_edges(
            "fuse", self._after_fuse, {"answer": "answer", "fallback": "fallback"}
        )
        b.add_edge("answer", "citations")
        b.add_edge("citations", END)
        b.add_edge("fallback", END)
        return b.compile()

    def answer(self, question, patient_id, approver=None) -> OrchestratorResult:
        final = self.graph.invoke({"question": question, "patient_id": patient_id})
        return OrchestratorResult(
            answer=final["answer"],
            planes=final.get("planes", []),
            sources=final.get("fused", []),
            cited_sources=final.get("cited_sources", []),
        )

    def _mask(self, state: QAState) -> dict:
        identifiers = get_patient_identifiers(self.engine, state["patient_id"])
        masked_llm = MaskingLLM(self.llm, identifiers)
        masked_question, _ = mask(state["question"], identifiers)
        return {
            "identifiers": identifiers,
            "masked_llm": masked_llm,
            "masked_question": masked_question,
        }

    def _route(self, state: QAState) -> dict:
        planes = route(state["question"], state["masked_llm"])
        return {"planes": planes}

    def _patient(self, state: QAState) -> dict:
        patient_evidence = get_patient_evidence(
            state["question"], state["patient_id"], self.engine, state["masked_llm"]
        )
        return {"patient_evidence": patient_evidence}

    def _guideline(self, state: QAState) -> dict:
        guideline_evidence = get_guideline_evidence(
            state["masked_question"], self.retriever, self.reranker
        )
        return {"guideline_evidence": guideline_evidence}

    def _fuse(self, state: QAState) -> dict:
        fused = fuse_evidence(
            state.get("patient_evidence") or [], state.get("guideline_evidence") or []
        )
        return {"fused": fused}

    def _answer(self, state: QAState) -> dict:
        context = format_evidence(state["fused"])
        qa_prompt = build_qa_prompt(state["question"], context)
        answer = state["masked_llm"].generate(SYSTEM_PROMPT, qa_prompt)
        return {"answer": answer}

    def _citations(self, state: QAState) -> dict:
        cited_sources = filter_cited_sources(
            extract_citation_ids(state["answer"]), state["fused"]
        )
        return {"cited_sources": cited_sources}

    def _fallback(self, state: QAState) -> dict:
        return {
            "answer": "The provided evidence does not contain enough information.",
            "cited_sources": [],
        }

    def _after_route(self, state):
        return "patient" if "patient" in state["planes"] else "guideline"

    def _after_patient(self, state):
        return "guideline" if "guideline" in state["planes"] else "fuse"

    def _after_fuse(self, state):
        return "answer" if state["fused"] else "fallback"

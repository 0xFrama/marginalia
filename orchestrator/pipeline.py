from dataclasses import dataclass, field

from masking.mask import mask
from masking.identifiers import get_patient_identifiers
from masking.llm import MaskingLLM
from models.evidence import Evidence
from orchestrator.fuse import fuse_evidence
from orchestrator.guideline_adapter import get_guideline_evidence
from orchestrator.patient_adapter import get_patient_evidence
from orchestrator.router import route
from qa.citations import extract_citation_ids, filter_cited_sources
from qa.prompt import build_qa_prompt
from retrieval import format_evidence

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


@dataclass
class OrchestratorResult:
    answer: str
    planes: list[str]
    sources: list[Evidence] = field(default_factory=list)
    cited_sources: list[Evidence] = field(default_factory=list)


def answer_question(question, patient_id, engine, llm, retriever, reranker=None):
    identifiers = get_patient_identifiers(engine, patient_id)
    masked_llm = MaskingLLM(llm, identifiers)

    masked_question, _ = mask(question, identifiers)

    planes = route(question, masked_llm)

    patient_ev = []
    guideline_ev = []
    if "patient" in planes:
        patient_ev = get_patient_evidence(question, patient_id, engine, masked_llm)
    if "guideline" in planes:
        guideline_ev = get_guideline_evidence(
            masked_question, retriever, reranker=reranker
        )

    fused = fuse_evidence(patient_ev, guideline_ev)
    evidence_context = format_evidence(fused)

    user_prompt = build_qa_prompt(question, evidence_context)

    answer_text = masked_llm.generate(SYSTEM_PROMPT, user_prompt)

    citation_ids = extract_citation_ids(answer_text)
    cited = filter_cited_sources(citation_ids, fused)

    return OrchestratorResult(
        answer=answer_text,
        planes=planes,
        sources=fused,
        cited_sources=cited,
    )

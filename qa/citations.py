import re

from models.evidence import EvidenceBlock


def extract_citation_ids(answer: str) -> set[int]:
    citation_ids = set()
    for bracket_content in re.findall(r"\[([^]]+)\]", answer):
        for num in re.findall(r"\d+", bracket_content):
            citation_ids.add(int(num))
    return citation_ids


def filter_cited_sources(
    citation_ids: set[int], evidence_blocks: list[EvidenceBlock]
) -> list[EvidenceBlock]:
    return [
        evidence for evidence in evidence_blocks if evidence.citation_id in citation_ids
    ]

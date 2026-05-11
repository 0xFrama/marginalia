from models import EvidenceBlock
from qa.citations import extract_citation_ids, filter_cited_sources


def make_evidence(citation_id: int) -> EvidenceBlock:
    return EvidenceBlock(
        citation_id=citation_id,
        text=f"Evidence {citation_id}",
        source_file="attention.pdf",
        page_start=citation_id,
        page_end=citation_id,
        section_title=None,
        score=0.5,
    )


def test_extract_citation_ids_from_answer():
    answer = "Attention maps queries to outputs [1]. Self-attention relates tokens [3]."

    assert extract_citation_ids(answer) == {1, 3}


def test_extract_citation_ids_from_grouped_citation():
    answer = "Attention is useful for sequence modeling [1, 3]."

    assert extract_citation_ids(answer) == {1, 3}


def test_extract_citation_ids_ignores_non_numeric_brackets():
    answer = "This is supported [1], but this bracket is not a citation [see appendix]."

    assert extract_citation_ids(answer) == {1}


def test_filter_cited_sources_returns_only_cited_evidence():
    evidence = [make_evidence(1), make_evidence(2), make_evidence(3)]

    cited = filter_cited_sources({1, 3}, evidence)

    assert [source.citation_id for source in cited] == [1, 3]

from models import Chunk, ChunkType, RetrievalHit
from models.evidence import Evidence
from retrieval.evidence import build_evidence_blocks, format_evidence


def test_build_evidence_blocks_preserves_hit_metadata():
    hit = RetrievalHit(
        query_text="What is attention?",
        chunk=Chunk(
            doc_id="attention.pdf",
            chunk_id="attention.pdf:000027",
            text="An attention function maps a query and key-value pairs to an output.",
            source_file="attention.pdf",
            page_start=3,
            page_end=3,
            section_title="3.2 Attention",
            chunk_type=ChunkType.BODY,
            chunk_index=27,
        ),
        score=0.6817968,
        rerank_score=7.23,
        rank=1,
    )

    blocks = build_evidence_blocks([hit])

    assert len(blocks) == 1
    assert blocks[0].citation_id == 1
    assert blocks[0].text == hit.chunk.text
    assert blocks[0].source_file == "attention.pdf"
    assert blocks[0].page_start == 3
    assert blocks[0].page_end == 3
    assert blocks[0].section_title == "3.2 Attention"
    assert blocks[0].score == 0.6817968
    assert blocks[0].rerank_score == 7.23


def test_format_evidence_renders_citation_source_and_score():
    hit = RetrievalHit(
        query_text="What is attention?",
        chunk=Chunk(
            doc_id="attention.pdf",
            chunk_id="attention.pdf:000027",
            text="An attention function maps a query and key-value pairs to an output.",
            source_file="attention.pdf",
            page_start=3,
            page_end=3,
            section_title="3.2 Attention",
            chunk_type=ChunkType.BODY,
            chunk_index=27,
        ),
        score=0.6817968,
        rank=1,
    )
    blocks = build_evidence_blocks([hit])

    formatted = format_evidence(blocks)

    assert "[1] attention.pdf, page: 3" in formatted
    assert "section: 3.2 Attention" in formatted
    assert "score: 0.682" in formatted
    assert "rerank_score" not in formatted
    assert hit.chunk.text in formatted


def test_format_evidence_renders_rerank_score_when_available():
    hit = RetrievalHit(
        query_text="What is attention?",
        chunk=Chunk(
            doc_id="attention.pdf",
            chunk_id="attention.pdf:000027",
            text="An attention function maps a query and key-value pairs to an output.",
            source_file="attention.pdf",
            page_start=3,
            page_end=3,
            section_title="3.2 Attention",
            chunk_type=ChunkType.BODY,
            chunk_index=27,
        ),
        score=0.6817968,
        rerank_score=7.23456,
        rank=1,
    )
    blocks = build_evidence_blocks([hit])

    formatted = format_evidence(blocks)

    assert "score: 0.682" in formatted
    assert "rerank_score: 7.235" in formatted


def test_format_evidence_handles_page_ranges_and_missing_section():
    hit = RetrievalHit(
        query_text="What does the table show?",
        chunk=Chunk(
            doc_id="agri-guide.pdf",
            chunk_id="agri-guide.pdf:000010",
            text="The table summarizes irrigation needs across crop stages.",
            source_file="agri-guide.pdf",
            page_start=4,
            page_end=5,
            section_title=None,
            chunk_type=ChunkType.TABLE,
            chunk_index=10,
        ),
        score=0.51234,
        rank=2,
    )
    blocks = build_evidence_blocks([hit])

    formatted = format_evidence(blocks)

    assert "[2] agri-guide.pdf, pages: 4-5" in formatted
    assert "section:" not in formatted
    assert "score: 0.512" in formatted


def test_format_evidence_returns_empty_string_for_no_blocks():
    assert format_evidence([]) == ""


def test_patient_evidence_renders():
    # A patient block has no page/score fields — it must render via the patient
    # branch (no crash on the None guideline fields) with its own label/as-of.
    block = Evidence(
        citation_id=3,
        text="HbA1c 6.8%",
        kind="patient",
        source_label="labs",
        as_of="2026-02-10",
    )
    formatted = format_evidence([block])

    assert formatted.startswith("[3] labs")
    assert "as-of 2026-02-10" in formatted
    assert "HbA1c 6.8%" in formatted
    # No guideline-only fields should leak into a patient citation.
    assert "page" not in formatted
    assert "score" not in formatted


def test_patient_evidence_renders_with_missing_optional_fields():
    # source_label and as_of are optional — rendering must not print "None".
    block = Evidence(citation_id=4, text="BP 130", kind="patient")
    formatted = format_evidence([block])

    assert formatted.startswith("[4] Patient record")
    assert "None" not in formatted

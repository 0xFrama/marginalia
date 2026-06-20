from models import RetrievalHit, EvidenceBlock


def build_evidence_blocks(hits: list[RetrievalHit]) -> list[EvidenceBlock]:
    return [
        EvidenceBlock(
            citation_id=hit.rank,
            text=hit.chunk.text,
            source_file=hit.chunk.source_file,
            page_start=hit.chunk.page_start,
            page_end=hit.chunk.page_end,
            section_title=hit.chunk.section_title,
            score=hit.score,
            rerank_score=hit.rerank_score,
        )
        for hit in hits
    ]


def format_evidence(blocks: list[EvidenceBlock]) -> str:
    formatted_text = []

    for block in blocks:
        if block.kind == "patient":
            label = block.source_label or "Patient record"
            asof = f" · as-of {block.as_of}" if block.as_of else ""
            formatted_text.append(
                f"[{block.citation_id}] {label}{asof}\n{block.text}"
            )
            continue
        if block.page_start == block.page_end:
            page_label = f"page: {block.page_start}"
        else:
            page_label = f"pages: {block.page_start}-{block.page_end}"

        section_label = (
            f", section: {block.section_title}" if block.section_title else ""
        )
        score_label = f", score: {round(block.score, 3)}"
        if block.rerank_score is not None:
            score_label += f", rerank_score: {round(block.rerank_score, 3)}"

        formatted_text.append(
            f"[{block.citation_id}] {block.source_file}, {page_label}"
            f"{section_label}{score_label}\n{block.text}"
        )

    return "\n\n".join(formatted_text)

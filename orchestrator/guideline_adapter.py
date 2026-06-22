from retrieval import build_evidence_blocks


def get_guideline_evidence(question, retriever, reranker=None, candidate_k=10, top_k=5):
    hits = retriever.retrieve(
        question, candidate_k=candidate_k, reranker=reranker, top_k=top_k
    )
    return build_evidence_blocks(hits)

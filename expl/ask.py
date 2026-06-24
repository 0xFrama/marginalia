import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from index import QdrantStore
from models import ChatMessage
from qa import Answerer, OpenAIService
from retrieval import CrossEncoderReranker, Retriever

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask a question over indexed PDFs.")
    parser.add_argument(
        "question",
        nargs="?",
        default="What is attention?",
        help="Question to answer using retrieved evidence.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Maximum number of retrieved evidence chunks to use.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=10,
        help="Number of initial retrieval candidates to fetch before reranking.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Optional minimum retrieval score for evidence chunks.",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable cross-encoder reranking before answer generation.",
    )
    parser.add_argument(
        "--history-file",
        type=Path,
        default=None,
        help="Optional JSON file containing prior chat messages.",
    )
    return parser.parse_args()


def load_chat_history(path: Path | None) -> list[ChatMessage]:
    if path is None:
        return []

    payload = json.loads(path.read_text(encoding="utf-8"))
    return [ChatMessage.model_validate(message) for message in payload]


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    args = parse_args()

    store = QdrantStore()
    retriever = Retriever(store)
    llm_client = OpenAIService()
    reranker = CrossEncoderReranker() if args.rerank else None
    chat_history = load_chat_history(args.history_file)
    answerer = Answerer(retriever=retriever, llm_client=llm_client)

    result = answerer.answer(
        args.question,
        candidate_k=args.candidate_k,
        top_k=args.top_k,
        min_score=args.min_score,
        reranker=reranker,
        chat_history=chat_history,
    )

    print(f"Question: {result.question}\n")
    print(f"Rewritten query: {result.rewritten_query}\n")
    print("Answer:")
    print(result.answer)
    print("\nCited resources:")
    for source in result.cited_sources:
        if source.page_start == source.page_end:
            page_label = f"page {source.page_start}"
        else:
            page_label = f"pages {source.page_start}-{source.page_end}"

        section_label = (
            f", section: {source.section_title}" if source.section_title else ""
        )
        rerank_label = f", rerank_score: {round(source.rerank_score, 3)}" if source.rerank_score is not None else ""
        print(
            f"[{source.citation_id}] {source.source_file}, {page_label}"
            f"{section_label}, score: {round(source.score, 3)}{rerank_label}"
        )
    print("\nRetrieved evidence:")
    for source in result.sources:
        if source.page_start == source.page_end:
            page_label = f"page {source.page_start}"
        else:
            page_label = f"pages {source.page_start}-{source.page_end}"

        section_label = (
            f", section: {source.section_title}" if source.section_title else ""
        )
        rerank_label = f", rerank_score: {round(source.rerank_score, 3)}" if source.rerank_score is not None else ""
        print(
            f"[{source.citation_id}] {source.source_file}, {page_label}"
            f"{section_label}, score: {round(source.score, 3)}{rerank_label}"
        )


if __name__ == "__main__":
    main()

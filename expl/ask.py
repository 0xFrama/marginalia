import argparse
import os

from dotenv import load_dotenv

from index import QdrantStore
from qa import Answerer, OpenAIService
from retrieval import Retriever

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
        "--min-score",
        type=float,
        default=None,
        help="Optional minimum retrieval score for evidence chunks.",
    )
    return parser.parse_args()


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    args = parse_args()

    store = QdrantStore()
    retriever = Retriever(store)
    llm_client = OpenAIService()
    answerer = Answerer(retriever=retriever, llm_client=llm_client)

    result = answerer.answer(
        args.question,
        top_k=args.top_k,
        min_score=args.min_score,
    )

    print(f"Question: {result.question}\n")
    print("Answer:")
    print(result.answer)
    print("\nRetrieved evidence:")
    for source in result.sources:
        if source.page_start == source.page_end:
            page_label = f"page {source.page_start}"
        else:
            page_label = f"pages {source.page_start}-{source.page_end}"

        section_label = (
            f", section: {source.section_title}" if source.section_title else ""
        )
        print(
            f"[{source.citation_id}] {source.source_file}, {page_label}"
            f"{section_label}, score: {round(source.score, 3)}"
        )


if __name__ == "__main__":
    main()

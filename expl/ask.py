import os
import sys
from dotenv import load_dotenv

from index import QdrantStore
from qa import Answerer, OpenAIService
from retrieval import Retriever

load_dotenv()


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    question = " ".join(sys.argv[1:]) or "What is attention?"

    store = QdrantStore()
    retriever = Retriever(store)
    llm_client = OpenAIService()
    answerer = Answerer(retriever=retriever, llm_client=llm_client)

    result = answerer.answer(question)

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

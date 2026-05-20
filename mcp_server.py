from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from index import QdrantStore
from index.pipeline import index_pdf
from qa import GraphAnswerer, OpenAIService
from retrieval import Retriever

load_dotenv()

mcp = FastMCP("Marginalia Agricultural RAG")

_store = QdrantStore()
_retriever = Retriever(_store)
_llm = OpenAIService()
_answerer = GraphAnswerer(retriever=_retriever, llm_client=_llm)


@mcp.tool()
def ask_agricultural_rag(question: str) -> str:
    """Query the agricultural knowledge base and return a grounded answer with citations.

    Use this tool when the user asks questions about crops, yields, production,
    irrigation, or any agricultural topic that may be covered in the knowledge base.
    """
    result = _answerer.answer(question)

    cited = "\n".join(
        f"[{b.citation_id}] {b.source_file}, page {b.page_start}, section: {b.section_title}"
        for b in result.cited_sources
    )

    if cited:
        return f"{result.answer}\n\nSources:\n{cited}"
    return result.answer


@mcp.tool()
def index_document(pdf_path: str) -> str:
    """Index a new PDF document into the agricultural knowledge base.

    Provide the full path to a local PDF file. Once indexed, its content
    will be available for retrieval by ask_agricultural_rag.
    """
    result = index_pdf(pdf_path, store=_store)
    return (
        f"Indexed {result.source_file}: "
        f"{result.indexed_count} chunks added to collection '{result.collection_name}'."
    )


if __name__ == "__main__":
    mcp.run()

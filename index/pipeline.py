from pathlib import Path

from ingest import ingest_pdf
from index import QdrantStore, embed_chunks
from models.indexing import IndexingResult
from models.retrieval import RetrievalHit


def retrieval(query: str, store: QdrantStore | None = None) -> list[RetrievalHit]:
    store = store or QdrantStore()
    hits = store.search(query)
    return hits


def index_pdf(pdf_path: str, store: QdrantStore | None = None) -> IndexingResult:
    chunks = ingest_pdf(pdf_path)
    records = embed_chunks(chunks)

    store = store or QdrantStore()
    store.add_embeddings(records)

    return IndexingResult(
        source_file=Path(pdf_path).name,
        chunk_count=len(chunks),
        indexed_count=len(records),
        collection_name=store.collection_name,
    )


if __name__ == "__main__":

    store = QdrantStore()

    index_pdf("samples/attention.pdf", store=store)

    query = "What is Attention?"
    print(retrieval(query, store=store))

import uuid

import pytest

from index.qdrant_store import QdrantStore
from models import Chunk, ChunkType, EmbeddingRecord


@pytest.fixture
def qdrant_store():
    collection_name = f"test_chunks_{uuid.uuid4().hex}"

    try:
        store = QdrantStore(
            collection_name=collection_name,
            vector_size=3,
            model=object(),
        )
    except Exception as exc:
        pytest.skip(f"Qdrant is not available locally: {exc}")

    try:
        yield store
    finally:
        store.client.delete_collection(collection_name)


def test_qdrant_store_indexes_and_searches_chunks(qdrant_store, monkeypatch):
    first_chunk = Chunk(
        doc_id="agri-guide.pdf",
        chunk_id="agri-guide.pdf:000001",
        text="Tomatoes need regular irrigation during hot and dry periods.",
        source_file="agri-guide.pdf",
        page_start=1,
        page_end=1,
        section_title="Irrigation",
        chunk_type=ChunkType.BODY,
        chunk_index=1,
    )
    second_chunk = Chunk(
        doc_id="agri-guide.pdf",
        chunk_id="agri-guide.pdf:000002",
        text="Powdery mildew can affect grape leaves in humid conditions.",
        source_file="agri-guide.pdf",
        page_start=2,
        page_end=2,
        section_title="Plant diseases",
        chunk_type=ChunkType.BODY,
        chunk_index=2,
    )
    records = [
        EmbeddingRecord(chunk=first_chunk, embedding=[1.0, 0.0, 0.0]),
        EmbeddingRecord(chunk=second_chunk, embedding=[0.0, 1.0, 0.0]),
    ]

    monkeypatch.setattr(
        "index.qdrant_store.embed_text",
        lambda query_text, model=None: [1.0, 0.0, 0.0],
    )

    qdrant_store.add_embeddings(records)

    hits = qdrant_store.search("How should tomatoes be irrigated?", top_k=1)

    assert len(hits) == 1
    assert hits[0].chunk.chunk_id == "agri-guide.pdf:000001"
    assert hits[0].rank == 1
    assert hits[0].score > 0


def test_qdrant_store_gets_chunk_by_id(qdrant_store):
    chunk = Chunk(
        doc_id="agri-guide.pdf",
        chunk_id="agri-guide.pdf:000003",
        text="Olive trees require careful pruning to improve canopy airflow.",
        source_file="agri-guide.pdf",
        page_start=3,
        page_end=3,
        section_title="Pruning",
        chunk_type=ChunkType.BODY,
        chunk_index=3,
    )

    qdrant_store.add_embeddings(
        [EmbeddingRecord(chunk=chunk, embedding=[0.0, 0.0, 1.0])]
    )

    stored_chunk = qdrant_store.get_by_chunk_id("agri-guide.pdf:000003")

    assert stored_chunk is not None
    assert stored_chunk.chunk_id == "agri-guide.pdf:000003"
    assert stored_chunk.text == chunk.text


def test_qdrant_store_filters_by_chunk_type(qdrant_store, monkeypatch):
    body_chunk = Chunk(
        doc_id="agri-guide.pdf",
        chunk_id="agri-guide.pdf:000004",
        text="Drip irrigation delivers water directly to crop roots.",
        source_file="agri-guide.pdf",
        page_start=4,
        page_end=4,
        section_title="Irrigation",
        chunk_type=ChunkType.BODY,
        chunk_index=4,
    )
    caption_chunk = Chunk(
        doc_id="agri-guide.pdf",
        chunk_id="agri-guide.pdf:000005",
        text="Figure 2: Irrigation system layout.",
        source_file="agri-guide.pdf",
        page_start=5,
        page_end=5,
        section_title="Irrigation figures",
        chunk_type=ChunkType.CAPTION,
        chunk_index=5,
    )
    records = [
        EmbeddingRecord(chunk=body_chunk, embedding=[1.0, 0.0, 0.0]),
        EmbeddingRecord(chunk=caption_chunk, embedding=[1.0, 0.0, 0.0]),
    ]

    monkeypatch.setattr(
        "index.qdrant_store.embed_text",
        lambda query_text, model=None: [1.0, 0.0, 0.0],
    )

    qdrant_store.add_embeddings(records)

    hits = qdrant_store.search(
        "How does drip irrigation work?",
        chunk_types=[ChunkType.BODY],
        top_k=5,
    )

    assert hits
    assert all(hit.chunk.chunk_type == ChunkType.BODY for hit in hits)

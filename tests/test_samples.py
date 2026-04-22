import pytest

from ingest import ingest_pdf
from models.chunk import Chunk, ChunkType


@pytest.fixture(scope="module")
def chunks():
    return ingest_pdf("samples/attention.pdf")


def test_ingest_returns_chunks(chunks):
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)


def test_chunks_have_required_metadata(chunks):
    for chunk in chunks:
        assert chunk.text.strip()
        assert chunk.source_file
        assert chunk.page_start >= 1
        assert chunk.page_end >= chunk.page_start
        assert isinstance(chunk.chunk_type, ChunkType)


def test_chunk_indices_are_sequential(chunks):
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i

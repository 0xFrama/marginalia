import pytest
import json

from ingest import ingest_pdf
from models.chunk import Chunk, ChunkType
from ingest.export import export_to_jsonl


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


def test_chunk_record_serializes_cleanly(chunks):
    jchunk = chunks[0].to_record()
    assert jchunk["schema_version"] == "chunk.v1"
    assert isinstance(jchunk["chunk_type"], str)
    assert jchunk["chunk_id"]
    assert jchunk["doc_id"]
    assert isinstance(jchunk, dict)

def test_export_to_jsonl_writes_one_line_per_chunk(tmp_path):
    fake_chunk_1 = Chunk(
        doc_id="name_1",
        chunk_id="name_1:000001",
        text="This is a first test",
        source_file="attention.pdf",
        page_start=1,
        page_end=1,
        section_title="Introduction",
        chunk_type=ChunkType.BODY,
        chunk_index=1,
    )

    fake_chunk_2 = Chunk(
        doc_id="name_1",
        chunk_id="name_1:000002",
        text="This is a second test",
        source_file="attention.pdf",
        page_start=1,
        page_end=1,
        section_title="Introduction",
        chunk_type=ChunkType.BODY,
        chunk_index=2,
    )

    output_path = tmp_path / "output.jsonl"
    export_to_jsonl([fake_chunk_1, fake_chunk_2], output_path)

    with output_path.open("r", encoding="utf-8") as f:
        content = [json.loads(line) for line in f]

    assert len(content) == 2
    assert isinstance(content[0], dict)
    assert isinstance(content[1], dict)
    assert content[0]["chunk_id"] == "name_1:000001"
    assert content[1]["chunk_id"] == "name_1:000002"

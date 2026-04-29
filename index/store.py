from __future__ import annotations

from pathlib import Path

import chromadb

from index.embeddings import embed_text, load_embed_model
from models.chunk import Chunk
from models.embedding import EmbeddingRecord
from models.retrieval import RetrievalHit


class ChromaStore:
    def __init__(self, path: str | Path = "./my_db", collection_name: str = "docs"):
        self.client = chromadb.PersistentClient(path=str(Path(path)))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.model = load_embed_model()

    def add_embeddings(self, records: list[EmbeddingRecord]) -> None:
        if not records:
            return

        ids = [record.chunk.chunk_id for record in records]
        embeddings = [record.embedding for record in records]
        documents = [record.chunk.text for record in records]
        metadatas = [self._chunk_metadata(record.chunk) for record in records]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def search(self, query_text: str, top_k: int = 5) -> list[RetrievalHit]:
        query_vec = embed_text(query_text, model=self.model)
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        hits: list[RetrievalHit] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for rank, (text, metadata, distance) in enumerate(
            zip(documents, metadatas, distances, strict=True),
            start=1,
        ):
            chunk = self._chunk_from_metadata(metadata, text)
            hits.append(
                RetrievalHit(
                    query_text=query_text,
                    chunk=chunk,
                    score=1.0 - float(distance),
                    rank=rank,
                )
            )

        return hits

    def get_by_chunk_id(self, chunk_id: str) -> Chunk | None:
        result = self.collection.get(
            ids=[chunk_id],
            include=["documents", "metadatas"],
        )

        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        if not documents or not metadatas:
            return None

        return self._chunk_from_metadata(metadatas[0], documents[0])

    @staticmethod
    def _chunk_metadata(chunk: Chunk) -> dict:
        return {
            "schema_version": "chunk.v1",
            "doc_id": chunk.doc_id,
            "chunk_id": chunk.chunk_id,
            "chunk_index": chunk.chunk_index,
            "source_file": chunk.source_file,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "section_title": chunk.section_title or "",
            "chunk_type": chunk.chunk_type.value,
            "content_hash": chunk.to_record()["content_hash"],
        }

    @staticmethod
    def _chunk_from_metadata(metadata: dict, text: str) -> Chunk:
        return Chunk(
            doc_id=metadata["doc_id"],
            chunk_id=metadata["chunk_id"],
            text=text,
            source_file=metadata["source_file"],
            page_start=int(metadata["page_start"]),
            page_end=int(metadata["page_end"]),
            section_title=metadata.get("section_title") or None,
            chunk_type=metadata["chunk_type"],
            chunk_index=int(metadata["chunk_index"]),
        )

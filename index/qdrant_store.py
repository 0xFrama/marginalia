import uuid
from typing import Any

from models.embedding import EmbeddingRecord
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from models import Chunk, RetrievalHit
from index import load_embed_model, embed_text


class QdrantStore:
    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "chunks",
        vector_size: int = 384,
        model: Any | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.client = QdrantClient(url=url)
        self.model = model or load_embed_model()

        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def add_embeddings(self, records: list[EmbeddingRecord]) -> None:
        if not records:
            return

        ids = [self._point_id(record.chunk.chunk_id) for record in records]
        embeddings = [record.embedding for record in records]
        metadatas = [self._chunk_metadata(record.chunk) for record in records]

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(id=idx, vector=embedding, payload=meta)
                for idx, embedding, meta in zip(ids, embeddings, metadatas)
            ],
        )

    def search(self, query_text: str, top_k: int = 5) -> list[RetrievalHit]:
        query_vector = embed_text(query_text, model=self.model)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        hits: list[RetrievalHit] = []

        for rank, hit in enumerate(results.points, start=1):
            chunk = self._chunk_from_metadata(hit.payload)
            hits.append(
                RetrievalHit(
                    query_text=query_text, chunk=chunk, score=hit.score, rank=rank
                )
            )

        return hits

    def get_by_chunk_id(self, chunk_id: str) -> Chunk | None:
        results = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[self._point_id(chunk_id)],
            with_payload=True,
            with_vectors=False,
        )

        if not results:
            return None

        return self._chunk_from_metadata(results[0].payload)

    @staticmethod
    def _chunk_from_metadata(metadata: dict) -> Chunk:
        return Chunk(
            doc_id=metadata["doc_id"],
            chunk_id=metadata["chunk_id"],
            text=metadata["text"],
            source_file=metadata["source_file"],
            page_start=int(metadata["page_start"]),
            page_end=int(metadata["page_end"]),
            section_title=metadata.get("section_title") or None,
            chunk_type=metadata["chunk_type"],
            chunk_index=int(metadata["chunk_index"]),
        )

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    @staticmethod
    def _chunk_metadata(chunk: Chunk) -> dict:
        return {
            "schema_version": "chunk.v1",
            "doc_id": chunk.doc_id,
            "chunk_id": chunk.chunk_id,
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
            "source_file": chunk.source_file,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "section_title": chunk.section_title or "",
            "chunk_type": chunk.chunk_type.value,
            "content_hash": chunk.to_record()["content_hash"],
        }

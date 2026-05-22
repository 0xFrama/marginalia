import uuid
from typing import Any

from models.chunk import ChunkType
from models.embedding import EmbeddingRecord
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, PointStruct, SparseVector, VectorParams

from models import Chunk, RetrievalHit
from index import load_embed_model, embed_text
from index.embeddings import sparse_embed_text

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"


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
            self._create_collection(vector_size)
            self.has_sparse = True
        else:
            self.has_sparse = self._detect_sparse_support()

    def _create_collection(self, vector_size: int) -> None:
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={DENSE_VECTOR: VectorParams(size=vector_size, distance=Distance.COSINE)},
            sparse_vectors_config={SPARSE_VECTOR: models.SparseVectorParams()},
        )

    def _detect_sparse_support(self) -> bool:
        info = self.client.get_collection(self.collection_name)
        sparse = info.config.params.sparse_vectors
        return sparse is not None and SPARSE_VECTOR in sparse

    def add_embeddings(self, records: list[EmbeddingRecord]) -> None:
        if not records:
            return

        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="chunk_type",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

        points = []
        for record in records:
            point_id = self._point_id(record.chunk.chunk_id)
            meta = self._chunk_metadata(record.chunk)

            if self.has_sparse and record.sparse_indices is not None:
                vector = {
                    DENSE_VECTOR: record.embedding,
                    SPARSE_VECTOR: SparseVector(
                        indices=record.sparse_indices,
                        values=record.sparse_values,
                    ),
                }
            else:
                vector = record.embedding

            points.append(PointStruct(id=point_id, vector=vector, payload=meta))

        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(
        self,
        query_text: str,
        chunk_types: list[ChunkType] | None = None,
        top_k: int = 5,
        prefetch_k: int = 20,
    ) -> list[RetrievalHit]:
        q_filter = self._chunk_type_filter(chunk_types)

        if self.has_sparse:
            results = self._search_hybrid(query_text, q_filter, top_k, prefetch_k)
        else:
            results = self._search_dense(query_text, q_filter, top_k)

        hits: list[RetrievalHit] = []
        for rank, hit in enumerate(results.points, start=1):
            chunk = self._chunk_from_metadata(hit.payload)
            hits.append(RetrievalHit(query_text=query_text, chunk=chunk, score=hit.score, rank=rank))

        return hits

    def _search_hybrid(
        self,
        query_text: str,
        q_filter: models.Filter | None,
        top_k: int,
        prefetch_k: int,
    ):
        dense_vector = embed_text(query_text, model=self.model)
        sparse_indices, sparse_values = sparse_embed_text(query_text)

        return self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using=DENSE_VECTOR,
                    filter=q_filter,
                    limit=prefetch_k,
                ),
                models.Prefetch(
                    query=SparseVector(indices=sparse_indices, values=sparse_values),
                    using=SPARSE_VECTOR,
                    filter=q_filter,
                    limit=prefetch_k,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

    def _search_dense(
        self,
        query_text: str,
        q_filter: models.Filter | None,
        top_k: int,
    ):
        query_vector = embed_text(query_text, model=self.model)
        return self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=q_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

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
    def _chunk_type_filter(chunk_types: list[ChunkType] | None) -> models.Filter | None:
        if not chunk_types:
            return None

        return models.Filter(
            should=[
                models.FieldCondition(
                    key="chunk_type", match=models.MatchValue(value=chunk_type.value)
                )
                for chunk_type in chunk_types
            ]
        )

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

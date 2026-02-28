"""
core/vector_store.py
Manages a ChromaDB collection that stores CLIP embeddings alongside
document metadata.  Replaces the FAISS vector store from the original notebook.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import chromadb
import numpy as np
from chromadb.config import Settings
from langchain_core.documents import Document

from config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR


@dataclass
class RetrievedDoc:
    """Lightweight wrapper returned by similarity search."""

    page_content: str
    metadata: dict[str, Any]
    distance: float = 0.0


class ChromaVectorStore:
    """Thin wrapper around a ChromaDB collection for multimodal embeddings."""

    def __init__(
        self,
        persist_directory: str = CHROMA_PERSIST_DIR,
        collection_name: str = CHROMA_COLLECTION_NAME,
        embedding_dim: int = 512,
    ) -> None:
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim

        self._client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ── write ─────────────────────────────────────────────────────────────────
    def add_documents(
        self,
        docs: list[Document],
        embeddings: list[np.ndarray],
    ) -> None:
        """Insert documents with their precomputed embeddings."""
        if len(docs) != len(embeddings):
            raise ValueError("docs and embeddings must have the same length.")

        ids = [str(uuid.uuid4()) for _ in docs]
        documents = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        embedding_list = [emb.tolist() for emb in embeddings]

        self._collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embedding_list,
        )

    def clear(self) -> None:
        """Delete and recreate the collection (useful between sessions)."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ── read ──────────────────────────────────────────────────────────────────
    def similarity_search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> list[RetrievedDoc]:
        """Return the top-k most similar documents for a query embedding."""
        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(k, self._collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )

        retrieved: list[RetrievedDoc] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            retrieved.append(RetrievedDoc(page_content=doc, metadata=meta, distance=dist))
        return retrieved

    def count(self) -> int:
        return self._collection.count()
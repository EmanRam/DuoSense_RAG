"""
core/retriever.py
Handles retrieval from ChromaDB and answer generation via the LLM.
"""

from __future__ import annotations

import os

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

import config
from core.embedder import CLIPEmbedder
from core.vector_store import ChromaVectorStore, RetrievedDoc


class MultimodalRetriever:
    """Retrieves relevant context and generates answers using an LLM."""

    def __init__(
        self,
        embedder: CLIPEmbedder,
        vector_store: ChromaVectorStore,
        image_data_store: dict[str, str],
        top_k: int = config.TOP_K,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.image_data_store = image_data_store
        self.top_k = top_k

        # configure OpenAI-compatible endpoint
        os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
        os.environ["OPENAI_API_BASE"] = config.OPENAI_API_BASE

        self.llm = init_chat_model(
            model=config.LLM_MODEL,
            max_tokens=100,
        )

    # ── public API ────────────────────────────────────────────────────────────
    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedDoc]:
        """Embed the query and return the top-k most relevant documents."""
        k = k or self.top_k
        query_embedding = self.embedder.embed_text(query)
        return self.vector_store.similarity_search(query_embedding, k=k)

    def answer(self, query: str) -> tuple[str, list[RetrievedDoc]]:
        """Full RAG pipeline: retrieve → build message → generate answer.

        Returns:
            (answer_text, retrieved_docs)
        """
        docs = self.retrieve(query)
        message = self._build_message(query, docs)
        response = self.llm.invoke([message])
        return response.content, docs

    # ── private helpers ───────────────────────────────────────────────────────
    def _build_message(self, query: str, docs: list[RetrievedDoc]) -> HumanMessage:
        """Construct a multimodal HumanMessage combining text and images."""
        content: list[dict] = []

        content.append({"type": "text", "text": f"Question: {query}\n\nContext:\n"})

        text_docs = [d for d in docs if d.metadata.get("type") == "text"]
        image_docs = [d for d in docs if d.metadata.get("type") == "image"]

        if text_docs:
            text_context = "\n\n".join(
                f"[Page {d.metadata['page']}]: {d.page_content}" for d in text_docs
            )
            content.append({"type": "text", "text": f"Text excerpts:\n{text_context}\n"})

        for doc in image_docs:
            image_id = doc.metadata.get("image_id")
            if image_id and image_id in self.image_data_store:
                content.append(
                    {"type": "text", "text": f"\n[Image from page {doc.metadata['page']}]:\n"}
                )
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{self.image_data_store[image_id]}"
                        },
                    }
                )

        content.append(
            {
                "type": "text",
                "text": "\n\nPlease answer the question based on the provided text and images.",
            }
        )

        return HumanMessage(content=content)
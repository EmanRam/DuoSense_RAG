"""
core/pdf_processor.py
Parses a PDF file, extracts text chunks and images, embeds them with CLIP,
and loads everything into the ChromaDB vector store.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image

from config import CHUNK_OVERLAP, CHUNK_SIZE
from core.embedder import CLIPEmbedder
from core.vector_store import ChromaVectorStore


class PDFProcessor:
    """Extracts text and images from a PDF and indexes them in ChromaDB."""

    def __init__(
        self,
        embedder: CLIPEmbedder,
        vector_store: ChromaVectorStore,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        # In-memory store mapping image_id → base64 string (for LLM vision)
        self.image_data_store: dict[str, str] = {}

    # ── public API ────────────────────────────────────────────────────────────
    def process(self, pdf_path: str | Path) -> None:
        """Full pipeline: parse → embed → store.

        Clears the vector store before indexing so re-uploads start fresh.
        """
        self.image_data_store.clear()
        self.vector_store.clear()

        all_docs: list[Document] = []
        all_embeddings = []

        pdf_path = Path(pdf_path)
        doc = fitz.open(str(pdf_path))

        try:
            for page_idx, page in enumerate(doc):
                text_docs, text_embs = self._process_text(page, page_idx)
                all_docs.extend(text_docs)
                all_embeddings.extend(text_embs)

                img_docs, img_embs = self._process_images(doc, page, page_idx)
                all_docs.extend(img_docs)
                all_embeddings.extend(img_embs)
        finally:
            doc.close()

        if all_docs:
            self.vector_store.add_documents(all_docs, all_embeddings)

    # ── private helpers ───────────────────────────────────────────────────────
    def _process_text(
        self, page: fitz.Page, page_idx: int
    ) -> tuple[list[Document], list]:
        """Split and embed all text on a single page."""
        text = page.get_text()
        if not text.strip():
            return [], []

        temp_doc = Document(
            page_content=text,
            metadata={"page": page_idx, "type": "text"},
        )
        chunks = self.splitter.split_documents([temp_doc])
        embeddings = [self.embedder.embed_text(chunk.page_content) for chunk in chunks]
        return chunks, embeddings

    def _process_images(
        self, doc: fitz.Document, page: fitz.Page, page_idx: int
    ) -> tuple[list[Document], list]:
        """Extract, embed, and store all images on a single page."""
        img_docs: list[Document] = []
        img_embeddings = []

        for img_idx, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                image_id = f"page_{page_idx}_img_{img_idx}"

                # Store base64 for GPT-4V vision calls
                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                self.image_data_store[image_id] = base64.b64encode(
                    buffered.getvalue()
                ).decode()

                embedding = self.embedder.embed_image(pil_image)
                img_embeddings.append(embedding)

                img_doc = Document(
                    page_content=f"[Image: {image_id}]",
                    metadata={"page": page_idx, "type": "image", "image_id": image_id},
                )
                img_docs.append(img_doc)

            except Exception as exc:
                print(f"Warning: could not process image {img_idx} on page {page_idx}: {exc}")

        return img_docs, img_embeddings
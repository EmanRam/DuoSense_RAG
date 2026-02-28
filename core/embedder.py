"""
core/embedder.py
Wraps the CLIP model for unified text and image embedding.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from config import CLIP_MODEL_NAME


class CLIPEmbedder:
    """Produces L2-normalised embeddings for both text and images using CLIP."""

    def __init__(self, model_name: str = CLIP_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model: CLIPModel | None = None
        self._processor: CLIPProcessor | None = None

    # ── lazy loading ──────────────────────────────────────────────────────────
    @property
    def model(self) -> CLIPModel:
        if self._model is None:
            self._model = CLIPModel.from_pretrained(self.model_name)
            self._model.eval()
        return self._model

    @property
    def processor(self) -> CLIPProcessor:
        if self._processor is None:
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
        return self._processor

    # ── public API ────────────────────────────────────────────────────────────
    def embed_text(self, text: str) -> np.ndarray:
        """Return a normalised 1-D CLIP text embedding."""
        inputs = self.processor(
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model.config.text_config.max_position_embeddings,
        )
        with torch.no_grad():
            text_outputs = self.model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            pooled = text_outputs.pooler_output
            features = self.model.text_projection(pooled)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().cpu().numpy()

    def embed_image(self, image: Image.Image | str) -> np.ndarray:
        """Return a normalised 1-D CLIP image embedding.

        Args:
            image: A PIL Image or a path to an image file.
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            vision_outputs = self.model.vision_model(
                pixel_values=inputs["pixel_values"]
            )
            pooled = vision_outputs.pooler_output
            features = self.model.visual_projection(pooled)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().cpu().numpy()

    def embedding_dimension(self) -> int:
        """Return the dimension of produced embeddings."""
        return self.model.config.projection_dim
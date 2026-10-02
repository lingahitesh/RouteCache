"""
Query embedding using sentence-transformers.

Loads a MiniLM-class model (CPU, ~80MB, free) and produces 384-dim
normalized vectors suitable for cosine similarity via FAISS IndexFlatIP.

First call downloads the model from HuggingFace — expect ~80MB.
Subsequent calls use the cached model.

Traceability: FR-2.1
Build Plan: Step 6
"""

import time
import numpy as np
from sentence_transformers import SentenceTransformer

from config.config import load_config


class Embedder:
    """
    Singleton-style embedding model wrapper.

    Usage:
        embedder = Embedder()                      # loads model
        vector = embedder.embed("How do I sort?")   # 384-dim numpy array
        vectors = embedder.embed_batch(["q1", "q2"])  # (2, 384) numpy array
    """

    def __init__(self, model_name: str | None = None, normalize: bool = True):
        """
        Args:
            model_name: HuggingFace model ID. Defaults to config value.
            normalize: L2-normalize output vectors. Must be True if using
                       IndexFlatIP for cosine similarity.
        """
        if model_name is None:
            config = load_config()
            model_name = config.embedding.model_name
            normalize = config.embedding.normalize

        self.model_name = model_name
        self.normalize = normalize
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single query string.

        Returns:
            1-D numpy array of shape (dimension,), dtype float32.
            L2-normalized if self.normalize is True.
        """
        vector = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        return vector.astype(np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Embed a batch of queries.

        Returns:
            2-D numpy array of shape (len(texts), dimension), dtype float32.
        """
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            batch_size=32,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)

    def embed_timed(self, text: str) -> tuple[np.ndarray, float]:
        """
        Embed and return (vector, latency_ms).
        Used in the request path to populate latency_embed_ms.
        """
        start = time.time()
        vector = self.embed(text)
        latency_ms = (time.time() - start) * 1000
        return vector, latency_ms
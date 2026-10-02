"""
FAISS index wrapper for RouteCache's semantic cache.

Uses IndexFlatIP (inner product on normalized vectors = cosine similarity).
This is the recall@k baseline for later ANN tuning (FR-8.5).

IMPORTANT: This is the retrieval layer only. It does NOT make caching
decisions — that's CacheManager's job (Step 7). The rest of the system
never calls this directly; it goes through CacheManager.lookup().

Traceability: FR-2.1, FR-8.5
Build Plan: Step 6
"""

import time
import numpy as np
import faiss
from pathlib import Path


class FaissIndex:
    """
    Flat inner-product index. Exact search, no approximation.

    Why flat (not HNSW/IVF): Build Plan Part C says "build and measure
    against IndexFlat first; only invest in ANN tuning once you have a
    recall@k baseline to improve against." Flat is the baseline.

    Usage:
        index = FaissIndex(dimension=384)
        index.add(entry_id=1, vector=np.array([...]))
        results = index.search(query_vector, top_k=5)
        # results = [(entry_id, similarity_score), ...]
    """

    def __init__(self, dimension: int = 384):
        """
        Args:
            dimension: Vector dimensionality. Must match embedding model output.
                       384 for all-MiniLM-L6-v2.
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)

        # FAISS doesn't store external IDs natively with IndexFlat.
        # We maintain a mapping: faiss_internal_position -> our entry_id.
        self._id_map: list[int] = []

    @property
    def size(self) -> int:
        """Number of vectors currently in the index."""
        return self.index.ntotal

    def add(self, entry_id: int, vector: np.ndarray) -> None:
        """
        Add a single vector to the index.

        Args:
            entry_id: Our cache entry ID (from SQLite cache_entries.id).
            vector:   1-D float32 array of shape (dimension,).
        """
        assert vector.shape == (self.dimension,), \
            f"Expected ({self.dimension},), got {vector.shape}"
        # FAISS expects 2D array: (1, dimension)
        self.index.add(vector.reshape(1, -1))
        self._id_map.append(entry_id)

    def add_batch(self, entry_ids: list[int], vectors: np.ndarray) -> None:
        """
        Add multiple vectors at once (faster than one-by-one).

        Args:
            entry_ids: List of cache entry IDs, length N.
            vectors:   2-D float32 array of shape (N, dimension).
        """
        assert vectors.shape[1] == self.dimension
        assert len(entry_ids) == vectors.shape[0]
        self.index.add(vectors)
        self._id_map.extend(entry_ids)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[tuple[int, float]]:
        """
        Search for the top-k most similar vectors.

        Args:
            query_vector: 1-D float32 array of shape (dimension,).
            top_k: Number of results to return.

        Returns:
            List of (entry_id, similarity_score) tuples, sorted by
            similarity descending. Similarity is cosine similarity
            (because vectors are L2-normalized, and we use inner product).
            Returns fewer than top_k if the index has fewer entries.
        """
        if self.index.ntotal == 0:
            return []

        k = min(top_k, self.index.ntotal)
        query_2d = query_vector.reshape(1, -1)
        scores, indices = self.index.search(query_2d, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:   # FAISS returns -1 for empty slots
                continue
            entry_id = self._id_map[idx]
            results.append((entry_id, float(score)))

        return results

    def search_timed(self, query_vector: np.ndarray, top_k: int = 5) -> tuple[list[tuple[int, float]], float]:
        """Search and return (results, latency_ms)."""
        start = time.time()
        results = self.search(query_vector, top_k)
        latency_ms = (time.time() - start) * 1000
        return results, latency_ms

    def remove(self, entry_id: int) -> bool:
        """
        Remove a vector by entry_id.

        NOTE: IndexFlatIP doesn't support direct removal. We mark the
        ID as deleted and rebuild periodically. For a research prototype
        at <=100k entries, this is fine. For production, use IndexIDMap.

        Returns True if the entry was found and marked for removal.
        """
        if entry_id in self._id_map:
            idx = self._id_map.index(entry_id)
            self._id_map[idx] = -1   # mark as deleted
            return True
        return False

    def save(self, path: Path | str | None = None) -> None:
        """Persist the FAISS index to disk (FR-2.8)."""
        if path is None:
            from app.utils.paths import get_data_dir
            path = get_data_dir() / "faiss_index.bin"
        path = Path(path) if not isinstance(path, Path) else path
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        # Save ID map alongside
        import json
        with open(str(path) + ".ids.json", "w") as f:
            json.dump(self._id_map, f)

    def load(self, path: Path | str | None = None) -> None:
        """Load a persisted FAISS index from disk."""
        if path is None:
            from app.utils.paths import get_data_dir
            path = get_data_dir() / "faiss_index.bin"
        path = Path(path) if not isinstance(path, Path) else path
        self.index = faiss.read_index(str(path))
        import json
        with open(path + ".ids.json", "r") as f:
            self._id_map = json.load(f)
"""
CacheManager — the single entry point for all cache operations.

This is the module the rest of the system calls. It composes:
  - Embedder (Step 6) for query embedding
  - FaissIndex (Step 6) for vector search
  - MetadataStore (Step 7) for entry CRUD

Callers NEVER touch FaissIndex or MetadataStore directly.
This is the concrete enforcement of the "FAISS is not the cache" principle
from the System Design Document.

Traceability: FR-2.2, FR-2.3, FR-2.4, FR-2.5, FR-2.11
Build Plan: Step 7
"""

import time
import hashlib
import numpy as np
from datetime import datetime

from app.db.connection import get_db
from config.config import load_config
from app.embeddings.embedder import Embedder
from app.cache.faiss_index import FaissIndex
from app.cache.metadata_store import MetadataStore
from app.cache.models import CacheCandidate, Provenance, RequestContext


def _context_hash(context: RequestContext) -> str:
    """Deterministic hash of context for cache key matching (FR-2.4)."""
    content = f"{context.system_prompt or ''}|{context.temperature}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def cache_zone(similarity: float, t_low: float, t_high: float) -> str:
    """
    The three-zone function from Detailed Design Document Section 4.3.

    Args:
        similarity: Cosine similarity between new query and cached query.
        t_low:  Lower threshold (below = MISS).
        t_high: Upper threshold (above = SERVE).

    Returns:
        "SERVE"  — similarity >= t_high, direct hit
        "VERIFY" — t_low <= similarity < t_high, needs verification
        "MISS"   — similarity < t_low, no usable candidate
    """
    if similarity >= t_high:
        return "SERVE"
    elif similarity >= t_low:
        return "VERIFY"
    else:
        return "MISS"


class CacheManager:
    """
    Semantic cache orchestrator.

    Usage:
        manager = CacheManager()

        # Lookup
        candidate, embed_ms, faiss_ms = manager.lookup(query, context)
        if candidate:
            zone = cache_zone(candidate.similarity, config.cache.t_low, config.cache.t_high)

        # Store after a model call
        manager.store(query, answer, context, provenance)

        # Evict if over capacity
        manager.evict_if_needed()
    """

    def __init__(self):
        config = load_config()
        self.config = config
        self.embedder = Embedder()
        self.index = FaissIndex(dimension=config.embedding.dimension)
        self.metadata = MetadataStore()
        
        # Initialize _next_embedding_id from database to avoid conflicts
        db = get_db()
        max_id_row = db.execute("SELECT MAX(embedding_id) FROM cache_entries").fetchone()
        self._next_embedding_id = (max_id_row[0] + 1) if max_id_row[0] is not None else 0
        
        # Rebuild FAISS index from existing database entries
        self._rebuild_index_from_db()

    def lookup(
        self,
        query: str,
        context: RequestContext,
    ) -> tuple[CacheCandidate | None, float, float]:
        """
        Look up the best matching cache entry for a query.

        Implements the full lookup pipeline from Detailed Design Doc Section 4.3:
          1. Embed the query
          2. FAISS search for top-k candidates
          3. Filter by context compatibility (FR-2.4)
          4. Return the best match (highest similarity)

        Args:
            query:   The user's query text.
            context: System prompt, temperature, etc. for context matching.

        Returns:
            (candidate, embed_latency_ms, faiss_latency_ms)
            candidate is None if no match found or index is empty.
        """
        # ── Embed ──
        vector, embed_ms = self.embedder.embed_timed(query)

        # ── FAISS search ──
        results, faiss_ms = self.index.search_timed(vector, top_k=self.config.cache.top_k)

        if not results:
            return None, embed_ms, faiss_ms

        # ── Retrieve metadata for all candidates ──
        entry_ids = [eid for eid, _ in results]
        candidates = self.metadata.get_by_ids(entry_ids)

        if not candidates:
            return None, embed_ms, faiss_ms

        # ── Filter by context compatibility (FR-2.4) ──
        ctx_hash = _context_hash(context)
        compatible = []
        for entry_id, score in results:
            if entry_id not in candidates:
                continue
            c = candidates[entry_id]
            # Context check: same system prompt hash and temperature
            if c.system_context_hash is not None and c.system_context_hash != ctx_hash:
                continue
            c.similarity = score
            compatible.append(c)

        if not compatible:
            return None, embed_ms, faiss_ms

        # ── Pick the best match ──
        best = max(compatible, key=lambda c: c.similarity)

        # Touch for LRU/LFU tracking
        self.metadata.touch(best.entry_id)

        return best, embed_ms, faiss_ms

    def store(
        self,
        query: str,
        answer: str,
        context: RequestContext,
        provenance: Provenance,
        has_temporal_risk: bool = False,
    ) -> int:
        """
        Store a new (query, answer) pair in the cache.

        Args:
            query:   The query text.
            answer:  The model's answer.
            context: Request context for cache key.
            provenance: Which model produced this (FR-2.11).
            has_temporal_risk: If True, uses shorter TTL (temporal queries).

        Returns:
            The new cache entry ID.
        """
        # ── Embed ──
        vector = self.embedder.embed(query)

        # ── Assign embedding ID ──
        embedding_id = self._next_embedding_id
        self._next_embedding_id += 1

        # ── Choose TTL ──
        ttl = (self.config.cache.ttl_seconds_temporal if has_temporal_risk
               else self.config.cache.ttl_seconds)

        # ── Store metadata in SQLite ──
        entry_id = self.metadata.put(
            query=query,
            answer=answer,
            embedding_id=embedding_id,
            context=context,
            provenance=provenance,
            ttl_seconds=ttl,
        )

        # ── Add vector to FAISS ──
        self.index.add(entry_id=entry_id, vector=vector)

        # ── Evict if needed ──
        self.evict_if_needed()

        return entry_id

    def evict_if_needed(self) -> int:
        """
        Evict entries if the cache exceeds max_entries.
        Returns the number of entries evicted.
        """
        count = self.metadata.count()
        if count <= self.config.cache.max_entries:
            return 0

        n_to_evict = count - self.config.cache.max_entries
        candidates = self.metadata.candidates_for_eviction(
            self.config.cache.eviction, n_to_evict
        )

        for entry_id in candidates:
            self.metadata.delete(entry_id)
            self.index.remove(entry_id)

        return len(candidates)

    def invalidate_by_tier(self, tier: str) -> int:
        """Invalidate all entries from a specific model tier (FR-2.12)."""
        return self.metadata.invalidate_by_tier(tier)

    def save(self) -> None:
        """Persist FAISS index to disk (FR-2.8)."""
        self.index.save()

    def load(self) -> None:
        """Load persisted FAISS index from disk."""
        self.index.load()
    
    def _rebuild_index_from_db(self) -> None:
        """Rebuild the in-memory FAISS index from existing database entries."""
        db = get_db()
        rows = db.execute(
            "SELECT id, query FROM cache_entries ORDER BY embedding_id"
        ).fetchall()
        
        if not rows:
            return  # No entries to rebuild
        
        # Re-embed all queries and add to FAISS
        for entry_id, query in rows:
            vector = self.embedder.embed(query)
            self.index.add(entry_id=entry_id, vector=vector)
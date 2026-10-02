"""
SQLite-backed metadata store for cache entries.

Handles all CRUD operations on the cache_entries table.
The FAISS index stores vectors; this stores everything else.

Traceability: FR-2.3, FR-2.4, FR-2.5, FR-2.11, FR-2.12
Build Plan: Step 7
"""

import hashlib
from datetime import datetime, timedelta

from app.db.connection import get_db
from app.cache.models import CacheCandidate, Provenance, RequestContext


def _hash_context(system_prompt: str | None, temperature: float) -> str:
    """Hash system prompt + temperature for context-aware cache keys (FR-2.4)."""
    content = f"{system_prompt or ''}|{temperature}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


class MetadataStore:
    """
    CRUD operations on the cache_entries SQLite table.

    Usage:
        store = MetadataStore()
        entry_id = store.put(query, answer, embedding_id, context, provenance, ttl)
        candidate = store.get(entry_id)
        store.touch(entry_id)
        store.delete(entry_id)
    """

    def put(
        self,
        query: str,
        answer: str,
        embedding_id: int,
        context: RequestContext,
        provenance: Provenance,
        ttl_seconds: int,
    ) -> int:
        """
        Store a new cache entry. Returns the new entry_id.

        Args:
            query: The original query text.
            answer: The model's answer.
            embedding_id: Matches the FAISS internal position.
            context: Request context for cache key matching.
            provenance: Which model produced this answer (FR-2.11).
            ttl_seconds: Time-to-live in seconds.
        """
        db = get_db()
        now = datetime.utcnow()
        expires = now + timedelta(seconds=ttl_seconds)
        context_hash = _hash_context(context.system_prompt, context.temperature)

        cursor = db.execute("""
            INSERT INTO cache_entries (
                query, embedding_id, answer,
                source_model, route_decision, from_cache,
                system_context_hash, temperature,
                created_at, last_accessed_at, access_count,
                verification_status, ttl_expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 'unverified', ?)
        """, (
            query, embedding_id, answer,
            provenance.source_model, provenance.route_decision, provenance.from_cache,
            context_hash, context.temperature,
            now.isoformat(), now.isoformat(),
            expires.isoformat(),
        ))
        db.commit()
        return cursor.lastrowid

    def get(self, entry_id: int) -> CacheCandidate | None:
        """
        Retrieve a cache entry by ID.
        Returns None if not found or if TTL has expired.
        """
        db = get_db()
        row = db.execute(
            "SELECT * FROM cache_entries WHERE id = ?", (entry_id,)
        ).fetchone()

        if row is None:
            return None

        # Check TTL expiry
        if row[13] is not None:   # ttl_expires_at
            expires = datetime.fromisoformat(row[13])
            if datetime.utcnow() > expires:
                self.delete(entry_id)
                return None

        return CacheCandidate(
            entry_id=row[0],
            query=row[1],
            answer=row[3],
            similarity=0.0,        # set by caller after FAISS search
            provenance=Provenance(
                source_model=row[4],
                route_decision=row[5],
                from_cache=bool(row[6]),
                created_at=datetime.fromisoformat(row[9]),
            ),
            system_context_hash=row[7],
            temperature=row[8],
        )

    def get_by_ids(self, entry_ids: list[int]) -> dict[int, CacheCandidate]:
        """Batch retrieval — used after FAISS search returns multiple IDs."""
        results = {}
        for eid in entry_ids:
            candidate = self.get(eid)
            if candidate is not None:
                results[eid] = candidate
        return results

    def touch(self, entry_id: int) -> None:
        """Update last_accessed_at and increment access_count (for LRU/LFU)."""
        db = get_db()
        db.execute("""
            UPDATE cache_entries
            SET last_accessed_at = ?, access_count = access_count + 1
            WHERE id = ?
        """, (datetime.utcnow().isoformat(), entry_id))
        db.commit()

    def update_verification_status(self, entry_id: int, status: str) -> None:
        """Mark an entry as 'safe' or 'unsafe' after verification (Step 8)."""
        db = get_db()
        db.execute(
            "UPDATE cache_entries SET verification_status = ? WHERE id = ?",
            (status, entry_id)
        )
        db.commit()

    def delete(self, entry_id: int) -> None:
        """Delete a single cache entry."""
        db = get_db()
        db.execute("DELETE FROM cache_entries WHERE id = ?", (entry_id,))
        db.commit()

    def invalidate_by_tier(self, tier: str) -> int:
        """
        Delete all entries produced by a specific model tier (FR-2.12).
        Returns count of entries deleted.
        """
        db = get_db()
        cursor = db.execute(
            "DELETE FROM cache_entries WHERE source_model = ?", (tier,)
        )
        db.commit()
        return cursor.rowcount

    def count(self) -> int:
        """Total number of cache entries."""
        db = get_db()
        row = db.execute("SELECT COUNT(*) FROM cache_entries").fetchone()
        return row[0]

    def candidates_for_eviction(self, policy: str, n: int) -> list[int]:
        """
        Return entry_ids of the n best candidates for eviction.

        Args:
            policy: "lru" (least recently used) or "lfu" (least frequently used)
            n: Number of entries to evict.
        """
        db = get_db()
        if policy == "lru":
            rows = db.execute(
                "SELECT id FROM cache_entries ORDER BY last_accessed_at ASC LIMIT ?",
                (n,)
            ).fetchall()
        elif policy == "lfu":
            rows = db.execute(
                "SELECT id FROM cache_entries ORDER BY access_count ASC LIMIT ?",
                (n,)
            ).fetchall()
        else:
            raise ValueError(f"Unknown eviction policy: {policy}")
        return [row[0] for row in rows]

    def sweep_expired(self) -> int:
        """
        Delete all entries past their TTL. Run periodically, not per-request.
        Returns count of entries deleted.
        """
        db = get_db()
        now = datetime.utcnow().isoformat()
        cursor = db.execute(
            "DELETE FROM cache_entries WHERE ttl_expires_at IS NOT NULL AND ttl_expires_at < ?",
            (now,)
        )
        db.commit()
        return cursor.rowcount
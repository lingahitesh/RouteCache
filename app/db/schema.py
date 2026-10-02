"""
SQLite schema for RouteCache.

Two tables for now:
  - request_logs:  every request that hits the proxy (Step 5)
  - cache_entries: the semantic cache metadata store (Step 7)

A third table (wrong_hit_labels) is added in Step 13 for offline evaluation.

The schema uses WAL mode for concurrent reads during writes — adequate
at the SRS's target scale. Move to Redis only if measurement shows
SQLite write contention is a bottleneck (it won't be for a research prototype).

Traceability: FR-6.4, NFR-9
Detailed Design Document: Section 5.1
Build Plan: Step 5
"""

import sqlite3
from pathlib import Path


SCHEMA_VERSION = 1

CREATE_TABLES = """
-- ─── Request Logs ────────────────────────────────────────────────────
-- Every single request that hits the proxy, with full decision trace.
-- This is the primary data source for error analysis (SRS Section 8.5).

CREATE TABLE IF NOT EXISTS request_logs (
    request_id             TEXT PRIMARY KEY,
    timestamp              TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    query_text             TEXT NOT NULL,
    query_hash             TEXT NOT NULL,
    cache_zone             TEXT,                   -- SERVE / VERIFY / MISS / NULL
    similarity             REAL,
    risk_features_json     TEXT,                   -- JSON: {"has_number": true, ...}
    risk_score             REAL,
    thresholds_json        TEXT,                   -- JSON: {"t_low": 0.82, "t_high": 0.93}
    verifier_verdict       TEXT,                   -- SAFE / UNSAFE / NULL
    verifier_score         REAL,
    route                  TEXT,                   -- small-model / large-model / cache
    router_score           REAL,
    served_entry_id        INTEGER,                -- FK to cache_entries.id if served from cache
    model_used             TEXT,                   -- actual model name called
    cost_usd               REAL NOT NULL DEFAULT 0,
    baseline_cost_usd      REAL NOT NULL DEFAULT 0,
    latency_embed_ms       REAL,
    latency_faiss_ms       REAL,
    latency_risk_ms        REAL,
    latency_verify_ms      REAL,
    latency_route_ms       REAL,
    latency_model_ms       REAL,
    latency_total_ms       REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON request_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_logs_query_hash ON request_logs(query_hash);

-- ─── Cache Entries ───────────────────────────────────────────────────
-- Metadata for every cached (query, answer) pair. The actual embedding
-- vector lives in the FAISS index, not here — this table stores everything
-- else: the text, provenance, TTL, access stats.
--
-- FR-2.11: source_model + route_decision + from_cache = full provenance.
-- FR-2.4: system_context_hash ensures different contexts don't collide.

CREATE TABLE IF NOT EXISTS cache_entries (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    query                  TEXT NOT NULL,
    embedding_id           INTEGER NOT NULL UNIQUE,  -- matches the FAISS internal ID
    answer                 TEXT NOT NULL,
    source_model           TEXT NOT NULL CHECK (source_model IN ('small', 'large')),
    route_decision         TEXT NOT NULL CHECK (route_decision IN ('small', 'large', 'cache')),
    from_cache             BOOLEAN NOT NULL DEFAULT 0,
    system_context_hash    TEXT,
    temperature            REAL,
    created_at             TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_accessed_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    access_count           INTEGER NOT NULL DEFAULT 0,
    verification_status    TEXT NOT NULL DEFAULT 'unverified'
                                CHECK (verification_status IN ('unverified', 'safe', 'unsafe')),
    ttl_expires_at         TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_cache_last_accessed ON cache_entries(last_accessed_at);
CREATE INDEX IF NOT EXISTS idx_cache_access_count ON cache_entries(access_count);
CREATE INDEX IF NOT EXISTS idx_cache_ttl ON cache_entries(ttl_expires_at);
CREATE INDEX IF NOT EXISTS idx_cache_source_model ON cache_entries(source_model);

-- ─── Wrong-Hit Labels ────────────────────────────────────────────────
-- Populated asynchronously during offline evaluation (Step 13).
-- Separate from request_logs because these labels arrive after the fact.

CREATE TABLE IF NOT EXISTS wrong_hit_labels (
    request_id             TEXT PRIMARY KEY,
    served_entry_id        INTEGER NOT NULL,
    pair_level_label       TEXT CHECK (pair_level_label IN ('equivalent', 'non_equivalent', 'unlabeled')),
    answer_level_verdict   TEXT CHECK (answer_level_verdict IN ('correct', 'wrong', 'insufficient', 'unjudged')),
    labeled_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ─── Schema Version ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def init_db(db_path: str = "data/routecache.db") -> sqlite3.Connection:
    """
    Initialize the SQLite database: create tables if they don't exist,
    enable WAL mode, check schema version.

    Args:
        db_path: Path to the SQLite file. Created if it doesn't exist.

    Returns:
        An open sqlite3.Connection with WAL mode and foreign keys enabled.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")

    # Create all tables
    conn.executescript(CREATE_TABLES)

    # Check/set schema version
    cursor = conn.execute(
        "SELECT value FROM schema_meta WHERE key = 'schema_version'"
    )
    row = cursor.fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO schema_meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),)
        )
        conn.commit()
    else:
        existing = int(row[0])
        if existing != SCHEMA_VERSION:
            raise RuntimeError(
                f"Schema version mismatch: DB has v{existing}, code expects v{SCHEMA_VERSION}. "
                f"Run migrations or delete {db_path} to recreate."
            )
    return conn
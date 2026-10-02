"""
Database connection management.

Provides a module-level singleton connection that all modules share.
Thread-safe because we set check_same_thread=False and use WAL mode.

Build Plan: Step 5
"""

from app.db.schema import init_db
from config.config import load_config

_conn = None


def get_db():
    """
    Get the singleton database connection.
    Initializes on first call (creates tables if needed).
    """
    global _conn
    if _conn is None:
        config = load_config()
        _conn = init_db(config.logging.db_path)
    return _conn
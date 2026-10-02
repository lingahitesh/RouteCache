# scripts/verify_logging.py
"""Step 5 definition of done: confirm all 30 queries show up in the DB."""

import sqlite3
from app.utils.paths import get_data_dir

conn = sqlite3.connect(get_data_dir() / "routecache.db")
conn.row_factory = sqlite3.Row

rows = conn.execute("SELECT COUNT(*) as cnt FROM request_logs").fetchone()
print(f"Total logged requests: {rows['cnt']}")
assert rows["cnt"] >= 30, f"Expected >= 30 rows, got {rows['cnt']}"

# Check fields are populated
sample = conn.execute(
    "SELECT * FROM request_logs ORDER BY timestamp DESC LIMIT 1"
).fetchone()
print(f"\nSample row:")
print(f"  request_id:     {sample['request_id']}")
print(f"  query_hash:     {sample['query_hash']}")
print(f"  route:          {sample['route']}")
print(f"  cost_usd:       {sample['cost_usd']}")
print(f"  latency_total:  {sample['latency_total_ms']:.1f} ms")

assert sample["request_id"] is not None
assert sample["cost_usd"] > 0
assert sample["latency_total_ms"] > 0

print("\nAll checks PASSED — logging is working.")
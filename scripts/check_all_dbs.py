"""Check which database files have records."""
import sqlite3
from pathlib import Path

db_paths = [
    "data/routecache.db",
    "app/data/routecache.db", 
    "scripts/data/routecache.db"
]

print("Checking all database files:")
print("=" * 60)

for db_path in db_paths:
    if Path(db_path).exists():
        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM request_logs").fetchone()[0]
        size = Path(db_path).stat().st_size
        print(f"\n{db_path}")
        print(f"  Rows: {count}")
        print(f"  Size: {size:,} bytes")
    else:
        print(f"\n{db_path}")
        print(f"  Does not exist")

print("\n" + "=" * 60)
print("\nverify_logging.py looks at: data/routecache.db")

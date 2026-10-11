"""
Properly consolidate databases by checkpointing WAL first.
"""

import shutil
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Source: the database with most records
source_db = PROJECT_ROOT / "app" / "data" / "routecache.db"
dest_db = PROJECT_ROOT / "data" / "routecache.db"

print(f"Source: {source_db.relative_to(PROJECT_ROOT)}")

# Checkpoint the WAL to consolidate all changes
conn = sqlite3.connect(source_db)
print("Checkpointing WAL...")
conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
count = conn.execute("SELECT COUNT(*) FROM request_logs").fetchone()[0]
conn.close()

print(f"Source has {count} rows after checkpoint")

# Now copy
print(f"\nCopying to {dest_db.relative_to(PROJECT_ROOT)}...")
dest_db.parent.mkdir(exist_ok=True)

# Copy the DB file
shutil.copy2(source_db, dest_db)

# Copy WAL and SHM if they exist
for ext in ["-wal", "-shm"]:
    src_file = Path(str(source_db) + ext)
    if src_file.exists():
        dest_file = Path(str(dest_db) + ext)
        shutil.copy2(src_file, dest_file)
        print(f"  Copied {src_file.name}")

# Verify destination
conn = sqlite3.connect(dest_db)
conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
dest_count = conn.execute("SELECT COUNT(*) FROM request_logs").fetchone()[0]
conn.close()

print(f"\n✓ Destination now has {dest_count} rows")

if dest_count == count:
    print("✓ Consolidation successful!")
else:
    print(f"⚠ Warning: row count mismatch ({dest_count} vs {count})")

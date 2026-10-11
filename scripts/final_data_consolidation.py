"""
Final consolidation: Copy the database with most records to correct location.
"""

import shutil
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Find which DB has the most records
dbs = [
    PROJECT_ROOT / "data" / "routecache.db",
    PROJECT_ROOT / "app" / "data" / "routecache.db",
    PROJECT_ROOT / "scripts" / "data" / "routecache.db",
]

print("Checking databases...")
max_count = 0
best_db = None

for db_path in dbs:
    if db_path.exists():
        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM request_logs").fetchone()[0]
        conn.close()
        print(f"{db_path.relative_to(PROJECT_ROOT)}: {count} rows")
        
        if count > max_count:
            max_count = count
            best_db = db_path

if best_db:
    print(f"\nBest database: {best_db.relative_to(PROJECT_ROOT)} ({max_count} rows)")
    
    correct_db = PROJECT_ROOT / "data" / "routecache.db"
    if best_db != correct_db:
        print(f"Copying to {correct_db.relative_to(PROJECT_ROOT)}...")
        correct_db.parent.mkdir(exist_ok=True)
        shutil.copy2(best_db, correct_db)
        
        # Verify
        conn = sqlite3.connect(correct_db)
        new_count = conn.execute("SELECT COUNT(*) FROM request_logs").fetchone()[0]
        conn.close()
        print(f"✓ Copied successfully: {new_count} rows")
    else:
        print("Already in correct location!")

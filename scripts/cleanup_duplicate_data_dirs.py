"""
Clean up duplicate data directories created by running scripts from wrong locations.

The correct structure is:
  RouteCache/
    ├── data/                  ← CORRECT (project root)
    │   ├── routecache.db
    │   ├── recorded_outputs/
    │   └── baseline_30_queries.json
    ├── app/
    │   └── data/              ← WRONG (created by running server from app/ dir)
    └── scripts/
        └── data/              ← WRONG (created by running scripts from scripts/ dir)

This script consolidates everything into the correct location.
"""

import shutil
from pathlib import Path
import sqlite3

# Define the correct and incorrect locations
PROJECT_ROOT = Path(__file__).parent.parent
CORRECT_DATA = PROJECT_ROOT / "data"
WRONG_DATA_DIRS = [
    PROJECT_ROOT / "app" / "data",
    PROJECT_ROOT / "scripts" / "data",
]

print("=" * 70)
print("CLEANUP: Consolidating data directories")
print("=" * 70)

# Ensure correct data dir exists
CORRECT_DATA.mkdir(exist_ok=True)
print(f"\n✓ Correct data directory: {CORRECT_DATA}")

# Check each wrong location
for wrong_dir in WRONG_DATA_DIRS:
    if not wrong_dir.exists():
        print(f"\n✓ {wrong_dir.relative_to(PROJECT_ROOT)} - does not exist")
        continue
    
    print(f"\n⚠ Found: {wrong_dir.relative_to(PROJECT_ROOT)}")
    
    # Check for database file
    db_file = wrong_dir / "routecache.db"
    if db_file.exists():
        conn = sqlite3.connect(db_file)
        count = conn.execute("SELECT COUNT(*) FROM request_logs").fetchone()[0]
        conn.close()
        
        if count > 0:
            print(f"   Database has {count} records")
            
            # Check if correct location has fewer records
            correct_db = CORRECT_DATA / "routecache.db"
            if correct_db.exists():
                correct_conn = sqlite3.connect(correct_db)
                correct_count = correct_conn.execute("SELECT COUNT(*) FROM request_logs").fetchone()[0]
                correct_conn.close()
                
                if count > correct_count:
                    print(f"   → This has MORE records than {CORRECT_DATA}/routecache.db ({correct_count})")
                    print(f"   → Moving this database to correct location")
                    shutil.copy2(db_file, correct_db)
                else:
                    print(f"   → Correct location already has {correct_count} records, keeping it")
            else:
                print(f"   → Moving to correct location")
                shutil.copy2(db_file, CORRECT_DATA / "routecache.db")
    
    # Check for other files
    other_files = [f for f in wrong_dir.iterdir() if f.name != "routecache.db"]
    if other_files:
        print(f"   Other files: {[f.name for f in other_files[:5]]}")
    
    # Ask before deletion
    print(f"   → Recommend deleting this directory after verification")

print("\n" + "=" * 70)
print("CONSOLIDATION COMPLETE")
print("=" * 70)
print(f"\nAll data should now be in: {CORRECT_DATA}")
print(f"\nYou can manually delete:")
for wrong_dir in WRONG_DATA_DIRS:
    if wrong_dir.exists():
        print(f"  - {wrong_dir}")
print("\nAfter verification that everything works.")

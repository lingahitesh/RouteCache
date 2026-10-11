"""
Inspect the RouteCache database schema and contents.
"""

import sqlite3

conn = sqlite3.connect('data/routecache.db')

print("=" * 70)
print("DATABASE SCHEMA")
print("=" * 70)

# List all tables
cursor = conn.execute("""
    SELECT name, sql 
    FROM sqlite_master 
    WHERE type='table' 
    ORDER BY name
""")

for table_name, create_sql in cursor.fetchall():
    if table_name.startswith('sqlite_'):
        continue
    print(f"\n📋 Table: {table_name}")
    print("-" * 70)
    if create_sql:
        # Show just the column definitions, not the full SQL
        cols = conn.execute(f"PRAGMA table_info({table_name})")
        for col in cols:
            cid, name, type_, notnull, default, pk = col
            pk_mark = " 🔑" if pk else ""
            null_mark = " NOT NULL" if notnull else ""
            print(f"  • {name:<25} {type_:<15}{null_mark}{pk_mark}")
    
    # Show row count
    count_cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = count_cursor.fetchone()[0]
    print(f"\n  Rows: {count}")

print("\n" + "=" * 70)
print("DATABASE READY ✓")
print("=" * 70)

"""
Quick test to verify database setup works.
Run this to create the database manually if you want to inspect it.
"""

from app.db.connection import get_db

print("Initializing database...")
conn = get_db()

# Check tables were created
cursor = conn.execute("""
    SELECT name FROM sqlite_master 
    WHERE type='table' 
    ORDER BY name
""")

tables = [row[0] for row in cursor.fetchall()]
print(f"\n✓ Database created at: data/routecache.db")
print(f"✓ Tables created: {', '.join(tables)}")

# Check schema version
cursor = conn.execute("SELECT value FROM schema_meta WHERE key='schema_version'")
version = cursor.fetchone()[0]
print(f"✓ Schema version: {version}")

print("\nDatabase is ready to use!")

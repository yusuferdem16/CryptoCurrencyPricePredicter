from sqlalchemy import text
from src.database import engine

def add_mape_column():
    print("🔄 Migrating database schema...")
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS mape FLOAT;"))
        conn.commit()
        print("✅ Column 'mape' added to 'predictions' table.")

if __name__ == "__main__":
    add_mape_column()
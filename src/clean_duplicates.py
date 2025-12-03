from sqlalchemy import text
from src.database import engine

def clean_duplicates():
    print("🧹 Cleaning duplicate predictions...")
    with engine.connect() as conn:
        # PostgreSQL magic: Keep the row with the HIGHEST id (latest), delete the rest
        query = text("""
            DELETE FROM predictions a USING predictions b
            WHERE a.id < b.id
            AND a.predicted_date = b.predicted_date
            AND a.model_version = b.model_version
            AND a.ticker = b.ticker;
        """)
        result = conn.execute(query)
        conn.commit()
        print(f"✅ Deleted {result.rowcount} duplicate rows.")

if __name__ == "__main__":
    clean_duplicates()
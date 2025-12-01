from sqlalchemy import text
from src.database import engine

def init_db():
    print("Initializing database schema...")
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ticker VARCHAR(10),
                model_version VARCHAR(50),
                predicted_date DATE,
                predicted_price FLOAT,
                actual_price FLOAT,
                mae FLOAT
            );
        """))
        conn.commit()
        print("✅ Tables 'predictions' created.")

if __name__ == "__main__":
    init_db()
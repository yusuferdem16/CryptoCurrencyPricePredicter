from sqlalchemy import text
from src.database import engine

def init_db():
    print("Initializing database schema...")
    
    with engine.connect() as conn:
        # Create Predictions Table
        # This table tracks every guess your model ever makes
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
        
        # Create Model Registry Table (Optional but Pro)
        # Keeps track of which model version is currently "live"
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS model_registry (
                version VARCHAR(50) PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                hyperparameters JSONB,
                metrics JSONB
            );
        """))
        
        conn.commit()
        print("✅ Tables 'predictions' and 'model_registry' are ready.")

if __name__ == "__main__":
    init_db()
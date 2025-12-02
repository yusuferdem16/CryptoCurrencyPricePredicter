from sqlalchemy import text
from src.database import engine

def backfill_mape():
    print("🔄 Backfilling MAPE for existing predictions...")
    
    with engine.connect() as conn:
        # SQL Magic: Calculate MAPE directly in the database
        # Formula: (MAE / Actual) * 100
        # We add a check (actual_price > 0) to avoid DivisionByZero errors
        query = text("""
            UPDATE predictions
            SET mape = (mae / actual_price) * 100
            WHERE actual_price IS NOT NULL 
              AND actual_price > 0 
              AND mape IS NULL;
        """)
        
        result = conn.execute(query)
        conn.commit()
        
        print(f"✅ Backfill complete. Updated {result.rowcount} rows.")

if __name__ == "__main__":
    backfill_mape()
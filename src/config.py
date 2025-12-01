import os
from dotenv import load_dotenv

# Load .env variables (only works locally)
load_dotenv()

class Config:
    # 1. Check if the Cloud has provided a full URL (Render/Neon usually do this)
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL")

    # 2. If not, build it from local parts (Docker)
    if not SQLALCHEMY_DATABASE_URI:
        POSTGRES_USER = os.getenv("POSTGRES_USER", "user")
        POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
        POSTGRES_DB = os.getenv("POSTGRES_DB", "crypto_data")
        POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
        POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
        
        SQLALCHEMY_DATABASE_URI = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
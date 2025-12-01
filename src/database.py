from sqlalchemy import create_engine
from src.config import Config

# Create the SQLAlchemy engine
# "echo=False" prevents it from printing every SQL query to the terminal (too noisy)

engine = create_engine(Config.SQLALCHEMY_DATABASE_URI, echo=False)

def get_db_connection():
    """Utility to get a raw connection if needed"""
    return engine.connect()
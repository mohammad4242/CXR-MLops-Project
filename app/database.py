import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

# 1. Load environment variables from the .env file
load_dotenv()

# 2. Read the username, password, and other details
# The second argument is a default value just in case the .env file is missing
DB_USERNAME = os.getenv("DB_USERNAME", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_HOST = os.getenv("DB_HOST", "db")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "cxr")

# 3. Create the database URL dynamically
DATABASE_URL = f"postgresql+psycopg://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# 4. Create the engine and session
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 5. SQLAlchemy 2.0 Base class
class Base(DeclarativeBase):
    pass

# 6. Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
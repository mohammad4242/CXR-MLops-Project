from datetime import datetime
from typing import Any

from sqlalchemy import String, Integer, Float, DateTime, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base

class PredictionRecord(Base):
    __tablename__ = "prediction_records"

    # id: Integer, Primary Key
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    
    # Metadata for the uploaded image
    original_filename: Mapped[str] = mapped_column(String(255))
    stored_filename: Mapped[str] = mapped_column(String(255), unique=True)
    file_path: Mapped[str] = mapped_column(String(500))
    content_type: Mapped[str] = mapped_column(String(100))
    file_size_bytes: Mapped[int] = mapped_column(Integer)
    
    # Metadata for the ML model and inference results
    model_name: Mapped[str] = mapped_column(String(100), default="densenet121-res224-all")
    prediction_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # max_prediction_score can be empty (nullable), so we use float | None
    max_prediction_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    # PostgreSQL JSONB is great for saving dictionaries directly
    predictions: Mapped[dict[str, Any]] = mapped_column(JSONB)
    
    # Automatically save the exact time the row is created
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now()
    )
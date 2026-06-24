from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class User(Base):
    """An application user who can authenticate and request predictions."""
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    # Email is the login identifier, so it must be unique.
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    full_name: Mapped[str | None] = mapped_column(String(120), nullable=True)

    # We only ever store the bcrypt hash, never the plaintext password.
    hashed_password: Mapped[str] = mapped_column(String(255))

    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class PredictionRecord(Base):
    __tablename__ = "prediction_records"

    # id: Integer, Primary Key
    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    # Owner of this prediction. Nullable so rows created before authentication
    # existed (and any anonymous legacy data) remain valid.
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True, index=True
    )

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

    # PostgreSQL JSONB is great for saving dictionaries directly. We persist the
    # full prediction set here, even though the API only surfaces the
    # high-confidence subset to the client. The variant keeps this portable:
    # JSONB in production (Postgres), generic JSON under SQLite (tests).
    predictions: Mapped[dict[str, Any]] = mapped_column(
        JSONB().with_variant(JSON(), "sqlite")
    )

    # Automatically save the exact time the row is created
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

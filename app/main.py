import time
from pathlib import Path
from uuid import uuid4
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware

from app.models import PredictionRecord, User
from app.database import engine, get_db, Base
from app.inference import CXRPredictor
from app.security import get_current_user
from app.routers import auth as auth_router

ml_models = {}
app_state = {"startup_time": None}
MODEL_NAME = "densenet121-res224-all"

# Only findings at or above this confidence are surfaced to the client. The full
# prediction set is still stored in the database for audit/history.
SIGNIFICANCE_THRESHOLD = 0.65
NO_SIGNIFICANT_FINDINGS_MESSAGE = (
    "No findings reached the 65% confidence threshold. The model did not identify a "
    "high-probability abnormality in this radiograph. This result is not a diagnosis — "
    "please consult a qualified clinician for interpretation."
)

# DIRECTORY SETUP: Create an 'images' folder if it doesn't exist
IMAGES_DIR = Path("images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing server...")
    app_state["startup_time"] = time.time()

    # DATABASE CREATION: create any missing tables (users, prediction_records).
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)

    # Idempotent migration: add user_id to databases that were created before
    # authentication existed. PostgreSQL's "IF NOT EXISTS" makes this safe to run
    # on every boot, and it is a best-effort step that never blocks startup.
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "ALTER TABLE prediction_records "
                    "ADD COLUMN IF NOT EXISTS user_id INTEGER REFERENCES users(id)"
                )
            )
    except Exception as exc:  # noqa: BLE001
        print(f"Skipping user_id migration: {exc}")

    ml_models["predictor"] = CXRPredictor()
    print("Model loaded successfully!")

    yield

    print("Shutting down server and cleaning up resources...")
    ml_models.clear()


app = FastAPI(
    title="Chest X-Ray API",
    description="API for analyzing Chest X-Ray images using TorchXRayVision",
    version="1.1.0",
    lifespan=lifespan,
)

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://62.60.198.132:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication routes: /auth/register, /auth/login, /auth/me
app.include_router(auth_router.router)


@app.get("/health")
async def health_check():
    uptime_seconds = round(time.time() - app_state["startup_time"], 2)
    return {
        "status": "healthy",
        "uptime_seconds": uptime_seconds,
        "model_loaded": "predictor" in ml_models,
        "model_name": MODEL_NAME,
        "version": app.version,
    }


# DEPENDENCY INJECTION: `db` gives us a database session and `current_user`
# enforces authentication — only signed-in users can run predictions.
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    file_path = None
    try:
        # Read image bytes directly into memory
        image_bytes = await file.read()
        file_size_bytes = len(image_bytes)

        # Run inference FIRST — only persist files that produce valid results
        results = ml_models["predictor"].predict(image_bytes)

        # SAVE FILE: Generate a unique name and save the image to the Linux folder
        file_extension = Path(file.filename).suffix or ".png"
        stored_filename = f"{uuid4().hex}{file_extension}"
        file_path = IMAGES_DIR / stored_filename

        with open(file_path, "wb") as image_file:
            image_file.write(image_bytes)

        # Calculate the highest prediction score for the database
        max_prediction_score = max(results.values()) if results else None

        # SAVE TO POSTGRESQL: persist the full result set, linked to the user.
        record = PredictionRecord(
            user_id=current_user.id,
            original_filename=file.filename,
            stored_filename=stored_filename,
            file_path=str(file_path),
            content_type=file.content_type,
            file_size_bytes=file_size_bytes,
            model_name=MODEL_NAME,
            prediction_count=len(results),
            max_prediction_score=max_prediction_score,
            predictions=results,
        )

        db.add(record)
        db.commit()
        db.refresh(record)  # Pull back the new ID and created_at

        # Surface only findings at/above the confidence threshold, ranked high→low.
        significant_findings = {
            pathology: score
            for pathology, score in sorted(
                results.items(), key=lambda item: item[1], reverse=True
            )
            if score >= SIGNIFICANCE_THRESHOLD
        }
        has_findings = bool(significant_findings)

        return {
            "id": record.id,
            "filename": record.original_filename,
            "success": True,
            "findings": significant_findings,
            "has_findings": has_findings,
            "message": None if has_findings else NO_SIGNIFICANT_FINDINGS_MESSAGE,
            "threshold": SIGNIFICANCE_THRESHOLD,
            "saved_at": record.created_at,
        }

    except HTTPException:
        # Let intentional 4xx responses pass through untouched.
        raise
    except Exception as e:
        db.rollback()
        # Remove the saved file if DB commit failed, so we don't accumulate orphans
        if file_path and file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

import os
import time
from pathlib import Path
from uuid import uuid4
from contextlib import asynccontextmanager

# 1. NEW IMPORTS: We need Depends for dependency injection and Session for DB typing
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session

# 2. NEW IMPORTS: Import our database engine, session getter, and the model
from app.models import PredictionRecord
from app.database import engine, get_db, Base
from app.inference import CXRPredictor

ml_models = {}
app_state = {"startup_time": None}
MODEL_NAME = "densenet121-res224-all"

# 3. DIRECTORY SETUP: Create an 'images' folder if it doesn't exist
IMAGES_DIR = Path("images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing server...")
    app_state["startup_time"] = time.time()
    
    # 4. DATABASE CREATION: This line physically creates the table in PostgreSQL!
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    
    ml_models["predictor"] = CXRPredictor()
    print("Model loaded successfully!")
    
    yield  
    
    print("Shutting down server and cleaning up resources...")
    ml_models.clear()


app = FastAPI(
    title="Chest X-Ray API",
    description="API for analyzing Chest X-Ray images using TorchXRayVision",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    uptime_seconds = round(time.time() - app_state["startup_time"], 2)
    return {
        "status": "healthy",
        "uptime_seconds": uptime_seconds,
        "model_loaded": "predictor" in ml_models,
        "model_name": MODEL_NAME,
        "version": app.version
    }

# 5. DEPENDENCY INJECTION: We added `db: Session = Depends(get_db)` to get the database session
@app.post("/predict")
async def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    try:
        # Read image bytes directly into memory
        image_bytes = await file.read()
        
        # 6. SAVE FILE: Generate a unique name and save the image to the Linux folder
        file_size_bytes = len(image_bytes)
        file_extension = Path(file.filename).suffix or ".png"
        stored_filename = f"{uuid4().hex}{file_extension}"
        file_path = IMAGES_DIR / stored_filename
        
        with open(file_path, "wb") as image_file:
            image_file.write(image_bytes)
            
        # Run inference
        results = ml_models["predictor"].predict(image_bytes)
        
        # Calculate the highest prediction score for the database
        max_prediction_score = max(results.values()) if results else None
        
        # 7. SAVE TO POSTGRESQL: Create a new PredictionRecord object
        record = PredictionRecord(
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
        
        # 8. COMMIT: Add the record to the session and save it to the database
        db.add(record)
        db.commit()
        db.refresh(record) # This updates the 'record' variable with the new ID and created_at time
        
        return {
            "id": record.id,
            "filename": record.original_filename,
            "success": True,
            "predictions": results,
            "saved_at": record.created_at
        }
    
    except Exception as e:
        # 9. ROLLBACK: If something crashes, undo the database transaction so we don't save bad data
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
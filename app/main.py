import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException

from inference import CXRPredictor

ml_models = {}
app_state = {"startup_time": None}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifespan.
    Loads the ML model into memory on startup and cleans up on shutdown.
    """
    print("Initializing server...")
    app_state["startup_time"] = time.time()
    
    # Load the model into memory
    ml_models["predictor"] = CXRPredictor()
    print("Model loaded successfully!")
    
    yield  # Yield control to the application
    
    # Clean up resources on shutdown
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
    """
    Health check endpoint for load balancers and monitoring tools.
    """
    uptime_seconds = round(time.time() - app_state["startup_time"], 2)
    return {
        "status": "healthy",
        "uptime_seconds": uptime_seconds,
        "model_loaded": "predictor" in ml_models,
        "model_name": "densenet121-res224-all",
        "version": app.version
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to receive a Chest X-Ray image and return disease predictions.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    try:
        # Read image bytes directly into memory
        image_bytes = await file.read()
        
        # Run inference using the pre-loaded model
        results = ml_models["predictor"].predict(image_bytes)
        
        return {
            "filename": file.filename,
            "success": True,
            "predictions": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

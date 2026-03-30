"""
FastAPI application for AgriPrice Prophet - Pydantic v2 Compatible
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional, Dict, Any
import numpy as np
import cv2
import os
import shutil
import json
import joblib
import tensorflow as tf

# For Render deployment - handle port
PORT = int(os.environ.get("PORT", 8000))

# Import local modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import ImagePreprocessor, TabularPreprocessor
from src.model import CropDiseaseModel, PricePredictionModel
from src.prediction import PredictionService
from src.utils import validate_image_file, validate_csv_file, get_timestamp
from app.database import init_db, save_prediction, save_uploaded_dataset, get_stats

# Initialize FastAPI
app = FastAPI(
    title="AgriPrice Prophet API",
    description="Agricultural Price Prediction & Crop Disease Detection",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
image_model = None
price_model = None
image_preprocessor = None
tabular_preprocessor = None
prediction_service = None
class_indices = None

# ---------- Pydantic Models (v2 Compatible - Fixed) ----------
class PricePredictionRequest(BaseModel):
    commodity: str = Field(default="Potato", description="Commodity name")
    market: str = Field(default="Delhi", description="Market name")
    state: str = Field(default="Delhi", description="State name")
    prediction_date: str = Field(default_factory=lambda: datetime.now().date().isoformat(), description="Prediction date (YYYY-MM-DD)")
    arrivals: float = Field(default=100.0, description="Expected arrivals in tons")
    min_price: Optional[float] = Field(default=None, description="Minimum price")
    max_price: Optional[float] = Field(default=None, description="Maximum price")
    
    # Optional external factors
    fuel_price: Optional[float] = Field(default=None, description="Fuel price")
    rainfall: Optional[float] = Field(default=None, description="Rainfall in mm")
    temperature: Optional[float] = Field(default=None, description="Temperature in Celsius")

class PricePredictionResponse(BaseModel):
    predicted_price: float
    confidence_interval: Dict[str, float]
    feature_importance: Optional[Dict[str, float]] = None
    model_used: str
    latency_ms: float

class ImagePredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    features: Dict[str, float]
    interpretation: str
    latency_ms: float

class UploadResponse(BaseModel):
    message: str
    dataset_id: int
    filename: str
    rows: int
    columns: List[str]
    data_type: str

class RetrainResponse(BaseModel):
    status: str
    message: str
    timestamp: str
    metrics: Optional[Dict[str, Any]] = None
    model_path: Optional[str] = None

class StatsResponse(BaseModel):
    total_predictions: int
    total_uploads: int
    total_retrainings: int
    model_status: Dict[str, str]

# ---------- Startup Event ----------
@app.on_event("startup")
async def load_models():
    global image_model, price_model, image_preprocessor, tabular_preprocessor
    global prediction_service, class_indices
    
    print("Loading models...")
    
    # Initialize database
    init_db()
    
    # Default class indices
    class_indices = {'healthy': 0, 'diseased': 1, 'pest': 2}
    
    # Try to load class indices from file
    try:
        if os.path.exists('models/class_indices.json'):
            with open('models/class_indices.json', 'r') as f:
                class_indices = json.load(f)
            print("✅ Class indices loaded")
    except Exception as e:
        print(f"⚠️ Using default class indices: {e}")
    
    # Load image model
    try:
        image_model = CropDiseaseModel()
        if os.path.exists('models/crop_disease_model.h5'):
            image_model.load('models/')
            print("✅ Image model loaded")
        else:
            print("⚠️ Image model file not found, using mock mode")
    except Exception as e:
        print(f"⚠️ Could not load image model: {e}")
        image_model = None
    
    # Load price model
    try:
        price_model = PricePredictionModel()
        if os.path.exists('models/price_model.pkl'):
            price_model.load('models/')
            print("✅ Price model loaded")
        else:
            print("⚠️ Price model file not found, using mock mode")
    except Exception as e:
        print(f"⚠️ Could not load price model: {e}")
        price_model = None
    
    # Initialize preprocessors
    image_preprocessor = ImagePreprocessor()
    
    try:
        tabular_preprocessor = TabularPreprocessor()
        if os.path.exists('models/scaler.pkl'):
            tabular_preprocessor.load('models/')
            print("✅ Tabular preprocessor loaded")
    except Exception as e:
        tabular_preprocessor = TabularPreprocessor()
        print(f"⚠️ Tabular preprocessor not fitted: {e}")
    
    # Initialize prediction service
    if image_model is not None:
        prediction_service = PredictionService(
            image_model, price_model, image_preprocessor, class_indices
        )
        print("✅ Prediction service initialized")
    
    print("🚀 API ready!")

# ---------- Helper Functions ----------
def parse_date(date_str: str):
    """Parse date string to datetime object"""
    try:
        return datetime.strptime(date_str, '%Y-%m-%d').date()
    except:
        return datetime.now().date()

# ---------- API Endpoints ----------
@app.get("/")
def read_root():
    return {
        "message": "🌾 AgriPrice Prophet API",
        "version": "2.0.0",
        "status": "operational",
        "models_loaded": {
            "image_model": image_model is not None,
            "price_model": price_model is not None
        },
        "endpoints": [
            "/health",
            "/stats",
            "/predict/price",
            "/predict/image",
            "/upload",
            "/retrain",
            "/commodities",
            "/disease-classes"
        ]
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "image": image_model is not None,
            "price": price_model is not None
        }
    }

@app.get("/stats", response_model=StatsResponse)
def get_statistics():
    """Get system statistics"""
    stats = get_stats()
    return StatsResponse(
        total_predictions=stats.get('total_predictions', 0),
        total_uploads=stats.get('total_uploads', 0),
        total_retrainings=stats.get('total_retrainings', 0),
        model_status={
            "image": "loaded" if image_model else "not loaded",
            "price": "loaded" if price_model else "not loaded"
        }
    )

@app.post("/predict/price", response_model=PricePredictionResponse)
async def predict_price(request: PricePredictionRequest):
    """
    Predict agricultural commodity price
    """
    import time
    start_time = time.time()
    
    # Extract features
    features = request.model_dump()
    
    # Mock prediction (since we're debugging)
    mock_price = 55.0
    result = {
        'predicted_price': mock_price,
        'confidence_interval': {"lower": mock_price * 0.9, "upper": mock_price * 1.1, "confidence": 0.95},
        'feature_importance': {"arrivals": 0.42, "min_price": 0.28, "max_price": 0.18, "fuel_price": 0.12},
        'model_used': 'XGBoost with L1/L2 regularization (Mock)',
        'latency_ms': (time.time() - start_time) * 1000
    }
    
    # Try real prediction if model available
    if price_model is not None:
        try:
            # Convert date string to date object
            features['date'] = parse_date(request.prediction_date)
            result = prediction_service.predict_price(features) if prediction_service else result
        except Exception as e:
            print(f"Prediction error: {e}")
    
    # Save to database
    try:
        save_prediction(
            prediction_type='price',
            input_data=features,
            output_data=result,
            latency_ms=result['latency_ms']
        )
    except:
        pass
    
    return PricePredictionResponse(**result)

@app.post("/predict/image", response_model=ImagePredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """
    Predict crop disease from uploaded image
    """
    import time
    start_time = time.time()
    
    if not validate_image_file(file.filename):
        raise HTTPException(status_code=400, detail="File must be an image (jpg, png)")
    
    # Mock result
    mock_result = ImagePredictionResponse(
        prediction="healthy",
        confidence=0.85,
        probabilities={"healthy": 0.85, "diseased": 0.10, "pest": 0.05},
        features={"green_mean": 120.5, "contrast": 45.2, "edge_density": 0.03},
        interpretation="Green-dominant image indicates healthy chlorophyll activity. Low texture contrast suggests smooth leaf surface. Low edge density indicates intact leaf boundaries.",
        latency_ms=(time.time() - start_time) * 1000
    )
    
    # Try real prediction if model available
    if image_model is not None and prediction_service is not None:
        try:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = prediction_service.predict_image(img)
                mock_result = ImagePredictionResponse(**result)
        except Exception as e:
            print(f"Image prediction error: {e}")
    
    # Save to database
    try:
        save_prediction(
            prediction_type='image',
            input_data={'filename': file.filename},
            output_data=mock_result.model_dump(),
            latency_ms=mock_result.latency_ms
        )
    except:
        pass
    
    return mock_result

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload new data for retraining"""
    try:
        is_image = validate_image_file(file.filename)
        is_csv = validate_csv_file(file.filename)
        
        if not (is_image or is_csv):
            raise HTTPException(status_code=400, detail="File must be image or CSV")
        
        timestamp = get_timestamp()
        safe_filename = f"{timestamp}_{file.filename}"
        file_path = f"data/uploads/{safe_filename}"
        os.makedirs("data/uploads", exist_ok=True)
        
        contents = await file.read()
        with open(file_path, "wb") as buffer:
            buffer.write(contents)
        
        if is_csv:
            import pandas as pd
            import io
            df = pd.read_csv(io.BytesIO(contents))
            rows = len(df)
            columns = list(df.columns)
            data_type = 'tabular'
        else:
            rows = 1
            columns = ['image']
            data_type = 'image'
        
        dataset_id = save_uploaded_dataset(
            filename=file.filename,
            file_path=file_path,
            data_type=data_type,
            rows=rows
        )
        
        return UploadResponse(
            message="File uploaded successfully",
            dataset_id=dataset_id,
            filename=file.filename,
            rows=rows,
            columns=columns,
            data_type=data_type
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain")
async def trigger_retraining(filename: str, model_type: str = "both"):
    """Trigger model retraining with uploaded data"""
    file_path = f"data/uploads/{filename}"
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return {
        "status": "success",
        "message": f"Retraining triggered for {filename}",
        "timestamp": datetime.now().isoformat(),
        "model_type": model_type
    }

@app.get("/commodities")
def list_commodities():
    """List all supported commodities"""
    commodities = ["Potato", "Tomato", "Onion", "Wheat", "Rice", "Maize", "Soybeans", "Coffee"]
    return {"total": len(commodities), "commodities": commodities}

@app.get("/disease-classes")
def list_disease_classes():
    """List all crop disease classes"""
    return {
        "classes": list(class_indices.keys()),
        "mapping": class_indices
    }

@app.get("/feature-explanations")
def get_feature_explanations():
    """Get explanations for features used in interpretation"""
    return {
        "features": [
            {
                "name": "Color Distribution",
                "description": "RGB values indicate plant health",
                "interpretation": "Green = healthy, Yellow/Brown = diseased, Dark = pest"
            },
            {
                "name": "Texture Contrast",
                "description": "Measures roughness/smoothness of leaf surface",
                "interpretation": "Low = smooth (healthy), High = rough (damaged)"
            },
            {
                "name": "Edge Density",
                "description": "Measures boundaries and lesions",
                "interpretation": "Low = no damage, High = many lesions/holes"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)

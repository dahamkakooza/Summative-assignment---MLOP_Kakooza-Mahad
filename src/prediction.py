"""
Prediction service module for both image and tabular models
"""

import numpy as np
import time
from typing import Dict, Any, Tuple
import json

class PredictionService:
    """Service class for making predictions"""
    
    def __init__(self, image_model, price_model, image_preprocessor, class_indices):
        self.image_model = image_model
        self.price_model = price_model
        self.image_preprocessor = image_preprocessor
        self.class_indices = class_indices
        self.idx_to_class = {v: k for k, v in class_indices.items()}
    
    def predict_image(self, image_array: np.ndarray) -> Dict[str, Any]:
        """
        Predict crop disease from image
        
        Args:
            image_array: numpy array of image
            
        Returns:
            Dictionary with prediction results including interpretation
        """
        start_time = time.time()
        
        # Preprocess image
        processed = self.image_preprocessor.preprocess_array(image_array)
        
        # Extract features for interpretation
        features = self.image_preprocessor.extract_features(image_array)
        
        # Make prediction
        class_idx, confidence, probabilities = self.image_model.predict(processed)
        
        # Get class name
        class_name = self.idx_to_class[class_idx]
        
        # Generate interpretation
        interpretation = self.image_preprocessor.get_interpretation(class_name, features)
        
        # Calculate latency
        latency = (time.time() - start_time) * 1000  # Convert to ms
        
        # Create response
        result = {
            'prediction': class_name,
            'confidence': float(confidence),
            'probabilities': {
                self.idx_to_class[i]: float(prob) 
                for i, prob in enumerate(probabilities)
            },
            'features': features,
            'interpretation': interpretation,
            'latency_ms': round(latency, 2)
        }
        
        return result
    
    def predict_price(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict commodity price from features
        
        Args:
            features: Dictionary with input features
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        # Make prediction
        predicted_price = self.price_model.predict(features)
        
        # Calculate confidence interval (simplified)
        confidence_interval = {
            'lower': round(predicted_price * 0.9, 2),
            'upper': round(predicted_price * 1.1, 2),
            'confidence': 0.95
        }
        
        # Get feature importance
        importance = self.price_model.get_feature_importance()
        
        # Calculate latency
        latency = (time.time() - start_time) * 1000
        
        result = {
            'predicted_price': round(predicted_price, 2),
            'confidence_interval': confidence_interval,
            'feature_importance': importance,
            'model_used': 'XGBoost with L1/L2 regularization',
            'latency_ms': round(latency, 2)
        }
        
        return result
    
    def predict_batch(self, images: list) -> list:
        """Predict batch of images"""
        results = []
        for img in images:
            results.append(self.predict_image(img))
        return results
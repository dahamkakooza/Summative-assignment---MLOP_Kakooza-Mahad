"""
Data preprocessing module for both image and tabular data
"""

import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
from typing import Tuple, Dict, Any
import pandas as pd

class ImagePreprocessor:
    """Preprocess images for crop disease classification"""
    
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        
    def load_and_preprocess(self, image_path: str) -> np.ndarray:
        """Load and preprocess a single image from path"""
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.target_size)
        img = img / 255.0  # Normalize
        return img
    
    def preprocess_array(self, image_array: np.ndarray) -> np.ndarray:
        """Preprocess a numpy array image"""
        if isinstance(image_array, list):
            image_array = np.array(image_array)
        
        # Resize if needed
        if image_array.shape[:2] != self.target_size:
            image_array = cv2.resize(image_array, self.target_size)
        
        # Normalize
        image_array = image_array / 255.0
        
        # Add batch dimension if needed
        if len(image_array.shape) == 3:
            image_array = np.expand_dims(image_array, axis=0)
            
        return image_array
    
    def extract_features(self, image_array: np.ndarray) -> Dict[str, float]:
        """
        Extract interpretable features from image
        This provides the 3+ feature interpretations required
        """
        features = {}
        
        # Remove batch dimension if present
        if len(image_array.shape) == 4:
            image_array = image_array[0]
        
        # Convert to uint8 for OpenCV
        img_uint8 = (image_array * 255).astype(np.uint8)
        
        # FEATURE 1: Color distribution
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
        features['hue_mean'] = float(np.mean(hsv[:,:,0]))
        features['saturation_mean'] = float(np.mean(hsv[:,:,1]))
        features['value_mean'] = float(np.mean(hsv[:,:,2]))
        
        # Color channel means
        features['red_mean'] = float(np.mean(img_uint8[:,:,0]))
        features['green_mean'] = float(np.mean(img_uint8[:,:,1]))
        features['blue_mean'] = float(np.mean(img_uint8[:,:,2]))
        
        # FEATURE 2: Texture (using GLCM)
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        
        # Calculate GLCM
        glcm = graycomatrix(gray, [1], [0], 256, symmetric=True)
        features['contrast'] = float(graycoprops(glcm, 'contrast')[0, 0])
        features['homogeneity'] = float(graycoprops(glcm, 'homogeneity')[0, 0])
        features['energy'] = float(graycoprops(glcm, 'energy')[0, 0])
        features['correlation'] = float(graycoprops(glcm, 'correlation')[0, 0])
        
        # FEATURE 3: Edge density
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = float(np.sum(edges > 0) / (gray.shape[0] * gray.shape[1]))
        
        # Additional features for interpretation
        features['brightness'] = float(np.mean(gray))
        features['std_dev'] = float(np.std(gray))
        
        return features
    
    def get_interpretation(self, class_name: str, features: Dict) -> str:
        """Generate human-readable interpretation based on features"""
        
        interpretations = {
            'healthy': {
                'color': "Green color dominance indicates healthy chlorophyll production",
                'texture': "Smooth texture shows uniform leaf structure with no lesions",
                'edges': "Low edge density indicates no disease or pest damage boundaries"
            },
            'diseased': {
                'color': "Yellow/brown spots indicate disease progression and tissue necrosis",
                'texture': "Rough texture from leaf lesions and fungal growth",
                'edges': "Medium edge density from spot boundaries and lesion margins"
            },
            'pest': {
                'color': "Dark patches from pest feeding damage and frass accumulation",
                'texture': "Irregular texture from tissue removal and damage",
                'edges': "High edge density from complex damage boundaries and holes"
            }
        }
        
        base = interpretations.get(class_name, interpretations['healthy'])
        
        # Customize based on actual feature values
        color_story = base['color']
        texture_story = base['texture']
        edge_story = base['edges']
        
        if features.get('contrast', 0) > 100:
            texture_story += " (high contrast detected, indicating severe damage)"
        elif features.get('contrast', 0) > 50:
            texture_story += " (moderate contrast, indicating early stage damage)"
        
        if features.get('edge_density', 0) > 0.15:
            edge_story += " - severe damage indicated"
        elif features.get('edge_density', 0) > 0.08:
            edge_story += " - moderate damage indicated"
        
        return f"📖 {color_story}. {texture_story}. {edge_story}."

class TabularPreprocessor:
    """Preprocess tabular data for price prediction"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.commodity_encoder = LabelEncoder()
        self.market_encoder = LabelEncoder()
        self.state_encoder = LabelEncoder()
        self.is_fitted = False
        
    def fit(self, df: pd.DataFrame):
        """Fit preprocessors on training data"""
        # Encode categoricals
        self.commodity_encoder.fit(df['commodity'].unique())
        self.market_encoder.fit(df['market'].unique())
        self.state_encoder.fit(df['state'].unique())
        
        # Prepare features for scaling
        feature_cols = ['arrivals', 'min_price', 'max_price']
        self.scaler.fit(df[feature_cols])
        
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted preprocessors"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        df_transformed = df.copy()
        
        # Encode categoricals
        df_transformed['commodity_code'] = self.commodity_encoder.transform(df['commodity'])
        df_transformed['market_code'] = self.market_encoder.transform(df['market'])
        df_transformed['state_code'] = self.state_encoder.transform(df['state'])
        
        # Scale numerical features
        feature_cols = ['arrivals', 'min_price', 'max_price']
        df_transformed[feature_cols] = self.scaler.transform(df[feature_cols])
        
        # Create date features
        df_transformed['date'] = pd.to_datetime(df_transformed['date'])
        df_transformed['year'] = df_transformed['date'].dt.year
        df_transformed['month'] = df_transformed['date'].dt.month
        df_transformed['day_of_year'] = df_transformed['date'].dt.dayofyear
        df_transformed['week'] = df_transformed['date'].dt.isocalendar().week
        
        return df_transformed
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step"""
        self.fit(df)
        return self.transform(df)
    
    def save(self, path: str):
        """Save preprocessors to disk"""
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.scaler, f"{path}/scaler.pkl")
        joblib.dump(self.commodity_encoder, f"{path}/commodity_encoder.pkl")
        joblib.dump(self.market_encoder, f"{path}/market_encoder.pkl")
        joblib.dump(self.state_encoder, f"{path}/state_encoder.pkl")
        
        # Save metadata
        metadata = {
            'is_fitted': self.is_fitted,
            'commodity_classes': self.commodity_encoder.classes_.tolist(),
            'market_classes': self.market_encoder.classes_.tolist(),
            'state_classes': self.state_encoder.classes_.tolist()
        }
        import json
        with open(f"{path}/preprocessor_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load(self, path: str):
        """Load preprocessors from disk"""
        self.scaler = joblib.load(f"{path}/scaler.pkl")
        self.commodity_encoder = joblib.load(f"{path}/commodity_encoder.pkl")
        self.market_encoder = joblib.load(f"{path}/market_encoder.pkl")
        self.state_encoder = joblib.load(f"{path}/state_encoder.pkl")
        self.is_fitted = True
        return self
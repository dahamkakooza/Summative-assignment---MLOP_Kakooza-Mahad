"""
Model tests for AgriPrice Prophet
"""

import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from src.model import CropDiseaseModel, PricePredictionModel
from src.preprocessing import ImagePreprocessor, TabularPreprocessor
import pandas as pd

class TestImageModel:
    """Test image classification model"""
    
    def setup_method(self):
        self.model = CropDiseaseModel(num_classes=3)
        self.preprocessor = ImagePreprocessor()
    
    def test_model_build(self):
        """Test model building"""
        model = self.model.build_model()
        assert model is not None
        assert model.output_shape[-1] == 3
    
    def test_prediction_shape(self):
        """Test prediction output shape"""
        self.model.build_model()
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        processed = self.preprocessor.preprocess_array(dummy_image)
        
        class_idx, confidence, probabilities = self.model.predict(processed)
        
        assert isinstance(class_idx, (int, np.integer))
        assert 0 <= confidence <= 1
        assert len(probabilities) == 3
    
    def test_feature_extraction(self):
        """Test feature extraction"""
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        features = self.preprocessor.extract_features(dummy_image)
        
        assert 'contrast' in features
        assert 'edge_density' in features
        assert 'green_mean' in features
        assert 'homogeneity' in features
        
    def test_get_interpretation(self):
        """Test interpretation generation"""
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        features = self.preprocessor.extract_features(dummy_image)
        
        interpretation = self.preprocessor.get_interpretation('healthy', features)
        assert isinstance(interpretation, str)
        assert len(interpretation) > 10

class TestPriceModel:
    """Test price prediction model"""
    
    def setup_method(self):
        self.model = PricePredictionModel()
        self.model.build_model()
    
    def test_model_prediction(self):
        """Test model prediction"""
        # Create dummy features
        features = {
            'commodity_code': 1,
            'market_code': 2,
            'state_code': 1,
            'year': 2024,
            'month': 3,
            'day_of_year': 60,
            'arrivals': 100,
            'min_price': 50,
            'max_price': 60
        }
        
        prediction = self.model.predict(features)
        assert isinstance(prediction, float)
        assert prediction > 0
    
    def test_feature_importance(self):
        """Test feature importance extraction"""
        importance = self.model.get_feature_importance()
        assert isinstance(importance, dict)

class TestTabularPreprocessor:
    """Test tabular preprocessor"""
    
    def setup_method(self):
        self.preprocessor = TabularPreprocessor()
        
        # Create sample data
        self.df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02'],
            'commodity': ['Potato', 'Tomato'],
            'market': ['Delhi', 'Mumbai'],
            'state': ['Delhi', 'Maharashtra'],
            'arrivals': [100, 200],
            'min_price': [50, 60],
            'max_price': [60, 70],
            'modal_price': [55, 65]
        })
    
    def test_fit_transform(self):
        """Test fit_transform"""
        df_transformed = self.preprocessor.fit_transform(self.df)
        
        assert 'commodity_code' in df_transformed.columns
        assert 'market_code' in df_transformed.columns
        assert 'year' in df_transformed.columns
        assert 'month' in df_transformed.columns
        
    def test_save_load(self):
        """Test save and load"""
        self.preprocessor.fit(self.df)
        
        # Save to temp location
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            self.preprocessor.save(tmpdir)
            
            # Load new preprocessor
            new_preprocessor = TabularPreprocessor()
            new_preprocessor.load(tmpdir)
            
            assert new_preprocessor.is_fitted
            assert new_preprocessor.scaler is not None
"""
Preprocessing tests for AgriPrice Prophet
"""

import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.preprocessing import ImagePreprocessor
import cv2

class TestImagePreprocessing:
    """Test image preprocessing functions"""
    
    def setup_method(self):
        self.preprocessor = ImagePreprocessor(target_size=(224, 224))
    
    def test_preprocess_array(self):
        """Test preprocessing of image array"""
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        processed = self.preprocessor.preprocess_array(dummy_image)
        
        # Check shape
        assert processed.shape == (1, 224, 224, 3)
        
        # Check normalization
        assert processed.max() <= 1.0
        assert processed.min() >= 0.0
    
    def test_preprocess_list(self):
        """Test preprocessing of list input"""
        dummy_list = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8).tolist()
        
        processed = self.preprocessor.preprocess_array(dummy_list)
        assert processed.shape == (1, 224, 224, 3)
    
    def test_feature_extraction_count(self):
        """Test that feature extraction returns at least 3 features"""
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        features = self.preprocessor.extract_features(dummy_image)
        
        assert len(features) >= 3
    
    def test_feature_extraction_keys(self):
        """Test that specific features exist"""
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        features = self.preprocessor.extract_features(dummy_image)
        
        expected_keys = ['contrast', 'homogeneity', 'edge_density', 
                        'green_mean', 'red_mean', 'blue_mean']
        
        for key in expected_keys:
            assert key in features
    
    def test_interpretation_generation(self):
        """Test interpretation generation for all classes"""
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        features = self.preprocessor.extract_features(dummy_image)
        
        for class_name in ['healthy', 'diseased', 'pest']:
            interpretation = self.preprocessor.get_interpretation(class_name, features)
            assert isinstance(interpretation, str)
            assert len(interpretation) > 20
            assert class_name in interpretation.lower() or class_name.capitalize() in interpretation
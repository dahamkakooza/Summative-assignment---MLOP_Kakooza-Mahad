"""
Model creation and training module with optimization techniques
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
import xgboost as xgb
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, mean_absolute_error,
                           mean_squared_error, r2_score)
import joblib
import json
import os
from typing import Dict, Tuple, Any, List

class CropDiseaseModel:
    """
    CNN model for crop disease classification
    Uses transfer learning (MobileNetV2) and early stopping
    """
    
    def __init__(self, num_classes: int = 3, input_shape: Tuple = (224, 224, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build CNN model with transfer learning (OPTIMIZATION TECHNIQUE 1)"""
        # Load pre-trained MobileNetV2 (transfer learning)
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Build model
        self.model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),  # Regularization
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),  # Regularization
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile with Adam optimizer
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
        
        return self.model
    
    def train(self, train_generator, validation_generator, epochs=30):
        """
        Train the model with early stopping (OPTIMIZATION TECHNIQUE 2)
        """
        # Callbacks for optimization
        callbacks = [
            # Early stopping - prevents overfitting
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            # Learning rate reduction on plateau
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-6,
                verbose=1
            ),
            # Model checkpoint
            keras.callbacks.ModelCheckpoint(
                'models/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def predict(self, image_array):
        """Make prediction on single image"""
        if len(image_array.shape) == 3:
            image_array = np.expand_dims(image_array, axis=0)
        
        predictions = self.model.predict(image_array, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        
        return class_idx, confidence, predictions[0].tolist()
    
    def evaluate(self, test_generator) -> Dict[str, Any]:
        """
        Evaluate model with multiple metrics (5+ metrics)
        """
        # Get predictions
        test_generator.reset()
        predictions = self.model.predict(test_generator, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_generator.classes
        class_names = list(test_generator.class_indices.keys())
        
        # METRIC 1: Accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # METRIC 2: Precision (weighted)
        precision = precision_score(y_true, y_pred, average='weighted')
        
        # METRIC 3: Recall (weighted)
        recall = recall_score(y_true, y_pred, average='weighted')
        
        # METRIC 4: F1 Score (weighted)
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # METRIC 5: Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # METRIC 6: Per-class metrics
        per_class_precision = precision_score(y_true, y_pred, average=None)
        per_class_recall = recall_score(y_true, y_pred, average=None)
        per_class_f1 = f1_score(y_true, y_pred, average=None)
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'per_class': {
                class_names[i]: {
                    'precision': float(per_class_precision[i]),
                    'recall': float(per_class_recall[i]),
                    'f1': float(per_class_f1[i])
                } for i in range(len(class_names))
            }
        }
        
        return metrics
    
    def save(self, path: str):
        """Save model to disk"""
        self.model.save(f"{path}/crop_disease_model.h5")
        
        # Save as TFLite for mobile deployment
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()
        with open(f"{path}/crop_disease_model.tflite", 'wb') as f:
            f.write(tflite_model)
    
    def load(self, path: str):
        """Load model from disk"""
        self.model = keras.models.load_model(f"{path}/crop_disease_model.h5")
        return self.model

class PricePredictionModel:
    """
    XGBoost model for price prediction
    Uses regularization (OPTIMIZATION TECHNIQUE 3)
    """
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        
    def build_model(self, params: Dict = None):
        """
        Build XGBoost model with regularization
        """
        if params is None:
            # Parameters with regularization
            params = {
                'n_estimators': 300,
                'learning_rate': 0.05,
                'max_depth': 6,
                'reg_alpha': 0.1,      # L1 regularization
                'reg_lambda': 1.0,      # L2 regularization
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
        
        self.model = xgb.XGBRegressor(**params)
        return self.model
    
    def train(self, X_train, y_train, X_test=None, y_test=None):
        """Train the model"""
        eval_set = [(X_train, y_train)]
        if X_test is not None:
            eval_set.append((X_test, y_test))
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        return self.model
    
    def predict(self, features):
        """Make prediction"""
        if isinstance(features, dict):
            import pandas as pd
            features = pd.DataFrame([features])
        
        prediction = self.model.predict(features)[0]
        return float(prediction)
    
    def evaluate(self, X_test, y_test) -> Dict[str, float]:
        """
        Evaluate with multiple regression metrics
        """
        y_pred = self.model.predict(X_test)
        
        # METRIC 1: Mean Absolute Error
        mae = mean_absolute_error(y_test, y_pred)
        
        # METRIC 2: Mean Squared Error
        mse = mean_squared_error(y_test, y_pred)
        
        # METRIC 3: Root Mean Squared Error
        rmse = np.sqrt(mse)
        
        # METRIC 4: R² Score
        r2 = r2_score(y_test, y_pred)
        
        # METRIC 5: Explained Variance
        explained_var = 1 - (np.var(y_test - y_pred) / np.var(y_test))
        
        metrics = {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'r2': float(r2),
            'explained_variance': float(explained_var)
        }
        
        return metrics
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            return {f"feature_{i}": float(imp) for i, imp in enumerate(importance)}
        return {}
    
    def save(self, path: str):
        """Save model to disk"""
        joblib.dump(self.model, f"{path}/price_model.pkl")
        
        # Save model metadata
        metadata = {
            'feature_importance': self.get_feature_importance(),
            'params': self.model.get_params()
        }
        with open(f"{path}/price_model_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load(self, path: str):
        """Load model from disk"""
        self.model = joblib.load(f"{path}/price_model.pkl")
        return self.model
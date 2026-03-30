"""
Initial training script for both models
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split

from src.model import CropDiseaseModel, PricePredictionModel
from src.preprocessing import TabularPreprocessor
from src.data_acquisition import DataAcquisition

def train_image_model():
    """Train the crop disease classification model"""
    print("\n" + "="*60)
    print("🖼️ Training Image Classification Model")
    print("="*60)
    
    # Data generators with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
    
    # Save class indices
    class_indices = train_generator.class_indices
    with open('models/class_indices.json', 'w') as f:
        json.dump(class_indices, f, indent=2)
    
    print(f"\nClasses: {class_indices}")
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {validation_generator.samples}")
    
    # Build and train model
    model = CropDiseaseModel(num_classes=len(class_indices))
    model.build_model()
    model.model.summary()
    
    print("\nTraining model with early stopping...")
    history = model.train(train_generator, validation_generator, epochs=30)
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = model.evaluate(validation_generator)
    
    print(f"\n✅ Model Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    
    # Save model
    model.save('models/')
    print("\n✅ Image model saved to models/")
    
    return model, metrics

def train_price_model():
    """Train the price prediction model"""
    print("\n" + "="*60)
    print("💰 Training Price Prediction Model")
    print("="*60)
    
    # Load data
    df = pd.read_csv('data/raw/price_data.csv')
    print(f"Loaded {len(df)} records")
    
    # Preprocess
    preprocessor = TabularPreprocessor()
    df_transformed = preprocessor.fit_transform(df)
    
    # Prepare features and target
    feature_cols = ['commodity_code', 'market_code', 'state_code', 
                    'year', 'month', 'day_of_year',
                    'arrivals', 'min_price', 'max_price']
    
    X = df_transformed[feature_cols]
    y = df_transformed['modal_price']
    
    print(f"Features: {len(feature_cols)}")
    print(f"Training samples: {len(X)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Build and train model
    model = PricePredictionModel()
    model.build_model()
    model.train(X_train, y_train, X_test, y_test)
    
    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    
    print(f"\n✅ Model Metrics:")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  Explained Variance: {metrics['explained_variance']:.4f}")
    
    # Save model and preprocessor
    model.save('models/')
    preprocessor.save('models/')
    
    print("\n✅ Price model saved to models/")
    
    return model, preprocessor, metrics

def main():
    """Main training function"""
    print("🚀 Training all models for AgriPrice Prophet")
    print("="*60)
    
    os.makedirs('models', exist_ok=True)
    
    # Train image model
    image_model, image_metrics = train_image_model()
    
    # Train price model
    price_model, preprocessor, price_metrics = train_price_model()
    
    # Summary
    print("\n" + "="*60)
    print("📊 TRAINING SUMMARY")
    print("="*60)
    
    print("\n📷 Image Model:")
    for k, v in image_metrics.items():
        if k != 'confusion_matrix' and k != 'per_class':
            print(f"  {k}: {v:.4f}")
    
    print("\n💰 Price Model:")
    for k, v in price_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print("\n✅ All models trained successfully!")
    print("\nModels saved in 'models/' directory:")
    os.system("ls -la models/")

if __name__ == "__main__":
    main()
"""
Data acquisition module for loading and generating datasets
"""

import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
from datetime import datetime, timedelta

class DataAcquisition:
    """Handle data loading and acquisition"""
    
    @staticmethod
    def generate_sample_images(output_dir='data', num_images_per_class=400):
        """
        Generate sample images for crop disease classification
        
        Args:
            output_dir: Base directory for output
            num_images_per_class: Number of images per class
        """
        classes = ['healthy', 'diseased', 'pest']
        img_size = 224
        
        print(f"Generating {num_images_per_class} images per class...")
        
        for class_name in classes:
            # Create directories
            train_dir = os.path.join(output_dir, 'train', class_name)
            test_dir = os.path.join(output_dir, 'test', class_name)
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)
            
            # Generate images
            for i in range(num_images_per_class):
                # Create base image
                if class_name == 'healthy':
                    # Green-dominant image
                    img = np.random.randint(30, 150, (img_size, img_size, 3), dtype=np.uint8)
                    img[:,:,1] = img[:,:,1] * 1.2  # Boost green channel
                    
                elif class_name == 'diseased':
                    # Yellow/brown spots
                    img = np.random.randint(100, 200, (img_size, img_size, 3), dtype=np.uint8)
                    # Add spots
                    for _ in range(random.randint(5, 15)):
                        x = random.randint(50, img_size-50)
                        y = random.randint(50, img_size-50)
                        radius = random.randint(10, 30)
                        cv2.circle(img, (x, y), radius, (0, 255, 255), -1)
                        
                else:  # pest
                    # Dark patches
                    img = np.random.randint(50, 180, (img_size, img_size, 3), dtype=np.uint8)
                    for _ in range(random.randint(3, 8)):
                        x = random.randint(50, img_size-50)
                        y = random.randint(50, img_size-50)
                        cv2.rectangle(img, (x, y), (x+40, y+40), (0, 0, 0), -1)
                
                # Split into train/test (80/20)
                if i < int(num_images_per_class * 0.8):
                    cv2.imwrite(os.path.join(train_dir, f'img_{i}.jpg'), img)
                else:
                    cv2.imwrite(os.path.join(test_dir, f'img_{i - int(num_images_per_class * 0.8)}.jpg'), img)
            
            print(f"  ✅ Generated {class_name} images")
    
    @staticmethod
    def generate_price_data(output_path='data/raw/price_data.csv', num_samples=5000):
        """
        Generate sample price data
        
        Args:
            output_path: Path to save CSV
            num_samples: Number of samples to generate
        """
        commodities = ['Potato', 'Tomato', 'Onion', 'Wheat', 'Rice', 'Maize']
        markets = ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata', 'Pune']
        states = ['Delhi', 'Maharashtra', 'Karnataka', 'Tamil Nadu', 'West Bengal', 'Maharashtra']
        
        base_prices = {
            'Potato': 25, 'Tomato': 30, 'Onion': 35, 
            'Wheat': 28, 'Rice': 40, 'Maize': 22
        }
        
        data = []
        start_date = datetime(2024, 1, 1)
        
        for i in range(num_samples):
            date = start_date + timedelta(days=i % 365)
            commodity = random.choice(commodities)
            market = random.choice(markets)
            state = random.choice(states)
            
            # Price with seasonal variation
            base = base_prices[commodity]
            seasonal = 10 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
            noise = np.random.normal(0, 3)
            
            modal_price = base + seasonal + noise
            modal_price = max(10, modal_price)
            
            arrivals = random.uniform(50, 1000)
            min_price = modal_price * random.uniform(0.8, 0.95)
            max_price = modal_price * random.uniform(1.05, 1.2)
            
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'commodity': commodity,
                'market': market,
                'state': state,
                'arrivals': round(arrivals, 2),
                'min_price': round(min_price, 2),
                'max_price': round(max_price, 2),
                'modal_price': round(modal_price, 2)
            })
        
        df = pd.DataFrame(data)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✅ Generated {len(df)} price records")
        
        return df
    
    @staticmethod
    def load_image_data(data_dir, target_size=(224, 224), batch_size=32):
        """
        Load image data using TensorFlow data generator
        
        Args:
            data_dir: Directory containing class subfolders
            target_size: Image size for resizing
            batch_size: Batch size for training
        """
        datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        train_generator = datagen.flow_from_directory(
            data_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        validation_generator = datagen.flow_from_directory(
            data_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        return train_generator, validation_generator
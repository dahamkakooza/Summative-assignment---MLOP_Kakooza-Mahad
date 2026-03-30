#!/usr/bin/env python3
"""
Generate sample data for AgriPrice Prophet
Run: python scripts/generate_data.py
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_acquisition import DataAcquisition

def main():
    """Generate all sample data"""
    print("="*60)
    print("🌾 AgriPrice Prophet - Data Generation")
    print("="*60)
    
    # Create directories
    os.makedirs('data/train', exist_ok=True)
    os.makedirs('data/test', exist_ok=True)
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/uploads', exist_ok=True)
    
    # Generate image data
    print("\n📷 Generating image data...")
    DataAcquisition.generate_sample_images(
        output_dir='data',
        num_images_per_class=400
    )
    
    # Generate price data
    print("\n💰 Generating price data...")
    DataAcquisition.generate_price_data(
        output_path='data/raw/price_data.csv',
        num_samples=5000
    )
    
    # Create a sample upload file
    print("\n📤 Creating sample upload file...")
    import pandas as pd
    df = pd.read_csv('data/raw/price_data.csv').sample(n=100)
    df.to_csv('data/uploads/sample_retrain_data.csv', index=False)
    
    print("\n" + "="*60)
    print("✅ Data generation complete!")
    print("="*60)
    
    # Show directory structure
    print("\n📁 Directory structure:")
    os.system("ls -R data/ | head -30")

if __name__ == "__main__":
    main()
"""
Utility functions for the project
"""

import json
import os
import numpy as np
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd

def save_to_json(data: Dict[str, Any], filename: str):
    """Save data to JSON file"""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def load_from_json(filename: str) -> Dict[str, Any]:
    """Load data from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)

def get_timestamp() -> str:
    """Get current timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def validate_image_file(filename: str) -> bool:
    """Check if file is a valid image"""
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    ext = os.path.splitext(filename)[1].lower()
    return ext in valid_extensions

def validate_csv_file(filename: str) -> bool:
    """Check if file is a valid CSV"""
    return filename.lower().endswith('.csv')

def calculate_confidence_interval(predictions: List[float], confidence: float = 0.95) -> Dict[str, float]:
    """Calculate confidence interval for predictions"""
    mean_pred = np.mean(predictions)
    std_pred = np.std(predictions)
    z_score = 1.96  # For 95% confidence
    
    margin = z_score * std_pred / np.sqrt(len(predictions))
    
    return {
        'mean': float(mean_pred),
        'lower': float(mean_pred - margin),
        'upper': float(mean_pred + margin),
        'confidence': confidence
    }

def format_metrics_table(metrics: Dict[str, Any]) -> str:
    """Format metrics as markdown table"""
    table = "| Metric | Value |\n|--------|-------|\n"
    for key, value in metrics.items():
        if isinstance(value, float):
            table += f"| {key} | {value:.4f} |\n"
        else:
            table += f"| {key} | {value} |\n"
    return table
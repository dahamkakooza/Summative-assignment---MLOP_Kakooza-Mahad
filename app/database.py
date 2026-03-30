"""
Database operations for AgriPrice Prophet
"""

import sqlite3
import pandas as pd
import json
from datetime import datetime
import os
from typing import Dict, List, Optional, Any

DB_PATH = "database/agriprice.db"

def init_db():
    """Initialize database tables"""
    os.makedirs("database", exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Uploaded datasets table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS uploaded_datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            data_type TEXT,
            row_count INTEGER,
            file_path TEXT,
            status TEXT DEFAULT 'pending'
        )
    ''')
    
    # Retraining history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS retraining_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER,
            model_type TEXT,
            retrain_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            model_path TEXT,
            metrics TEXT,
            status TEXT,
            FOREIGN KEY (dataset_id) REFERENCES uploaded_datasets (id)
        )
    ''')
    
    # Predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            prediction_type TEXT,
            input_data TEXT,
            output_data TEXT,
            latency_ms REAL
        )
    ''')
    
    # Model performance tracking
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            model_type TEXT,
            accuracy REAL,
            precision REAL,
            recall REAL,
            f1_score REAL,
            latency_avg REAL,
            requests_count INTEGER
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized")

def save_uploaded_dataset(filename: str, file_path: str, data_type: str, rows: int) -> int:
    """Save uploaded dataset metadata"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO uploaded_datasets 
        (filename, data_type, row_count, file_path, status)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        filename,
        data_type,
        rows,
        file_path,
        'uploaded'
    ))
    
    dataset_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return dataset_id

def get_dataset_id_by_filename(filename: str) -> Optional[int]:
    """Get dataset ID by filename"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id FROM uploaded_datasets 
        WHERE filename = ? 
        ORDER BY upload_date DESC LIMIT 1
    ''', (filename,))
    
    result = cursor.fetchone()
    conn.close()
    
    return result[0] if result else None

def save_retraining_result(dataset_id: int, results: Dict):
    """Save retraining results"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    for model_type, result in results.get('results', {}).items():
        cursor.execute('''
            INSERT INTO retraining_history 
            (dataset_id, model_type, model_path, metrics, status)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            dataset_id,
            model_type,
            result.get('model_path'),
            json.dumps(result.get('metrics', {})),
            'completed'
        ))
    
    # Update dataset status
    cursor.execute('''
        UPDATE uploaded_datasets SET status = 'processed' 
        WHERE id = ?
    ''', (dataset_id,))
    
    conn.commit()
    conn.close()

def save_prediction(prediction_type: str, input_data: Dict, output_data: Dict, latency_ms: float):
    """Save prediction for tracking"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO predictions 
        (prediction_type, input_data, output_data, latency_ms)
        VALUES (?, ?, ?, ?)
    ''', (
        prediction_type,
        json.dumps(input_data),
        json.dumps(output_data),
        latency_ms
    ))
    
    conn.commit()
    conn.close()

def save_model_performance(metrics: Dict):
    """Save model performance metrics"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO model_performance 
        (model_type, accuracy, precision, recall, f1_score, latency_avg, requests_count)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        metrics.get('model_type'),
        metrics.get('accuracy'),
        metrics.get('precision'),
        metrics.get('recall'),
        metrics.get('f1_score'),
        metrics.get('latency_avg'),
        metrics.get('requests_count')
    ))
    
    conn.commit()
    conn.close()

def get_stats() -> Dict[str, Any]:
    """Get statistics for dashboard"""
    conn = sqlite3.connect(DB_PATH)
    
    stats = {}
    
    # Total counts
    try:
        stats['total_predictions'] = pd.read_sql(
            "SELECT COUNT(*) as count FROM predictions", conn
        ).iloc[0, 0]
    except:
        stats['total_predictions'] = 0
    
    try:
        stats['total_uploads'] = pd.read_sql(
            "SELECT COUNT(*) as count FROM uploaded_datasets", conn
        ).iloc[0, 0]
    except:
        stats['total_uploads'] = 0
    
    try:
        stats['total_retrainings'] = pd.read_sql(
            "SELECT COUNT(*) as count FROM retraining_history", conn
        ).iloc[0, 0]
    except:
        stats['total_retrainings'] = 0
    
    # Recent activity
    try:
        stats['recent_predictions'] = pd.read_sql("""
            SELECT prediction_date, prediction_type, latency_ms 
            FROM predictions 
            ORDER BY prediction_date DESC 
            LIMIT 10
        """, conn).to_dict('records')
    except:
        stats['recent_predictions'] = []
    
    try:
        stats['recent_uploads'] = pd.read_sql("""
            SELECT filename, upload_date, data_type, row_count 
            FROM uploaded_datasets 
            ORDER BY upload_date DESC 
            LIMIT 10
        """, conn).to_dict('records')
    except:
        stats['recent_uploads'] = []
    
    conn.close()
    
    return stats

def get_retraining_history(limit: int = 20) -> List[Dict]:
    """Get retraining history"""
    conn = sqlite3.connect(DB_PATH)
    
    try:
        history = pd.read_sql("""
            SELECT 
                r.retrain_date,
                d.filename,
                r.model_type,
                r.metrics,
                r.status
            FROM retraining_history r
            JOIN uploaded_datasets d ON r.dataset_id = d.id
            ORDER BY r.retrain_date DESC
            LIMIT ?
        """, conn, params=(limit,)).to_dict('records')
    except:
        history = []
    
    conn.close()
    return history

def get_prediction_stats() -> Dict[str, Any]:
    """Get prediction statistics"""
    conn = sqlite3.connect(DB_PATH)
    
    stats = {}
    
    try:
        # Average latency by prediction type
        stats['avg_latency'] = pd.read_sql("""
            SELECT prediction_type, AVG(latency_ms) as avg_latency
            FROM predictions
            GROUP BY prediction_type
        """, conn).to_dict('records')
    except:
        stats['avg_latency'] = []
    
    try:
        # Predictions over time
        stats['predictions_over_time'] = pd.read_sql("""
            SELECT DATE(prediction_date) as date, 
                   COUNT(*) as count,
                   AVG(latency_ms) as avg_latency
            FROM predictions
            WHERE prediction_date > DATE('now', '-7 days')
            GROUP BY DATE(prediction_date)
            ORDER BY date
        """, conn).to_dict('records')
    except:
        stats['predictions_over_time'] = []
    
    conn.close()
    
    return stats

# Initialize on import
init_db()
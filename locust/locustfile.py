"""
Locust load testing file for AgriPrice Prophet
Simulates flood of requests to test model performance
"""

from locust import HttpUser, task, between, tag
import random
import numpy as np
from datetime import datetime, timedelta
import json

class AgriPriceUser(HttpUser):
    """
    Simulated user for load testing
    """
    wait_time = between(0.5, 2)  # Wait between requests
    
    def on_start(self):
        """Initialize user session"""
        self.commodities = ['Potato', 'Tomato', 'Onion', 'Wheat', 'Rice', 'Maize']
        self.markets = ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
        self.states = ['Delhi', 'Maharashtra', 'Karnataka', 'Tamil Nadu', 'West Bengal']
        
        # Create test image (small 50x50 RGB image)
        self.test_image = self.create_test_image()
    
    def create_test_image(self):
        """Create a simple test image"""
        # Create a random 50x50 RGB image
        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        return img.tolist()
    
    @task(5)  # Higher weight = more frequent
    def predict_price(self):
        """Test price prediction endpoint"""
        payload = {
            "commodity": random.choice(self.commodities),
            "market": random.choice(self.markets),
            "state": random.choice(self.states),
            "date": (datetime.now() + timedelta(days=random.randint(1, 30))).date().isoformat(),
            "arrivals": round(random.uniform(50, 500), 2),
            "min_price": round(random.uniform(30, 80), 2),
            "max_price": round(random.uniform(40, 100), 2)
        }
        
        with self.client.post("/predict/price", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status {response.status_code}")
    
    @task(3)
    def health_check(self):
        """Test health endpoint"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    @task(2)
    def get_commodities(self):
        """Test commodities endpoint"""
        with self.client.get("/commodities", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed: {response.status_code}")
    
    @task(1)
    def get_disease_classes(self):
        """Test disease classes endpoint"""
        with self.client.get("/disease-classes", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed: {response.status_code}")

class ImagePredictionUser(HttpUser):
    """Specialized user for image prediction load testing"""
    wait_time = between(1, 3)
    
    def on_start(self):
        self.test_image = self.create_test_image()
    
    def create_test_image(self):
        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        return img.tolist()
    
    @task
    def predict_image(self):
        """Test image prediction endpoint"""
        # For image prediction, we need to send as multipart/form-data
        # This is simplified - in real test you'd use proper file upload
        files = {'file': ('test.jpg', json.dumps(self.test_image), 'image/jpeg')}
        
        with self.client.post("/predict/image", files=files, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed: {response.status_code}")
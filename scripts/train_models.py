#!/usr/bin/env python3
"""
Train all models for AgriPrice Prophet
Run: python scripts/train_models.py
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.train import main

if __name__ == "__main__":
    main()
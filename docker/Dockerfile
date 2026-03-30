FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/
COPY data/ ./data/
COPY locust/ ./locust/
COPY scripts/ ./scripts/

# Create necessary directories
RUN mkdir -p data/uploads data/train data/test data/raw \
    models/retrained database locust/results

# Expose ports
EXPOSE 8000 8501 8089

# Set environment variables
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run both FastAPI and Streamlit
CMD sh -c "uvicorn app.api:app --host 0.0.0.0 --port 8000 & streamlit run app/ui.py --server.port 8501 --server.address 0.0.0.0"
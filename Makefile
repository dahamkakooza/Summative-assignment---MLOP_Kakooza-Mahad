.PHONY: help install run-api run-ui run-all test docker-build docker-up load-test clean

help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies and generate data"
	@echo "  make run-api      - Run FastAPI backend"
	@echo "  make run-ui       - Run Streamlit frontend"
	@echo "  make run-all      - Run both API and UI"
	@echo "  make test         - Run tests"
	@echo "  make docker-build - Build Docker images"
	@echo "  make docker-up    - Start Docker containers"
	@echo "  make load-test    - Run Locust load test interactively"
	@echo "  make clean        - Clean cache files"

install:
	pip install -r requirements.txt
	python scripts/generate_data.py
	python scripts/train_models.py

run-api:
	uvicorn app.api:app --reload --port 8000

run-ui:
	streamlit run app/ui.py

run-all:
	@echo "Starting API and UI..."
	@make run-api & make run-ui

test:
	pytest tests/ -v --cov=app --cov=src

docker-build:
	docker-compose build

docker-up:
	docker-compose up

docker-scale:
	docker-compose up --scale api=$(c)

load-test:
	locust -f locust/locustfile.py --host=http://localhost:8000

load-test-headless:
	locust -f locust/locustfile.py --headless -u $(users) -r 10 --run-time 5m --host=http://localhost:8000 --csv=locust/results/test_run

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
.PHONY: setup train test run docker-build docker-run clean

# Default target
all: setup test

# Install Python dependencies
setup:
	pip install -r requirements.txt

# Train all models and save artifacts
train:
	MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=. python3 scripts/train_pipeline.py

# Run all tests
test:
	python3 -m pytest tests/ -v --tb=short

# Start the FastAPI server
run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Build Docker image
docker-build:
	docker build -t credit-risk-scoring .

# Run Docker container
docker-run:
	docker run -p 8000:8000 credit-risk-scoring

# Clean Python cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov/ 2>/dev/null || true

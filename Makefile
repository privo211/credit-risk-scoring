.PHONY: setup train test run docker-build docker-run clean

# Default target
all: setup test

# Install Python dependencies
setup:
	pip install -r requirements.txt

# Train all models and save artifacts
train:
	python3 -c "
from src.data_loader import load_and_split_data
from src.preprocess import engineer_features, build_preprocessor
from src.train import train_all_models, select_best_model, save_model, save_preprocessor, save_metadata
from src.evaluate import evaluate_model, print_metrics
from src.config import MODEL_VERSION
import warnings
warnings.filterwarnings('ignore')

print('=' * 60)
print('CREDIT RISK SCORING - MODEL TRAINING')
print('=' * 60)

X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data()
X_train_fe = engineer_features(X_train)
X_val_fe = engineer_features(X_val)
X_test_fe = engineer_features(X_test)

preprocessor = build_preprocessor()
X_train_p = preprocessor.fit_transform(X_train_fe)
X_val_p = preprocessor.transform(X_val_fe)
X_test_p = preprocessor.transform(X_test_fe)
save_preprocessor(preprocessor)

models = train_all_models(X_train_p, y_train)
best_name, best_model = select_best_model(models, X_val_p, y_val)
save_model(best_model, 'models/best_model.pkl')
save_model(models.get('Logistic Regression'), 'models/logistic_regression.pkl')
save_model(models.get('Random Forest'), 'models/random_forest.pkl')
save_model(models.get('XGBoost'), 'models/xgboost.pkl')
test_metrics = evaluate_model(best_model, X_test_p, y_test)
print_metrics(test_metrics, 'TEST SET')
save_metadata(best_name, test_metrics)
print('\\nTraining complete!')
"

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

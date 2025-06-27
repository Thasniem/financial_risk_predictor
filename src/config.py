import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    "train_path": os.path.join(BASE_DIR, "data", "train.csv"),
    "test_path": os.path.join(BASE_DIR, "data", "test.csv"),
    "model_path": os.path.join(BASE_DIR, "models", "model.pkl"),
    "metrics_path": os.path.join(BASE_DIR, "outputs", "metrics.json"),
    "random_state": 42
}
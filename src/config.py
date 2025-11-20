# src/config.py
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Data
IMAGE_DIR = os.path.join(ROOT, "data", "images")
CSV_PATH = os.path.join(ROOT, "data", "labels.csv")

# Model / training
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 8
TOP_K = None  # None => use all labels found in CSV
MODEL_DIR = os.path.join(ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.h5")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.json")

# Reports & DB
REPORT_DIR = os.path.join(ROOT, "reports")
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "chestxray_db"
COLLECTION_NAME = "predictions"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

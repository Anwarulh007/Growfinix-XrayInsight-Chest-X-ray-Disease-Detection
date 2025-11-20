# src/preprocess.py
import cv2
import numpy as np
from config import IMG_SIZE

def load_and_preprocess_image(path, target_size=(IMG_SIZE, IMG_SIZE)):
    """Read image from disk, resize and normalize to [0,1]. Returns RGB numpy array float32."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype("float32") / 255.0
    return img

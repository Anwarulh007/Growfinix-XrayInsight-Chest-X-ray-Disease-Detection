# src/predict.py
import json
import numpy as np
from tensorflow.keras.models import load_model
from config import MODEL_PATH, LABEL_MAP_PATH, IMG_SIZE
from preprocess import load_and_preprocess_image
from gradcam import make_gradcam_heatmap, find_last_conv_layer, overlay_heatmap
from PIL import Image
import os

_model = None
_labels = None
_last_conv = None

def load_all():
    global _model, _labels, _last_conv
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Train the model first.")
        _model = load_model(MODEL_PATH, compile=False)
    if _labels is None:
        if not os.path.exists(LABEL_MAP_PATH):
            raise FileNotFoundError(f"Label map not found at {LABEL_MAP_PATH}. Run training first.")
        with open(LABEL_MAP_PATH, 'r') as f:
            _labels = json.load(f)
    if _last_conv is None:
        _last_conv = find_last_conv_layer(_model)
    return _model, _labels, _last_conv

def predict_with_gradcam(image_path_or_pil):
    model, labels, last_conv = load_all()
    # accept PIL.Image or filepath
    if isinstance(image_path_or_pil, Image.Image):
        pil = image_path_or_pil.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        img_arr = np.array(pil).astype("float32") / 255.0
    else:
        img_arr = load_and_preprocess_image(image_path_or_pil, target_size=(IMG_SIZE, IMG_SIZE))
        pil = Image.fromarray((img_arr * 255).astype("uint8"))
    inp = np.expand_dims(img_arr, axis=0)
    preds = model.predict(inp)[0]
    top_idx = int(np.argmax(preds))
    heatmap = make_gradcam_heatmap(inp, model, last_conv, pred_index=top_idx)
    overlay = overlay_heatmap(pil, heatmap)
    # predictions as dict
    preds_dict = {labels[i]: float(preds[i]) for i in range(len(labels))}
    return {
        "predictions": preds_dict,
        "top_index": top_idx,
        "top_label": labels[top_idx],
        "overlay_pil": overlay,
        "pil": pil
    }

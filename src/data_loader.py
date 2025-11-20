# src/dataloader.py
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from config import CSV_PATH, IMAGE_DIR, IMG_SIZE, BATCH_SIZE, TOP_K
AUTOTUNE = tf.data.AUTOTUNE

def load_dataframe():
    """Load CSV, create label list and multi-hot vectors. Returns dataframe and label list."""
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]
    if 'Image Index' not in df.columns or 'Finding Labels' not in df.columns:
        raise ValueError("CSV must contain 'Image Index' and 'Finding Labels' columns")
    df['Finding Labels'] = df['Finding Labels'].fillna('No Finding').astype(str)

    # Ensure filenames include extension; common NIH CSV already has .png in Image Index
    def resolve_fname(x):
        x = str(x).strip()
        # if path exists as-is under IMAGE_DIR, keep it
        if os.path.exists(os.path.join(IMAGE_DIR, x)):
            return x
        # try adding .png or .jpg
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = x + ext
            if os.path.exists(os.path.join(IMAGE_DIR, candidate)):
                return candidate
        return x  # return original; downstream will filter non-existing

    df['filename'] = df['Image Index'].apply(resolve_fname)

    # collect labels from 'Finding Labels' by splitting '|'
    all_labels = df['Finding Labels'].str.split('|').explode().str.strip()
    counts = all_labels.value_counts()
    if TOP_K:
        label_list = list(counts.head(TOP_K).index)
    else:
        label_list = list(counts.index)
    print("Labels modeled:", label_list)

    def labels_to_vec(text):
        present = [t.strip() for t in text.split('|')]
        return [1 if lab in present else 0 for lab in label_list]

    df['labels_vec'] = df['Finding Labels'].apply(labels_to_vec)
    df['file_path'] = df['filename'].apply(lambda x: os.path.join(IMAGE_DIR, x))
    # filter out rows where file doesn't exist
    df = df[df['file_path'].apply(lambda p: os.path.exists(p))].reset_index(drop=True)
    print(f"Found {len(df)} usable image files in {IMAGE_DIR}")
    return df, label_list

def _preprocess_image_tf(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return img

def make_dataset(df, batch_size=BATCH_SIZE, shuffle=True):
    """Create a tf.data.Dataset from dataframe with (image, label) pairs."""
    paths = df['file_path'].values
    labels = np.stack(df['labels_vec'].values)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, l: (_preprocess_image_tf(p), l), num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=2048)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds

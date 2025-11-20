# src/train.py
import os
import json
from src.data_loader import load_dataframe, make_dataset
from src.model import build_model
from src.config import MODEL_PATH, LABEL_MAP_PATH, EPOCHS
import tensorflow as tf

def main():
    df, labels = load_dataframe()
    os.makedirs(os.path.dirname(LABEL_MAP_PATH), exist_ok=True)
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(labels, f)
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42, shuffle=True)
    train_ds = make_dataset(train_df, shuffle=True)
    val_ds = make_dataset(val_df, shuffle=False)

    model = build_model(num_classes=len(labels), base_trainable=False)
    print(model.summary())

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor='loss', save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    # fine-tune: unfreeze last layers
    try:
        base = None
        for layer in model.layers:
            if hasattr(layer, "layers") and len(layer.layers) > 0:
                base = layer
                break
        if base:
            base.trainable = True
            for lay in base.layers[:-30]:
                lay.trainable = False
            model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='binary_crossentropy')
            model.fit(train_ds, validation_data=val_ds, epochs=max(1, EPOCHS//2), callbacks=callbacks)
    except Exception:
        pass

    model.save(MODEL_PATH)
    print("Saved model to", MODEL_PATH)

if __name__ == "__main__":
    main()

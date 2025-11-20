# src/gradcam.py
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

def find_last_conv_layer(model):
    # search for Conv2D in model layers (including nested)
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        # nested model/layers
        if hasattr(layer, 'layers'):
            for inner in reversed(getattr(layer, 'layers')):
                if isinstance(inner, tf.keras.layers.Conv2D):
                    return inner.name
    raise ValueError("No Conv2D layer found in model")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    img_array: shape (1, H, W, 3) float32 in [0,1]
    returns: heatmap resized to HxW (numpy float, 0..1)
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()
    # resize to input spatial dims
    heatmap = tf.image.resize(heatmap[..., tf.newaxis], (img_array.shape[1], img_array.shape[2])).numpy().squeeze()
    heatmap = np.clip(heatmap, 0, 1)
    return heatmap

def overlay_heatmap(pil_img, heatmap, alpha=0.4, cmap=plt.cm.jet):
    img = pil_img.convert("RGB").resize((heatmap.shape[1], heatmap.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored = cmap(heatmap_uint8)[:, :, :3]
    colored = np.uint8(colored * 255)
    overlay = Image.blend(img, Image.fromarray(colored), alpha=alpha)
    return overlay

"""
Preprocessing module for wafer defect analysis.
Provides coordinate channel generation and image preprocessing functions.
"""

import numpy as np
import tensorflow as tf
from PIL import Image

def generate_coord_tensor(size: int):
    """
    Generate normalized x, y, and radial coordinate maps.

    Args:
        size (int): Height and width of the output tensor.

    Returns:
        tf.Tensor: Tensor of shape (size, size, 3) containing normalized x, y, r.
    """
    h = w = size
    x = np.linspace(-1.0, 1.0, w)
    y = np.linspace(-1.0, 1.0, h)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2)
    r = r / r.max()  # normalize to [0, 1]
    coord = np.stack([xx, yy, r], axis=-1).astype(np.float32)
    return tf.constant(coord)

def preprocess_image_tf(image_path: str, target_size: int, coord_tensor: tf.Tensor):
    """
    Preprocess image using TensorFlow operations (suitable for tf.data pipeline).

    Args:
        image_path (str): Path to the image.
        target_size (int): Desired height/width after resizing.
        coord_tensor (tf.Tensor): Precomputed coordinate tensor.

    Returns:
        tf.Tensor: Preprocessed image tensor of shape (1, target_size, target_size, 6).
    """
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (target_size, target_size), method='lanczos3')
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.concat([img, coord_tensor], axis=-1)
    return tf.expand_dims(img, axis=0)

def preprocess_image_pil(image_path: str, target_size: int):
    """
    Preprocess image using PIL (suitable for batch inference or when TF is not available).

    Args:
        image_path (str): Path to the image.
        target_size (int): Desired height/width after resizing.

    Returns:
        np.ndarray: Preprocessed image array of shape (1, target_size, target_size, 6).
    """
    # Load and resize
    img = Image.open(image_path).convert('RGB')
    img = img.resize((target_size, target_size), Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0   # (H, W, 3)

    # Generate coordinate channels
    h = w = target_size
    x = np.linspace(-1.0, 1.0, w)
    y = np.linspace(-1.0, 1.0, h)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2)
    r = r / r.max()
    coord = np.stack([xx, yy, r], axis=-1).astype(np.float32)  # (H, W, 3)

    # Concatenate
    input_array = np.concatenate([img_array, coord], axis=-1)   # (H, W, 6)
    return np.expand_dims(input_array, axis=0)                 # (1, H, W, 6)
#!/usr/bin/env python3
"""
Wafer defect classifier inference using ONNX Runtime.
Usage: python infer.py <image_path> [--plot [output.png]]
"""

import os
import sys
import argparse
import numpy as np
import joblib
import onnxruntime as ort
from PIL import Image
import matplotlib.pyplot as plt
from src.preprocess import preprocess_image_pil

def load_onnx_model_and_encoder(onnx_path, encoder_path, variant_path):
    with open(variant_path, 'r') as f:
        variant = f.readline().strip()
        target_size = int(f.readline().strip())
    print(f"Loading {variant} ONNX model (input size {target_size})")
    sess = ort.InferenceSession(onnx_path)
    label_encoder = joblib.load(encoder_path)
    return sess, label_encoder, target_size

def visualize(image_path, classes, probabilities, save_path=None):
    """
    Display or save a figure with original image and probability bar chart.
    If save_path is None, display interactively; otherwise save to file.
    """
    # Load original image for display
    original_img = Image.open(image_path)
    
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Original image
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title("Original Input Image")
    plt.axis('off')
    
    # Subplot 2: Probability bar chart
    plt.subplot(1, 2, 2)
    y_pos = np.arange(len(classes))
    plt.bar(y_pos, probabilities, align='center', alpha=0.7)
    plt.xticks(y_pos, classes, rotation=45, ha='right')
    plt.ylabel('Probability')
    plt.title('Class Probability Distribution')
    plt.ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Wafer defect inference with ONNX Runtime")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--plot", nargs='?', const="visualization.png", default=None,
                        help="Save the visualization to a file (default: visualization.png)")
    args = parser.parse_args()

    image_path = args.image_path
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        sys.exit(1)

    # Default paths – can be overridden with environment variables (useful for Docker)
    onnx_path = os.environ.get("ONNX_MODEL_PATH", "./model/S/best_model.onnx")
    encoder_path = os.environ.get("LABEL_ENCODER_PATH", "./model/S/label_encoder.pkl")
    variant_path = os.environ.get("VARIANT_PATH", "./model/S/variant.txt")

    # Load model and encoder
    sess, label_encoder, target_size = load_onnx_model_and_encoder(
        onnx_path, encoder_path, variant_path
    )
    classes = label_encoder.classes_

    # Preprocess and predict
    input_np = preprocess_image_pil(image_path, target_size)   # shape (1, H, W, 6)
    onnx_output = sess.run(None, {"input": input_np})[0]       # shape (1, 9)
    predictions = onnx_output[0]
    idx = np.argmax(predictions)
    pred_class = classes[idx]
    confidence = predictions[idx]

    print(f"Predicted class: {pred_class}")
    print(f"Confidence: {confidence:.4f}")

    # Show or save visualization
    visualize(image_path, classes, predictions, save_path=args.plot)

if __name__ == "__main__":
    main()

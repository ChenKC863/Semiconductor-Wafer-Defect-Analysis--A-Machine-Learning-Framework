# Semiconductor-Wafer-Defect-Analysis--A-Machine-Learning-Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

End-to-end machine learning pipeline for **9‑class wafer defect classification** and **per‑class anomaly detection**, designed for semiconductor manufacturing quality control. The framework includes model training, ONNX conversion with runtime verification, and a SQLite-based monitoring system.

## Features

- **Classification**: EfficientNetV2 (S/M) backbone with **coordinate channels** (X, Y, radial) to inject spatial awareness.
- **Training Strategy**: Two‑phase training (head only → fine‑tune last 40 layers) with data augmentation and class weights.
- **Anomaly Detection**: Per‑class PCA (1280D → 32D) + Isolation Forest to detect outliers within each predicted defect type.
- **Deployment Ready**: ONNX conversion with **ONNX Runtime verification** (max difference < 1e‑4). All inference artifacts (model, label encoder, variant info) are saved.
- **Monitoring**: Results stored in SQLite database for easy querying and dashboard integration.

## Dataset

- **Source**: [Multi‑class Semiconductor Wafer Image Dataset](https://www.kaggle.com/datasets/drtawfikrrahman/multi-class-semiconductor-wafer-image-dataset)
- **Categories**: Center, Donut, Edge‑Loc, Edge‑Ring, Local, Near‑Full, Normal, Random, Scratch
- **Split**: 5607 training, 702 validation, 702 test images (623 samples per class for training, 78 for validation/test)

## Model Architecture

- **Backbone**: EfficientNetV2‑S (384×384) or EfficientNetV2‑M (480×480) pretrained on ImageNet
- **Input**: RGB + coordinate channels → 6‑channel input (projected to 3 channels for the pretrained backbone)
- **Classification Head**: Two dense layers (256, 128) with dropout and L2 regularization, softmax output

## Anomaly Detection Pipeline

1. Extract 1280‑D features from the backbone.
2. For each predicted class, reduce dimensionality with PCA (32 components).
3. Train an Isolation Forest on the reduced features (contamination = 0.01).
4. During inference, compute anomaly score; flag if score < 0.

## ONNX Conversion & Verification

- Converted Keras models to ONNX (opset 15) using `tf2onnx`.
- Verified with ONNX Runtime: max difference between Keras and ONNX outputs **< 1e‑4** for both variants.

## Results

| Variant | Test Accuracy | Max Difference (ONNX vs Keras) | Anomaly Rate (Train) |
|---------|---------------|-------------------------------|----------------------|
| S       | 98.42%         | 3.6e‑7                        | 0.54% – 3.87%        |
| M       | 98.42%         | 1.13e‑6                       | 0.90% – 1.79%        |

Detailed per‑class anomaly rates and variance plots are available in the `anomaly_reports/` folder.

## Getting Started

### Requirements

- Python 3.9+
- TensorFlow 2.19.1
- ONNX Runtime 1.24.4
- See `requirements.txt` for full list.

### Installation

```bash
git clone https://github.com/your-username/Semiconductor-Wafer-Defect-Analysis-A-Machine-Learning-Framework.git
cd Semiconductor-Wafer-Defect-Analysis-A-Machine-Learning-Framework
pip install -r requirements.txt


Inference Example
python
from src.predict import predict
pred_class, confidence = predict("path/to/image.jpg")
print(f"Predicted: {pred_class}, Confidence: {confidence:.4f}")
ONNX Inference
python
import onnxruntime as ort
import numpy as np
from src.preprocess import preprocess_image

sess = ort.InferenceSession("model/efficientnetv2-m.onnx")
input_tensor = preprocess_image("image.jpg", target_size=480)
output = sess.run(None, {"input": input_tensor})[0]
pred_class = np.argmax(output)

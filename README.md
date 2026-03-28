# Wafer Defect Classification with ONNX

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Pre‑trained EfficientNetV2 model for 9‑class wafer defect classification, converted to ONNX and ready for inference.

## Quick Start (with Docker)

Pull the image and run:

```bash
docker run --rm -v /path/to/image.jpg:/data/test.jpg yourdockerhubusername/wafer-model:latest /data/test.jpg
Output: predicted class and confidence.
```

## Local Inference (without Docker)
```bash
pip install -r requirements.txt
python infer.py /path/to/image.jpg
```

## Model Variants
•	S (384×384)  
•	M (480×480)  
To use S/M variant, set environment variable MODEL_VARIANT=S(or M) before running, or build with --build-arg MODEL_VARIANT=S(or M).

## Training
Trained on Kaggle. See the-defect-analysis-of-wafer.ipynb (not included in this repo) for training code.

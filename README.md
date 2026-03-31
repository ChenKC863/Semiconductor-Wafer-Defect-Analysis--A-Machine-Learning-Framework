# Wafer Defect Classification with ONNX

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Pre‑trained EfficientNetV2 model for 9‑class wafer defect classification, converted to ONNX and ready for inference.

## Dataset

- **Source**: [Multi‑class Semiconductor Wafer Image Dataset](https://www.kaggle.com/datasets/drtawfikrrahman/multi-class-semiconductor-wafer-image-dataset)  
Note: The dataset is used under the Kaggle Dataset Terms of Service for non‑commercial purposes.

## Model Variants
•	S (384×384)  
•	M (480×480)  
To use S/M variant, set environment variable MODEL_VARIANT=S(or M) before running, or build with --build-arg MODEL_VARIANT=S(or M).

## Training
Trained on Kaggle. See the-defect-analysis-of-wafer.ipynb (not included in this repo) for training code.

## Local Inference
```bash
pip install -r requirements.txt
python infer.py /path/to/image.jpg
```
• `preprocess.py` is a module imported by `infer.py`, providing image preprocessing and coordinate channel generation functions.

• During inference, `infer.py` calls functions such as `preprocess_image_pil`. If `preprocess.py` is missing, inference will fail.

## Quick Start (with Docker)

The pre‑built images are available on [Docker Hub](https://hub.docker.com/r/steven710382/wafer-model/tags).

Pull and run the S variant:

```bash
docker pull steven710382/wafer-model:S
docker run --rm -v /path/to/your/image.jpg:/data/test.jpg steven710382/wafer-model:S /data/test.jpg
ˋˋˋ
• For the M variant, replace :S with :M.

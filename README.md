# Wafer Defect Classification with ONNX

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker Pulls](https://img.shields.io/docker/pulls/steven710382/wafer-model)](https://hub.docker.com/r/steven710382/wafer-model)

Pre‑trained EfficientNetV2 model for 9‑class wafer defect classification, converted to ONNX and ready for inference.

## Dataset

- **Source**: [Multi‑class Semiconductor Wafer Image Dataset](https://www.kaggle.com/datasets/drtawfikrrahman/multi-class-semiconductor-wafer-image-dataset)  
  *Note: The dataset is used under the Kaggle Dataset Terms of Service for non‑commercial purposes.*

## 📌 Overview

This project provides an **end‑to‑end machine learning framework** for semiconductor wafer defect analysis. It includes:

- **Deep learning classification** using EfficientNetV2 (S and M variants) with coordinate channel augmentation and two‑phase training.
- **ONNX model conversion** for efficient inference.
- **Per‑class anomaly detection** using PCA + Isolation Forest, storing results in SQLite.
- **REST API** (FastAPI) for real‑time wafer image classification, with optional LLM commentary via Ollama.
- **Java client** demonstrating cross‑language REST API consumption.
- **Kubernetes deployment** manifests (Deployment + Service).
- **Natural language query interface** (Streamlit + Ollama) that allows users to query the defect database using plain English/Chinese.

All models and results are version‑controlled using **Git LFS**.

## 📁 Project Structure

The complete directory structure is as follows (also available in [Semiconductor-Wafer-Defect-Analysis-A-Machine-Learning-Framework.txt](https://github.com/ChenKC863/Semiconductor-Wafer-Defect-Analysis--A-Machine-Learning-Framework/blob/main/Semiconductor-Wafer-Defect-Analysis-A-Machine-Learning-Framework.txt))


## Model Variants

- **S** – input size 384×384  
- **M** – input size 480×480  

To use a specific variant, set the environment variable `MODEL_VARIANT=S` or `MODEL_VARIANT=M` before running, or build with `--build-arg MODEL_VARIANT=S` or `--build-arg MODEL_VARIANT=M`.

## Training

The model was trained on Kaggle using dual Tesla T4 GPUs. The training notebook (`the-defect-analysis-of-wafer.ipynb`) is included in this repository for reference.

## Local Inference (without Docker)

```bash
pip install -r requirements.txt
python infer.py /path/to/image.jpg
```
• `preprocess.py` is a module imported by `infer.py`, providing image preprocessing and coordinate channel generation functions.

• During inference, `infer.py` calls functions such as `preprocess_image_pil`. If `preprocess.py` is missing, inference will fail.

## Quick Start (with Docker)

The pre‑built images are available on [Docker Hub](https://hub.docker.com/r/steven710382/wafer-model/tags).

Pull and run the S variant (if for the M variant, replace :S with :M):

```bash
docker pull steven710382/wafer-model:S
docker run --rm -v /path/to/your/image.jpg:/data/test.jpg steven710382/wafer-model:S /data/test.jpg
> **Note**: Replace `/path/to/your/image.jpg` with the absolute path to your image file.
```
The output will show the predicted class and confidence.    

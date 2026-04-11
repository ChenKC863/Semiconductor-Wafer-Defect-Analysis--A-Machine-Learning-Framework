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

The complete directory structure is as follows (also available in [Semiconductor-Wafer-Defect-Analysis-A-Machine-Learning-Framework.txt](https://github.com/ChenKC863/Semiconductor-Wafer-Defect-Analysis--A-Machine-Learning-Framework/blob/main/Semiconductor-Wafer-Defect-Analysis-A-Machine-Learning-Framework.txt)):

```text
SEMICONDUCTOR_WAFER/  # Project root directory
├── .github/
│   └── workflows/
│       └── docker-build.yml        # GitHub Actions CI/CD
├── src/
│   ├── preprocess.py               # Preprocessing + coordinate channels
│   ├── predict.py                  # (Retained, not used in deployment)
│   ├── train.py                    # Training script (reference)
│   └── anomaly.py                  # Anomaly detection (reference)
├── model/
│   ├── S/
│   │   ├── best_model.onnx         # ONNX model (inference core)
│   │   ├── label_encoder.pkl       # Category encoder
│   │   └── variant.txt             # Model variant info
│   └── M/
│       ├── best_model.onnx
│       ├── label_encoder.pkl
│       └── variant.txt
├── wafer_llm_query/                # Natural language query interface
│   ├── app.py                      # Streamlit main application
│   ├── llm_client.py               # Ollama / OpenAI client
│   ├── sql_executor.py             # Safe SQL executor (SELECT only)
│   └── requirements.txt            # Extra dependencies (if any)
├── java_client/                    # Java example client for REST API
│   ├── WaferDefectClient.class     # Compiled Java bytecode
│   └── WaferDefectClient.java      # Java source code
├── k8s/                            # Kubernetes deployment files
│   ├── deployment.yaml             # Deployment definition (2 replicas, probes)
│   └── service.yaml                # LoadBalancer service
├── inference_api.py                # FastAPI REST service (ONNX + Ollama)
├── Dockerfile                      # Docker image recipe (for inference_api.py)
├── infer.py                        # Original inference script (ONNX Runtime)
├── requirements.txt                # All Python dependencies
├── .gitignore
├── .gitattributes                  # (Optional) for Git LFS
└── README.md
```
---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/ChenKC863/Semiconductor-Wafer-Defect-Analysis--A-Machine-Learning-Framework.git
```

### 2. Set up Python environment
```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 3. Run local inference using the ONNX model (without Docker)
```bash
python infer.py /path/to/wafer_image.jpg # please change the directory of image 
```
Model Variants

- **S** – input size 384×384  
- **M** – input size 480×480  

To use a specific variant, set the environment variable `MODEL_VARIANT=S` or `MODEL_VARIANT=M` before running, or build with `--build-arg MODEL_VARIANT=S` or `--build-arg MODEL_VARIANT=M`.

**Note**:

• `preprocess.py` is a module imported by `infer.py`, providing image preprocessing and coordinate channel generation functions.

• During inference, `infer.py` calls functions such as `preprocess_image_pil`. If `preprocess.py` is missing, inference will fail.

### 4. Start the REST API server
```bash
set ONNX_MODEL_PATH=model/S/best_model.onnx
set LABEL_ENCODER_PATH=model/S/label_encoder.pkl
set VARIANT_PATH=model/S/variant.txt
uvicorn inference_api:app --host 0.0.0.0 --port 8000 --reload
```
Test with **curl** :
```bash
curl -X POST -F "file=@test.jpg" http://localhost:8000/predict
```
### 5. Quick Start (with Docker)

The pre‑built images are available on [Docker Hub](https://hub.docker.com/r/steven710382/wafer-model/tags).

Pull and run the S variant (if for the M variant, replace :S with :M):

### Run with pre-built Docker image (recommended)
```bash
docker pull steven710382/wafer-model:S
docker run --rm -v /path/to/your/image.jpg:/data/test.jpg steven710382/wafer-model:S /data/test.jpg
```

### Build an image from source code
```bash
docker build --build-arg MODEL_VARIANT=S -t wafer-model:S .
docker run --rm -v /path/to/your/image.jpg:/data/test.jpg wafer-model:S /data/test.jpg
```

**Note**: Replace `/path/to/your/image.jpg` with the absolute path to your image file.

The output will show the predicted class and confidence.


### 6. Launch natural language query interface
First install and start Ollama, then pull a model:

```bash
ollama serve
ollama pull llama3.2:3b
```
Then run Streamlit:
```bash
streamlit run wafer_llm_query/app.py
```
Open http://localhost:8501 and start querying (e.g., “Donut 類別中，異常分數最高的前 5 筆資料是哪幾張圖片？”).

## 🧪 Example Results
Classification output (sample)
```text
Predicted class: Center
Confidence: 0.9622
```


## Training

The model was trained on Kaggle using dual Tesla T4 GPUs. The training notebook (`the-defect-analysis-of-wafer.ipynb`) is included in this repository for reference.

 

import os
import numpy as np
import joblib
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io
import sys
import requests
import tempfile

sys.path.append(os.path.dirname(__file__))
from src.preprocess import preprocess_image_pil

app = FastAPI(title="Wafer Defect Classifier with LLM")

# ---------- Loading ONNX Model ----------
ONNX_PATH = os.environ.get("ONNX_MODEL_PATH")
ENCODER_PATH = os.environ.get("LABEL_ENCODER_PATH")
VARIANT_PATH = os.environ.get("VARIANT_PATH")

if not ONNX_PATH or not ENCODER_PATH or not VARIANT_PATH:
    raise RuntimeError("Missing required environment variables")

with open(VARIANT_PATH, "r") as f:
    variant = f.readline().strip()
    target_size = int(f.readline().strip())

print(f"Loading {variant} ONNX model from {ONNX_PATH}")
sess = ort.InferenceSession(ONNX_PATH)
label_encoder = joblib.load(ENCODER_PATH)
classes = label_encoder.classes_.tolist()

# ---------- Ollama Setting ----------
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")

def ask_ollama(prompt: str) -> str:
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }, timeout=30)
        resp.raise_for_status()
        return resp.json().get("response", "")
    except Exception as e:
        return f"LLM 呼叫失敗: {str(e)}"

# ---------- API Endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok", "variant": variant, "target_size": target_size}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # create a temporary file (automatically created in the system temp directory and deleted after use)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name
        
        input_tensor = preprocess_image_pil(tmp_path, target_size)
        onnx_output = sess.run(None, {"input": input_tensor})[0]
        probs = onnx_output[0]
        pred_idx = int(np.argmax(probs))
        pred_class = classes[pred_idx]
        confidence = float(probs[pred_idx])

        os.unlink(tmp_path)  # delete the temporary file
        return {
            "predicted_class": pred_class,
            "confidence": confidence,
            "probabilities": {cls: float(probs[i]) for i, cls in enumerate(classes)}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_with_llm")
async def predict_with_llm(file: UploadFile = File(...)):
    try:
        # The classification section is the same as above.
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        temp_path = f"/tmp/temp_{file.filename}"
        img.save(temp_path)

        input_tensor = preprocess_image_pil(temp_path, target_size)
        onnx_output = sess.run(None, {"input": input_tensor})[0]
        probs = onnx_output[0]
        pred_idx = int(np.argmax(probs))
        pred_class = classes[pred_idx]
        confidence = float(probs[pred_idx])

        os.remove(temp_path)

        prompt = f"""
        你是一個半導體晶圓缺陷分析專家。一張晶圓影像被分類為「{pred_class}」缺陷類別，信心度為 {confidence:.2%}。
        請簡短說明此類缺陷的可能成因、對晶圓良率的影響，以及建議的檢查或改善措施。
        """
        llm_response = ask_ollama(prompt)

        return {
            "predicted_class": pred_class,
            "confidence": confidence,
            "llm_analysis": llm_response,
            "probabilities": {cls: float(probs[i]) for i, cls in enumerate(classes)}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
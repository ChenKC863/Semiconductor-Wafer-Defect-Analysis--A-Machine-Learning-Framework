FROM python:3.10.6-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src/preprocess.py /app/src/preprocess.py
COPY ./inference_api.py /app/inference_api.py
COPY ./model /app/model

ARG MODEL_VARIANT=S
ENV MODEL_VARIANT=${MODEL_VARIANT}
ENV ONNX_MODEL_PATH=/app/model/${MODEL_VARIANT}/best_model.onnx
ENV LABEL_ENCODER_PATH=/app/model/${MODEL_VARIANT}/label_encoder.pkl
ENV VARIANT_PATH=/app/model/${MODEL_VARIANT}/variant.txt

ENV OLLAMA_URL=http://host.docker.internal:11434/api/generate
ENV OLLAMA_MODEL=llama3

EXPOSE 8000

CMD ["uvicorn", "inference_api:app", "--host", "0.0.0.0", "--port", "8000"]
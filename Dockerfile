FROM python:3.10.6-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ./src/preprocess.py /app/src/preprocess.py
COPY ./infer.py /app/infer.py
COPY ./model /app/model
ARG MODEL_VARIANT=S
ENV MODEL_VARIANT=${MODEL_VARIANT}
ENV ONNX_MODEL_PATH=/app/model/${MODEL_VARIANT}/best_model.onnx
ENV LABEL_ENCODER_PATH=/app/model/${MODEL_VARIANT}/label_encoder.pkl
ENV VARIANT_PATH=/app/model/${MODEL_VARIANT}/variant.txt
ENTRYPOINT ["python", "infer.py"]
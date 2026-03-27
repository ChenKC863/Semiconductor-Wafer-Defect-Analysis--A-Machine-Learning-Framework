# Wafer Defect Classification with ONNX

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Pre‑trained EfficientNetV2 model for 9‑class wafer defect classification, converted to ONNX and ready for inference.

## Quick Start (with Docker)

Pull the image and run:

```bash
docker run --rm -v /path/to/image.jpg:/data/test.jpg yourdockerhubusername/wafer-model:latest /data/test.jpg

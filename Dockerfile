FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy lightgbm "scikit-learn>=1.8.0" onnxruntime transformers==4.46.3 sentencepiece protobuf tqdm

COPY features.py .
COPY main.py .
COPY models/ models/

ENTRYPOINT ["python", "main.py"]

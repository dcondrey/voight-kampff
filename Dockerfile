FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir numpy lightgbm tqdm

COPY features.py .
COPY main.py .
COPY models/ models/

ENTRYPOINT ["python", "main.py"]

"""Generate DeBERTa predictions on val.jsonl using Modal GPU.

Usage:
    modal run predict_deberta_val.py

Saves predictions to models/deberta_val_probs.npy
"""

import json
import logging
from pathlib import Path

import modal
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

app = modal.App("vk-deberta-predict")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers==4.46.3",
        "onnxruntime",
        "numpy",
        "sentencepiece",
        "protobuf",
    )
)

volume = modal.Volume.from_name("vk-models", create_if_missing=True)
MOUNT_PATH = "/vol"


@app.function(image=image, gpu="A10G", timeout=1800, volumes={MOUNT_PATH: volume})
def predict_batch(texts: list[str]) -> list[float]:
    import logging

    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer

    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    model_dir = Path(f"{MOUNT_PATH}/deberta_onnx")
    onnx_path = model_dir / "model.onnx"

    session = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    input_names = {n.name for n in session.get_inputs()}

    log.info("Loaded model, predicting %d texts...", len(texts))
    all_probs = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(batch, truncation=True, max_length=512, padding=True, return_tensors="np")
        feed = {k: v for k, v in encoded.items() if k in input_names}
        logits = session.run(None, feed)[0]
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        all_probs.extend(probs[:, 1].tolist())

    log.info("Done. %d predictions.", len(all_probs))
    return all_probs


@app.local_entrypoint()
def main():
    val_path = Path("data/val.jsonl")

    log.info("Loading val data...")
    texts = []
    ids = []
    labels = []
    with open(val_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                texts.append(r["text"])
                ids.append(r["id"])
                labels.append(r["label"])

    log.info("Val: %d texts", len(texts))

    probs = predict_batch.remote(texts)
    probs = np.array(probs, dtype=np.float64)
    labels = np.array(labels, dtype=np.int32)

    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
    preds = (probs > 0.5).astype(int)
    log.info("DeBERTa val AUC=%.4f, F1=%.4f, Acc=%.4f",
             roc_auc_score(labels, probs),
             f1_score(labels, preds, average="macro"),
             accuracy_score(labels, preds))

    output_path = Path("models/deberta_val_probs.npy")
    np.save(output_path, probs)
    log.info("Saved to %s", output_path)

    ids_path = Path("models/deberta_val_ids.json")
    with open(ids_path, "w") as f:
        json.dump(ids, f)
    log.info("Saved IDs to %s", ids_path)

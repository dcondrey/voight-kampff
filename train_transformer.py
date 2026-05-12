"""Fine-tune DeBERTa-v3-base for AI text detection on Modal.

Usage:
    modal run train_transformer.py

Downloads the trained ONNX model to ./models/deberta_quantized/
"""

import json
import logging
from pathlib import Path

import modal

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

app = modal.App("voight-kampff-deberta")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "transformers==4.46.3",
        "datasets",
        "accelerate",
        "optimum[onnxruntime]",
        "scikit-learn",
        "numpy",
        "sentencepiece",
        "protobuf",
    )
)

volume = modal.Volume.from_name("vk-models", create_if_missing=True)

MOUNT_PATH = "/vol"
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 512


@app.function(
    image=image,
    gpu="A10G",
    timeout=7200,
    volumes={MOUNT_PATH: volume},
)
def train_and_export(train_data: list[dict], val_data: list[dict]):
    import logging

    import numpy as np
    import torch
    from datasets import Dataset
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from optimum.onnxruntime.quantization import ORTQuantizer
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    output_dir = f"{MOUNT_PATH}/deberta_checkpoints"
    final_dir = f"{MOUNT_PATH}/deberta_onnx"
    quantized_dir = f"{MOUNT_PATH}/deberta_quantized"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )

    def make_dataset(records):
        texts = [r["text"] for r in records]
        labels = [r["label"] for r in records]
        ds = Dataset.from_dict({"text": texts, "label": labels})

        def tokenize(batch):
            return tokenizer(
                batch["text"], truncation=True, max_length=MAX_LENGTH, padding=False
            )

        return ds.map(tokenize, batched=True, remove_columns=["text"])

    train_ds = make_dataset(train_data)
    val_ds = make_dataset(val_data)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="macro"),
            "auc": roc_auc_score(labels, probs),
        }

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="auc",
        greater_is_better=True,
        fp16=True,
        dataloader_num_workers=4,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    trainer.train()

    eval_results = trainer.evaluate()
    log.info("Final eval results: %s", eval_results)

    best_dir = f"{output_dir}/best_pytorch"
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    log.info("Exporting to ONNX...")
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        best_dir, export=True
    )
    ort_model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    log.info("Quantizing ONNX model...")
    quantizer = ORTQuantizer.from_pretrained(final_dir)
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=quantized_dir, quantization_config=qconfig)
    tokenizer.save_pretrained(quantized_dir)

    volume.commit()

    onnx_size = sum(
        f.stat().st_size for f in Path(final_dir).rglob("*") if f.is_file()
    )
    quant_size = sum(
        f.stat().st_size for f in Path(quantized_dir).rglob("*") if f.is_file()
    )
    log.info("ONNX model size: %.1f MB", onnx_size / 1e6)
    log.info("Quantized model size: %.1f MB", quant_size / 1e6)

    return eval_results


@app.function(image=image, volumes={MOUNT_PATH: volume})
def download_model():
    """Download quantized model files from volume."""
    import logging

    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    quantized_dir = Path(f"{MOUNT_PATH}/deberta_quantized")
    files = {}
    for f in quantized_dir.rglob("*"):
        if f.is_file():
            rel = f.relative_to(quantized_dir)
            files[str(rel)] = f.read_bytes()
            log.info("  %s (%.1f MB)", rel, f.stat().st_size / 1e6)
    return files


@app.local_entrypoint()
def main():
    data_dir = Path("data")
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"

    log.info("Loading data...")
    train_data = []
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                train_data.append({"text": r["text"], "label": r["label"]})

    val_data = []
    with open(val_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                val_data.append({"text": r["text"], "label": r["label"]})

    log.info("Train: %d, Val: %d", len(train_data), len(val_data))

    eval_results = train_and_export.remote(train_data, val_data)
    log.info("Training complete. Results: %s", eval_results)

    log.info("Downloading quantized model...")
    output_dir = Path("models/deberta_quantized")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = download_model.remote()
    for rel_path, content in files.items():
        out_path = output_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(content)
        log.info("  Saved %s (%.1f MB)", rel_path, len(content) / 1e6)

    log.info("Model saved to %s/", output_dir)

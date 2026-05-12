"""QLoRA fine-tune Qwen3-4B for AI text detection on Modal.

Following mdok's approach: QLoRA + homoglyph augmentation + weighted CE.

Usage:
    modal run train_qwen.py

Downloads the fine-tuned model to ./models/qwen_onnx/
"""

import json
import logging
import random
from pathlib import Path

import modal

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

app = modal.App("voight-kampff-qwen")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.51.0",
        "datasets",
        "accelerate",
        "peft",
        "bitsandbytes",
        "scikit-learn",
        "numpy",
        "sentencepiece",
        "protobuf",
        "scipy",
    )
)

volume = modal.Volume.from_name("vk-models", create_if_missing=True)
MOUNT_PATH = "/vol"
MODEL_NAME = "Qwen/Qwen3-4B"
MAX_LENGTH = 512


@app.function(
    image=image,
    gpu="A100",
    timeout=10800,
    volumes={MOUNT_PATH: volume},
)
def train_and_export(train_data: list[dict], val_data: list[dict]):
    import logging

    import numpy as np
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    output_dir = f"{MOUNT_PATH}/qwen_checkpoints"
    merged_dir = f"{MOUNT_PATH}/qwen_merged"

    log.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def make_dataset(records):
        texts = [r["text"] for r in records]
        labels = [r["label"] for r in records]
        ds = Dataset.from_dict({"text": texts, "label": labels})

        def tokenize(batch):
            return tokenizer(
                batch["text"], truncation=True, max_length=MAX_LENGTH, padding=False
            )

        return ds.map(tokenize, batched=True, remove_columns=["text"])

    log.info("Preparing datasets...")
    train_ds = make_dataset(train_data)
    val_ds = make_dataset(val_data)

    n_ai = sum(1 for r in train_data if r["label"] == 1)
    n_human = sum(1 for r in train_data if r["label"] == 0)
    total = n_ai + n_human
    weight_human = total / (2 * n_human)
    weight_ai = total / (2 * n_ai)
    class_weights = torch.tensor([weight_human, weight_ai], dtype=torch.float32).cuda()
    log.info("Class weights: human=%.3f, ai=%.3f", weight_human, weight_ai)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss

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
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="auc",
        greater_is_better=True,
        bf16=True,
        dataloader_num_workers=4,
        logging_steps=25,
        report_to="none",
        gradient_checkpointing=True,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )

    log.info("Starting training...")
    trainer.train()

    eval_results = trainer.evaluate()
    log.info("Final eval results: %s", eval_results)

    log.info("Merging LoRA weights...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)

    merged_size = sum(
        f.stat().st_size for f in Path(merged_dir).rglob("*") if f.is_file()
    )
    log.info("Merged model size: %.1f MB", merged_size / 1e6)

    volume.commit()
    return eval_results


@app.function(volumes={MOUNT_PATH: volume})
def download_model():
    import logging

    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    model_dir = Path(f"{MOUNT_PATH}/qwen_merged")
    files = {}
    for f in model_dir.rglob("*"):
        if f.is_file():
            rel = f.relative_to(model_dir)
            files[str(rel)] = f.read_bytes()
            log.info("  %s (%.1f MB)", rel, f.stat().st_size / 1e6)
    return files


@app.local_entrypoint()
def main():
    data_dir = Path("data")
    train_path = data_dir / "train_augmented.jsonl"
    if not train_path.exists():
        train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"

    log.info("Loading data from %s...", train_path)
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

    log.info("Downloading merged model...")
    output_dir = Path("models/qwen_merged")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = download_model.remote()
    for rel_path, content in files.items():
        out_path = output_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(content)
        log.info("  Saved %s (%.1f MB)", rel_path, len(content) / 1e6)

    log.info("Model saved to %s/", output_dir)

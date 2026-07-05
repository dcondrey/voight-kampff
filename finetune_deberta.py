"""Fine-tune DeBERTa-v2 on PAN Generative AI Detection training data.

Converts ONNX weights → PyTorch, fine-tunes, exports back to ONNX.

Usage:
    uv run python finetune_deberta.py --data data/train.jsonl --val data/val.jsonl
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import onnx
import onnx.numpy_helper
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DebertaV2ForSequenceClassification,
    DebertaV2Config,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

MODEL_DIR = Path("models/deberta_onnx")


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(
            texts, truncation=True, max_length=max_length,
            padding="max_length", return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def load_onnx_weights_into_pytorch(onnx_path, pytorch_model):
    """Convert ONNX initializer weights into a PyTorch state dict."""
    onnx_model = onnx.load(str(onnx_path))
    onnx_weights = {}
    for init in onnx_model.graph.initializer:
        onnx_weights[init.name] = onnx.numpy_helper.to_array(init)

    state_dict = pytorch_model.state_dict()

    # Build index of unnamed ONNX MatMul tensors (transposed weight matrices)
    unnamed_tensors = []
    for name, arr in onnx_weights.items():
        if name.startswith("onnx::MatMul_"):
            unnamed_tensors.append((name, arr))

    loaded, skipped = 0, 0
    unnamed_idx = 0
    for name, param in state_dict.items():
        if name in onnx_weights:
            tensor = torch.from_numpy(onnx_weights[name].copy())
            if tensor.shape == param.shape:
                state_dict[name] = tensor
                loaded += 1
            else:
                log.warning("  Shape mismatch for %s: onnx=%s pt=%s",
                            name, tensor.shape, param.shape)
                skipped += 1
        elif name.endswith(".weight") and unnamed_idx < len(unnamed_tensors):
            # ONNX export transposes weight matrices and renames to onnx::MatMul_*
            onnx_name, arr = unnamed_tensors[unnamed_idx]
            tensor = torch.from_numpy(arr.copy())
            if tensor.shape == param.shape:
                state_dict[name] = tensor
                loaded += 1
                unnamed_idx += 1
            elif tensor.T.shape == param.shape:
                state_dict[name] = torch.from_numpy(arr.T.copy())
                loaded += 1
                unnamed_idx += 1
            else:
                skipped += 1
        else:
            skipped += 1

    pytorch_model.load_state_dict(state_dict, strict=False)
    log.info("  Loaded %d/%d weights from ONNX (%d skipped)",
             loaded, loaded + skipped, skipped)
    return pytorch_model


def load_data(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--val", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    log.info("Device: %s", device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    # Initialize model from config, then load ONNX weights
    log.info("Loading DeBERTa model...")
    config = DebertaV2Config.from_pretrained(str(MODEL_DIR))
    model = DebertaV2ForSequenceClassification(config)

    onnx_path = MODEL_DIR / "model.onnx"
    log.info("Converting ONNX weights to PyTorch...")
    model = load_onnx_weights_into_pytorch(onnx_path, model)
    model = model.to(device)

    # Load data
    log.info("Loading training data...")
    train_records = load_data(args.data)
    train_texts = [r["text"] for r in train_records]
    train_labels = [r["label"] for r in train_records]
    log.info("  %d training samples", len(train_records))

    val_dataset = None
    if args.val and args.val.exists():
        val_records = load_data(args.val)
        val_texts = [r["text"] for r in val_records]
        val_labels = [r["label"] for r in val_records]
        val_dataset = TextDataset(val_texts, val_labels, tokenizer, args.max_length)
        log.info("  %d validation samples", len(val_records))

    train_dataset = TextDataset(train_texts, train_labels, tokenizer, args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Training loop
    log.info("\nTraining for %d epochs (%d steps)...", args.epochs, total_steps)
    best_val_acc = 0.0
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += len(batch["labels"])

            if (step + 1) % 100 == 0:
                log.info("  Epoch %d step %d/%d  loss=%.4f  acc=%.4f",
                         epoch + 1, step + 1, len(train_loader),
                         total_loss / (step + 1), correct / total)

        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)
        log.info("Epoch %d: loss=%.4f  train_acc=%.4f", epoch + 1, avg_loss, train_acc)

        # Validation
        if val_dataset is not None:
            model.eval()
            val_correct = 0
            val_total = 0
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2)
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    preds = outputs.logits.argmax(dim=-1)
                    val_correct += (preds == batch["labels"]).sum().item()
                    val_total += len(batch["labels"])
            val_acc = val_correct / val_total
            log.info("  Val acc=%.4f (%d/%d)", val_acc, val_correct, val_total)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                log.info("  New best model saved (val_acc=%.4f)", val_acc)

    # Load best model and save PyTorch checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
        log.info("\nLoaded best model (val_acc=%.4f)", best_val_acc)

    pt_path = args.output_dir / "model_finetuned.pt"
    torch.save(model.state_dict(), pt_path)
    log.info("Saved PyTorch checkpoint to %s", pt_path)

    # Export to ONNX
    model = model.to("cpu")
    model.eval()

    log.info("Exporting to ONNX...")
    output_onnx = args.output_dir / "model_finetuned.onnx"
    dummy_input = tokenizer(
        "This is a test.", truncation=True, max_length=args.max_length,
        padding="max_length", return_tensors="pt",
    )
    input_names = list(dummy_input.keys())

    torch.onnx.export(
        model,
        tuple(dummy_input[k] for k in input_names),
        str(output_onnx),
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes={
            name: {0: "batch_size", 1: "sequence"} for name in input_names
        } | {"logits": {0: "batch_size"}},
        opset_version=14,
    )
    log.info("Exported ONNX model to %s", output_onnx)

    # Verify ONNX model
    import onnxruntime as ort
    session = ort.InferenceSession(str(output_onnx), providers=["CPUExecutionProvider"])
    test_enc = tokenizer("Test sentence.", truncation=True, max_length=64,
                         padding=True, return_tensors="np")
    feed = {k: v for k, v in test_enc.items() if k in {n.name for n in session.get_inputs()}}
    result = session.run(None, feed)
    log.info("ONNX verification: output shape=%s", result[0].shape)
    log.info("Done!")


if __name__ == "__main__":
    main()

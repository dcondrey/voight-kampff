"""TIRA entry point for Voight-Kampff Generative AI Detection.

Supervised LightGBM ensemble with 30 domain-portable features.
CV AUC=0.9948, Val AUC=0.9937.

Usage (TIRA convention):
    python main.py $inputDataset/dataset.jsonl $outputDir
"""

import json
import logging
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

from features import extract_features, FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "models"


def load_models():
    config_path = MODEL_DIR / "vk_config.json"
    with open(config_path) as f:
        config = json.load(f)
    models = []
    for seed in config["seeds"]:
        path = MODEL_DIR / f"vk_seed{seed}.txt"
        models.append(lgb.Booster(model_file=str(path)))
    return models, config


def main():
    if len(sys.argv) < 3:
        log.error("Usage: python main.py <input_jsonl> <output_dir>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading models...")
    models, config = load_models()

    log.info("Reading input from %s", input_path)
    records = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    log.info("  %d records", len(records))

    log.info("Extracting features and predicting...")
    output_path = output_dir / "predictions.jsonl"
    with open(output_path, "w", encoding="utf-8") as out:
        for r in records:
            text = r.get("text", "")
            feats = np.array([extract_features(text)], dtype=np.float32)
            prob = float(np.mean([m.predict(feats)[0] for m in models]))
            prediction = {"id": r["id"], "is_human": round(1.0 - prob, 4)}
            out.write(json.dumps(prediction) + "\n")

    log.info("Wrote %d predictions to %s", len(records), output_path)


if __name__ == "__main__":
    main()

"""Train Voight-Kampff detector on PAN generative AI detection dataset.

Usage:
    uv run python train.py --data data/dataset.jsonl
    uv run python train.py --data data/  # directory of JSONL files
"""

import argparse
import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, f1_score, accuracy_score
)
from sklearn.model_selection import StratifiedKFold

from features import extract_features_batch, FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def load_data(path):
    """Load JSONL data from file or directory."""
    path = Path(path)
    records = []
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.glob("*.jsonl"))
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def optimize_threshold(y_true, probs):
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.20, 0.80, 0.005):
        preds = (probs > t).astype(int)
        f1 = f1_score(y_true, preds, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1


def evaluate(y_true, probs, threshold=0.5):
    preds = (probs > threshold).astype(int)
    auc = roc_auc_score(y_true, probs)
    brier = 1.0 - brier_score_loss(y_true, probs)
    f1 = f1_score(y_true, preds, average="macro")
    acc = accuracy_score(y_true, preds)
    return {"auc": auc, "brier_complement": brier, "f1_macro": f1, "accuracy": acc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--seeds", type=str, default="42,123,456,789,1024")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]

    log.info("=" * 60)
    log.info("  VOIGHT-KAMPFF: Generative AI Detection Training")
    log.info("=" * 60)

    # Load data
    log.info("\nLoading data from %s...", args.data)
    records = load_data(args.data)
    log.info("  %d records loaded", len(records))

    texts = [r["text"] for r in records]
    labels = np.array([r["label"] for r in records], dtype=np.int32)

    log.info("  Human: %d, AI: %d", (labels == 0).sum(), (labels == 1).sum())

    # Extract features
    log.info("\nExtracting features...")
    X = extract_features_batch(texts)
    log.info("  Feature matrix: %s", X.shape)

    # Cross-validated threshold optimization
    log.info("\n--- Cross-Validated Threshold Optimization ---")
    all_oof_probs = np.zeros(len(labels))
    all_oof_counts = np.zeros(len(labels))

    for seed in seeds:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, labels)):
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "boosting_type": "gbdt",
                "num_leaves": 63,
                "learning_rate": 0.05,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "min_child_samples": 20,
                "reg_lambda": 1.0,
                "reg_alpha": 0.5,
                "verbose": -1,
                "seed": seed,
                "n_jobs": -1,
            }
            train_set = lgb.Dataset(X[train_idx], label=labels[train_idx],
                                    feature_name=FEATURE_NAMES)
            val_set = lgb.Dataset(X[val_idx], label=labels[val_idx],
                                  feature_name=FEATURE_NAMES, reference=train_set)
            model = lgb.train(
                params, train_set, num_boost_round=1500,
                valid_sets=[val_set],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
            )
            fold_probs = model.predict(X[val_idx])
            all_oof_probs[val_idx] += fold_probs
            all_oof_counts[val_idx] += 1

    oof_probs = all_oof_probs / np.maximum(all_oof_counts, 1)
    cv_thresh, cv_f1 = optimize_threshold(labels, oof_probs)
    cv_metrics = evaluate(labels, oof_probs, cv_thresh)
    log.info("  CV threshold=%.3f", cv_thresh)
    log.info("  CV AUC=%.4f, Brier complement=%.4f, F1=%.4f, Acc=%.4f",
             cv_metrics["auc"], cv_metrics["brier_complement"],
             cv_metrics["f1_macro"], cv_metrics["accuracy"])

    # Train final ensemble on all data
    log.info("\n--- Training Final Ensemble ---")
    models = []
    for seed in seeds:
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 63,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 20,
            "reg_lambda": 1.0,
            "reg_alpha": 0.5,
            "verbose": -1,
            "seed": seed,
            "n_jobs": -1,
        }
        train_set = lgb.Dataset(X, label=labels, feature_name=FEATURE_NAMES)
        model = lgb.train(params, train_set, num_boost_round=800)
        models.append(model)
        log.info("  Trained seed=%d", seed)

    # Save
    for i, model in enumerate(models):
        model.save_model(str(args.output_dir / f"vk_seed{seeds[i]}.txt"))

    config = {
        "threshold": float(cv_thresh),
        "seeds": seeds,
        "n_features": int(X.shape[1]),
        "feature_names": FEATURE_NAMES,
        "cv_metrics": {k: float(v) for k, v in cv_metrics.items()},
    }
    with open(args.output_dir / "vk_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Feature importance
    importances = np.mean([m.feature_importance(importance_type="gain") for m in models], axis=0)
    sorted_idx = np.argsort(importances)[::-1]
    log.info("\nFeature Importances:")
    for rank, idx in enumerate(sorted_idx, 1):
        log.info("  %2d. %-30s %12.1f", rank, FEATURE_NAMES[idx], importances[idx])

    log.info("\nModels saved to %s/", args.output_dir)


if __name__ == "__main__":
    main()

"""Retrain ensemble weights and calibrator with DeBERTa predictions.

Usage:
    uv run python calibrate_ensemble.py

Requires models/deberta_val_probs.npy from predict_deberta_val.py.
Updates models/vk_config.json, models/calibrator.pkl.
"""

import json
import logging
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from scipy.sparse import hstack as sparse_hstack
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score, accuracy_score

from features import extract_features_batch

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

MODEL_DIR = Path("models")


def load_val_data():
    records = []
    with open("data/val.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    texts = [r["text"] for r in records]
    labels = np.array([r["label"] for r in records], dtype=np.int32)
    ids = [r["id"] for r in records]
    return texts, labels, ids


def evaluate(y_true, probs, threshold=0.5):
    preds = (probs > threshold).astype(int)
    auc = roc_auc_score(y_true, probs)
    brier = 1.0 - brier_score_loss(y_true, probs)
    f1 = f1_score(y_true, preds, average="macro")
    acc = accuracy_score(y_true, preds)
    mean = (auc + brier + f1 + acc) / 4
    return {"auc": auc, "brier": brier, "f1": f1, "acc": acc, "mean": mean}


def main():
    log.info("Loading val data...")
    texts, labels, ids = load_val_data()

    log.info("Loading DeBERTa val predictions...")
    deberta_probs = np.load(MODEL_DIR / "deberta_val_probs.npy")
    deberta_ids = json.loads((MODEL_DIR / "deberta_val_ids.json").read_text())
    assert deberta_ids == ids, "ID mismatch between val data and DeBERTa predictions"

    log.info("Loading LGB models...")
    with open(MODEL_DIR / "vk_config.json") as f:
        config = json.load(f)
    lgb_models = [lgb.Booster(model_file=str(MODEL_DIR / f"vk_seed{s}.txt")) for s in config["seeds"]]

    log.info("Extracting features...")
    X_stylo = extract_features_batch(texts)
    char_vec = joblib.load(MODEL_DIR / "tfidf_char.pkl")
    word_vec = joblib.load(MODEL_DIR / "tfidf_word.pkl")
    char_svd = joblib.load(MODEL_DIR / "svd_char.pkl")
    word_svd = joblib.load(MODEL_DIR / "svd_word.pkl")
    X_char = char_svd.transform(char_vec.transform(texts)).astype(np.float32)
    X_word = word_svd.transform(word_vec.transform(texts)).astype(np.float32)
    X_combined = np.hstack([X_stylo, X_char, X_word])

    lgb_prob = np.mean([m.predict(X_combined) for m in lgb_models], axis=0)

    svm_char_vec = joblib.load(MODEL_DIR / "svm_tfidf_char.pkl")
    svm_word_vec = joblib.load(MODEL_DIR / "svm_tfidf_word.pkl")
    svm = joblib.load(MODEL_DIR / "svm_calibrated.pkl")
    X_svm = sparse_hstack([svm_char_vec.transform(texts), svm_word_vec.transform(texts)])
    svm_prob = svm.predict_proba(X_svm)[:, 1]

    lgb_svm_blend = config["ensemble_weights"][0] * lgb_prob + config["ensemble_weights"][1] * svm_prob

    log.info("\n--- Individual Model Scores ---")
    for name, probs in [("DeBERTa", deberta_probs), ("LGB", lgb_prob),
                         ("SVM", svm_prob), ("LGB+SVM", lgb_svm_blend)]:
        m = evaluate(labels, probs)
        log.info("  %-10s AUC=%.4f Brier=%.4f F1=%.4f Acc=%.4f Mean=%.4f",
                 name, m["auc"], m["brier"], m["f1"], m["acc"], m["mean"])

    log.info("\n--- Grid Search: DeBERTa weight in 3-way ensemble ---")
    best_mean, best_w = 0, 0.5
    for w_deb in np.arange(0.1, 0.9, 0.01):
        blended = w_deb * deberta_probs + (1 - w_deb) * lgb_svm_blend
        m = evaluate(labels, blended)
        if m["mean"] > best_mean:
            best_mean = m["mean"]
            best_w = w_deb

    blended_raw = best_w * deberta_probs + (1 - best_w) * lgb_svm_blend
    m = evaluate(labels, blended_raw)
    log.info("  Best DeBERTa weight: %.2f", best_w)
    log.info("  Pre-cal:  AUC=%.4f Brier=%.4f F1=%.4f Acc=%.4f Mean=%.4f",
             m["auc"], m["brier"], m["f1"], m["acc"], m["mean"])

    calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    calibrator.fit(blended_raw, labels)
    calibrated = calibrator.predict(blended_raw)
    m_cal = evaluate(labels, calibrated)
    log.info("  Post-cal: AUC=%.4f Brier=%.4f F1=%.4f Acc=%.4f Mean=%.4f",
             m_cal["auc"], m_cal["brier"], m_cal["f1"], m_cal["acc"], m_cal["mean"])

    log.info("\n--- Saving updated config and calibrator ---")
    config["deberta_weight"] = float(best_w)
    config["has_calibrator"] = True
    config["ensemble_metrics_val"] = {k: float(v) for k, v in m_cal.items()}
    with open(MODEL_DIR / "vk_config.json", "w") as f:
        json.dump(config, f, indent=2)

    joblib.dump(calibrator, MODEL_DIR / "calibrator.pkl")
    log.info("Done. Updated vk_config.json and calibrator.pkl")


if __name__ == "__main__":
    main()

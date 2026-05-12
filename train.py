"""Train Voight-Kampff detector on PAN generative AI detection dataset.

Usage:
    uv run python train.py --data data/train.jsonl --val data/val.jsonl
    uv run python train.py --data data/  # directory of JSONL files
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from scipy.sparse import hstack as sparse_hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, f1_score, accuracy_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

from features import extract_features_batch, FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 127,
    "learning_rate": 0.02,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "min_child_samples": 30,
    "max_depth": 8,
    "reg_lambda": 2.0,
    "reg_alpha": 1.0,
    "verbose": -1,
    "n_jobs": -1,
}


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


def build_tfidf_features(texts, char_vec=None, word_vec=None,
                          char_svd=None, word_svd=None, fit=False):
    """Build TF-IDF SVD features. If fit=True, fit transformers on texts."""
    if fit:
        char_tfidf = char_vec.fit_transform(texts)
        word_tfidf = word_vec.fit_transform(texts)
        char_reduced = char_svd.fit_transform(char_tfidf)
        word_reduced = word_svd.fit_transform(word_tfidf)
    else:
        char_tfidf = char_vec.transform(texts)
        word_tfidf = word_vec.transform(texts)
        char_reduced = char_svd.transform(char_tfidf)
        word_reduced = word_svd.transform(word_tfidf)
    return np.hstack([char_reduced, word_reduced]).astype(np.float32)


def build_sparse_tfidf(texts, char_vec, word_vec, fit=False):
    """Build raw sparse TF-IDF for SVM."""
    if fit:
        char_tfidf = char_vec.fit_transform(texts)
        word_tfidf = word_vec.fit_transform(texts)
    else:
        char_tfidf = char_vec.transform(texts)
        word_tfidf = word_vec.transform(texts)
    return sparse_hstack([char_tfidf, word_tfidf])


def make_tfidf_transformers():
    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5),
        max_features=50000, sublinear_tf=True, min_df=3,
    )
    word_vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2),
        max_features=30000, sublinear_tf=True, min_df=5,
    )
    char_svd = TruncatedSVD(n_components=50, random_state=42)
    word_svd = TruncatedSVD(n_components=30, random_state=42)
    return char_vec, word_vec, char_svd, word_svd


# Feature names for LGB when using stylometric + SVD features
def lgb_feature_names(n_char_svd=50, n_word_svd=30):
    names = list(FEATURE_NAMES)
    names += [f"char_svd_{i}" for i in range(n_char_svd)]
    names += [f"word_svd_{i}" for i in range(n_word_svd)]
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--val", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--seeds", type=str, default="42,123,456,789,1024")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]

    log.info("=" * 60)
    log.info("  VOIGHT-KAMPFF: Generative AI Detection Training")
    log.info("=" * 60)

    # Load training data
    log.info("\nLoading training data from %s...", args.data)
    records = load_data(args.data)
    log.info("  %d records loaded", len(records))

    texts = [r["text"] for r in records]
    labels = np.array([r["label"] for r in records], dtype=np.int32)
    log.info("  Human: %d, AI: %d", (labels == 0).sum(), (labels == 1).sum())

    # Load validation data
    val_texts, val_labels = None, None
    if args.val and args.val.exists():
        log.info("\nLoading validation data from %s...", args.val)
        val_records = load_data(args.val)
        val_texts = [r["text"] for r in val_records]
        val_labels = np.array([r["label"] for r in val_records], dtype=np.int32)
        log.info("  %d val records (Human: %d, AI: %d)",
                 len(val_records), (val_labels == 0).sum(), (val_labels == 1).sum())

    # Extract stylometric features
    log.info("\nExtracting stylometric features...")
    X_stylo = extract_features_batch(texts)
    log.info("  Stylometric feature matrix: %s", X_stylo.shape)

    if val_texts:
        X_val_stylo = extract_features_batch(val_texts)

    # ================================================================
    # TF-IDF + SVD features
    # ================================================================
    log.info("\n--- Fitting TF-IDF + SVD ---")
    char_vec, word_vec, char_svd, word_svd = make_tfidf_transformers()
    X_tfidf_svd = build_tfidf_features(
        texts, char_vec, word_vec, char_svd, word_svd, fit=True
    )
    log.info("  TF-IDF SVD features: %s", X_tfidf_svd.shape)

    X_combined = np.hstack([X_stylo, X_tfidf_svd])
    feat_names = lgb_feature_names()
    log.info("  Combined feature matrix: %s", X_combined.shape)

    if val_texts:
        X_val_tfidf_svd = build_tfidf_features(
            val_texts, char_vec, word_vec, char_svd, word_svd, fit=False
        )
        X_val_combined = np.hstack([X_val_stylo, X_val_tfidf_svd])

    # ================================================================
    # Cross-validated LGB with TF-IDF features
    # ================================================================
    log.info("\n--- Cross-Validated LGB (stylometric + TF-IDF SVD) ---")
    all_oof_probs = np.zeros(len(labels))
    all_oof_counts = np.zeros(len(labels))

    for seed in seeds:
        params = {**LGB_PARAMS, "seed": seed}
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        # For CV, refit TF-IDF per fold to avoid leakage
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_stylo, labels)):
            fold_char, fold_word, fold_csvd, fold_wsvd = make_tfidf_transformers()
            fold_texts_train = [texts[i] for i in train_idx]
            fold_texts_val = [texts[i] for i in val_idx]

            fold_tfidf_train = build_tfidf_features(
                fold_texts_train, fold_char, fold_word,
                fold_csvd, fold_wsvd, fit=True
            )
            fold_tfidf_val = build_tfidf_features(
                fold_texts_val, fold_char, fold_word,
                fold_csvd, fold_wsvd, fit=False
            )

            X_fold_train = np.hstack([X_stylo[train_idx], fold_tfidf_train])
            X_fold_val = np.hstack([X_stylo[val_idx], fold_tfidf_val])

            train_set = lgb.Dataset(X_fold_train, label=labels[train_idx],
                                    feature_name=feat_names)
            val_set = lgb.Dataset(X_fold_val, label=labels[val_idx],
                                  feature_name=feat_names, reference=train_set)
            model = lgb.train(
                params, train_set, num_boost_round=3000,
                valid_sets=[val_set],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
            )
            fold_probs = model.predict(X_fold_val)
            all_oof_probs[val_idx] += fold_probs
            all_oof_counts[val_idx] += 1

    oof_probs_lgb = all_oof_probs / np.maximum(all_oof_counts, 1)
    cv_thresh, _ = optimize_threshold(labels, oof_probs_lgb)
    cv_metrics_lgb = evaluate(labels, oof_probs_lgb, cv_thresh)
    log.info("  LGB CV threshold=%.3f", cv_thresh)
    log.info("  LGB CV AUC=%.4f, Brier=%.4f, F1=%.4f, Acc=%.4f",
             cv_metrics_lgb["auc"], cv_metrics_lgb["brier_complement"],
             cv_metrics_lgb["f1_macro"], cv_metrics_lgb["accuracy"])

    # ================================================================
    # TF-IDF SVM (on raw sparse features, not SVD-reduced)
    # ================================================================
    log.info("\n--- Training TF-IDF SVM ---")
    # Use separate vectorizers for SVM (same settings, fitted on full train)
    svm_char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5),
        max_features=50000, sublinear_tf=True, min_df=3,
    )
    svm_word_vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2),
        max_features=30000, sublinear_tf=True, min_df=5,
    )
    X_svm_sparse = build_sparse_tfidf(texts, svm_char_vec, svm_word_vec, fit=True)
    log.info("  SVM sparse features: %s", X_svm_sparse.shape)

    svm = CalibratedClassifierCV(
        LinearSVC(C=1.0, max_iter=5000), cv=5, method="isotonic"
    )
    svm.fit(X_svm_sparse, labels)
    svm_oof_probs = svm.predict_proba(X_svm_sparse)[:, 1]
    svm_metrics = evaluate(labels, svm_oof_probs, 0.5)
    log.info("  SVM train AUC=%.4f, Brier=%.4f, F1=%.4f, Acc=%.4f",
             svm_metrics["auc"], svm_metrics["brier_complement"],
             svm_metrics["f1_macro"], svm_metrics["accuracy"])

    # ================================================================
    # Train final LGB ensemble with val.jsonl early stopping
    # ================================================================
    log.info("\n--- Training Final LGB Ensemble ---")
    lgb_models = []
    for seed in seeds:
        params = {**LGB_PARAMS, "seed": seed}
        train_set = lgb.Dataset(X_combined, label=labels, feature_name=feat_names)

        callbacks = [lgb.log_evaluation(0)]
        valid_sets = [train_set]
        num_rounds = 1500

        if val_texts is not None:
            val_set = lgb.Dataset(X_val_combined, label=val_labels,
                                  feature_name=feat_names, reference=train_set)
            valid_sets = [val_set]
            callbacks.append(lgb.early_stopping(100))
            num_rounds = 3000

        model = lgb.train(
            params, train_set, num_boost_round=num_rounds,
            valid_sets=valid_sets, callbacks=callbacks,
        )
        lgb_models.append(model)
        log.info("  Trained seed=%d  best_iter=%d", seed, model.best_iteration)

    # ================================================================
    # Ensemble calibration on val set
    # ================================================================
    ensemble_weights = [0.5, 0.5]  # [lgb, svm] default
    calibrator = None

    if val_texts is not None:
        log.info("\n--- Ensemble Calibration on Val Set ---")
        lgb_val_prob = np.mean(
            [m.predict(X_val_combined) for m in lgb_models], axis=0
        )
        X_svm_val_sparse = build_sparse_tfidf(
            val_texts, svm_char_vec, svm_word_vec, fit=False
        )
        svm_val_prob = svm.predict_proba(X_svm_val_sparse)[:, 1]

        # Find optimal weights by grid search
        best_mean, best_w = 0, 0.5
        for w in np.arange(0.3, 0.8, 0.01):
            blended = w * lgb_val_prob + (1 - w) * svm_val_prob
            metrics = evaluate(val_labels, blended, 0.5)
            mean_score = np.mean(list(metrics.values()))
            if mean_score > best_mean:
                best_mean = mean_score
                best_w = w
        ensemble_weights = [float(best_w), float(1 - best_w)]
        log.info("  Optimal weights: LGB=%.2f, SVM=%.2f", best_w, 1 - best_w)

        blended_val = best_w * lgb_val_prob + (1 - best_w) * svm_val_prob
        val_metrics_pre = evaluate(val_labels, blended_val, 0.5)
        log.info("  Ensemble val (pre-cal): AUC=%.4f, Brier=%.4f, F1=%.4f, Acc=%.4f",
                 val_metrics_pre["auc"], val_metrics_pre["brier_complement"],
                 val_metrics_pre["f1_macro"], val_metrics_pre["accuracy"])

        # Isotonic calibration
        calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        calibrator.fit(blended_val, val_labels)
        calibrated_val = calibrator.predict(blended_val)
        val_metrics_post = evaluate(val_labels, calibrated_val, 0.5)
        log.info("  Ensemble val (post-cal): AUC=%.4f, Brier=%.4f, F1=%.4f, Acc=%.4f",
                 val_metrics_post["auc"], val_metrics_post["brier_complement"],
                 val_metrics_post["f1_macro"], val_metrics_post["accuracy"])

    # ================================================================
    # Save all artifacts
    # ================================================================
    log.info("\n--- Saving Models ---")
    for i, model in enumerate(lgb_models):
        model.save_model(str(args.output_dir / f"vk_seed{seeds[i]}.txt"))

    joblib.dump(char_vec, args.output_dir / "tfidf_char.pkl")
    joblib.dump(word_vec, args.output_dir / "tfidf_word.pkl")
    joblib.dump(char_svd, args.output_dir / "svd_char.pkl")
    joblib.dump(word_svd, args.output_dir / "svd_word.pkl")
    joblib.dump(svm_char_vec, args.output_dir / "svm_tfidf_char.pkl")
    joblib.dump(svm_word_vec, args.output_dir / "svm_tfidf_word.pkl")
    joblib.dump(svm, args.output_dir / "svm_calibrated.pkl")

    if calibrator is not None:
        joblib.dump(calibrator, args.output_dir / "calibrator.pkl")

    config = {
        "threshold": float(cv_thresh),
        "seeds": seeds,
        "n_stylo_features": len(FEATURE_NAMES),
        "n_char_svd": 50,
        "n_word_svd": 30,
        "feature_names": feat_names,
        "ensemble_weights": ensemble_weights,
        "has_calibrator": calibrator is not None,
        "cv_metrics_lgb": {k: float(v) for k, v in cv_metrics_lgb.items()},
    }
    with open(args.output_dir / "vk_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Feature importance
    importances = np.mean(
        [m.feature_importance(importance_type="gain") for m in lgb_models], axis=0
    )
    sorted_idx = np.argsort(importances)[::-1]
    log.info("\nTop 20 Feature Importances:")
    for rank, idx in enumerate(sorted_idx[:20], 1):
        log.info("  %2d. %-30s %12.1f", rank, feat_names[idx], importances[idx])

    log.info("\nAll models saved to %s/", args.output_dir)


if __name__ == "__main__":
    main()

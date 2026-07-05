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
from sklearn.linear_model import LogisticRegression
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
    "num_leaves": 63,
    "learning_rate": 0.015,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.6,
    "bagging_freq": 5,
    "min_child_samples": 50,
    "max_depth": 8,
    "reg_lambda": 5.0,
    "reg_alpha": 2.0,
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


def optimize_threshold_and_abstention(y_true, probs):
    """Grid search over abstention margin to maximize PAN mean.

    Threshold is fixed at 0.5 to match the PAN evaluator.
    """
    t = 0.5
    best_mean, best_margin = 0, 0.0
    for margin in np.arange(0.0, 0.15, 0.005):
        adjusted = probs.copy()
        uncertain = np.abs(adjusted - t) < margin
        adjusted[uncertain] = t
        metrics = pan_mean(y_true, adjusted, threshold=t)
        if metrics["mean"] > best_mean:
            best_mean = metrics["mean"]
            best_margin = margin
    return t, best_margin, best_mean


def compute_cat1(y_true, probs, threshold=0.5):
    """c@1: rewards abstention (prob == 0.5) over wrong answers."""
    n = len(y_true)
    if n == 0:
        return 0.0
    preds = np.where(probs > threshold, 1, np.where(probs < threshold, 0, -1))
    nc = int(np.sum((preds != -1) & (preds == y_true)))
    nu = int(np.sum(preds == -1))
    return (nc + nu * nc / n) / n


def compute_f05u(y_true, probs, threshold=0.5):
    """F0.5u: F0.5 adjusted for unanswered cases."""
    n = len(y_true)
    if n == 0:
        return 0.0
    preds = np.where(probs > threshold, 1, np.where(probs < threshold, 0, -1))
    answered = preds != -1
    nu = int(np.sum(~answered))

    if answered.sum() == 0:
        return 0.0

    y_ans = y_true[answered]
    p_ans = preds[answered]

    tp = int(np.sum((p_ans == 1) & (y_ans == 1)))
    fp = int(np.sum((p_ans == 1) & (y_ans == 0)))
    fn = int(np.sum((p_ans == 0) & (y_ans == 1)))

    # Unanswered positives count as partial false negatives
    fn_u = fn + nu * (y_true.sum() / n) if n > 0 else fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn_u) if (tp + fn_u) > 0 else 0.0

    beta = 0.5
    beta2 = beta * beta
    if precision + recall == 0:
        return 0.0
    return (1 + beta2) * precision * recall / (beta2 * precision + recall)


def pan_mean(y_true, probs, threshold=0.5):
    """Compute the full PAN mean: average of (roc-auc, brier, c@1, f1, f05u)."""
    auc = roc_auc_score(y_true, probs)
    brier = 1.0 - brier_score_loss(y_true, probs)
    cat1 = compute_cat1(y_true, probs, threshold)
    preds = np.where(probs > threshold, 1, np.where(probs < threshold, 0, -1))
    answered = preds != -1
    if answered.sum() > 0:
        f1 = f1_score(y_true[answered], preds[answered], average="macro")
    else:
        f1 = 0.0
    f05u = compute_f05u(y_true, probs, threshold)
    return {
        "roc_auc": auc, "brier": brier, "cat1": cat1,
        "f1": f1, "f05u": f05u,
        "mean": float(np.mean([auc, brier, cat1, f1, f05u])),
    }


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


def load_deberta(deberta_dir):
    """Load DeBERTa ONNX model if available."""
    deberta_dir = Path(deberta_dir)
    if not deberta_dir.exists():
        return None, None
    import onnxruntime as ort
    from transformers import AutoTokenizer

    onnx_path = deberta_dir / "model_quantized.onnx"
    if not onnx_path.exists():
        candidates = list(deberta_dir.glob("*.onnx"))
        if not candidates:
            return None, None
        onnx_path = candidates[0]
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    tokenizer = AutoTokenizer.from_pretrained(str(deberta_dir))
    return session, tokenizer


def predict_deberta(texts, session, tokenizer, max_length=512, batch_size=16):
    """Run DeBERTa ONNX inference, return P(AI) for each text."""
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encoded = tokenizer(
            batch_texts, truncation=True, max_length=max_length,
            padding=True, return_tensors="np",
        )
        feed = {k: v for k, v in encoded.items()
                if k in {n.name for n in session.get_inputs()}}
        logits = session.run(None, feed)[0]
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        all_probs.append(probs[:, 1])
    return np.concatenate(all_probs)


def compute_gpt2_features(texts, gpt2_dir, max_tokens=512):
    """Compute GPT-2 perplexity features + Binoculars-style score for each text.

    Returns array of shape (N, 4): [log_ppl, burstiness, rank1_acc, binoculars].
    """
    import onnxruntime as ort
    from transformers import AutoTokenizer

    gpt2_path = Path(gpt2_dir) / "model.onnx"
    if not gpt2_path.exists():
        return None
    session = ort.InferenceSession(str(gpt2_path), providers=["CPUExecutionProvider"])
    tokenizer = AutoTokenizer.from_pretrained(str(gpt2_dir))
    results = []
    for text in texts:
        tokens = tokenizer(text, return_tensors="np", truncation=True, max_length=max_tokens)
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        position_ids = np.arange(input_ids.shape[1]).reshape(1, -1).astype(np.int64)
        logits = session.run(None, {
            "input_ids": input_ids, "attention_mask": attention_mask,
            "position_ids": position_ids,
        })[0]
        ids = input_ids[0]
        n_tokens = len(ids) - 1
        if n_tokens < 2:
            results.append([3.0, 1.0, 0.0, 1.0])
            continue

        log_probs = []
        cross_ents = []
        rank1_hits = 0
        for i in range(1, len(ids)):
            logit = logits[0, i - 1]
            shifted = logit - np.max(logit)
            probs = np.exp(shifted) / np.exp(shifted).sum()
            target_prob = probs[ids[i]]
            log_probs.append(np.log(target_prob + 1e-10))
            cross_ents.append(-np.log(probs.max() + 1e-10))
            if np.argmax(logit) == ids[i]:
                rank1_hits += 1

        lp = np.array(log_probs)
        log_ppl = float(-np.mean(lp))
        burstiness = float(np.std(lp))
        rank1_acc = float(rank1_hits / n_tokens)
        # Binoculars-style: ratio of log-perplexity to cross-entropy
        mean_cross_ent = float(np.mean(cross_ents))
        binoculars = log_ppl / (mean_cross_ent + 1e-10)
        results.append([log_ppl, burstiness, rank1_acc, binoculars])
    return np.array(results, dtype=np.float32)


GPT2_FEATURE_NAMES = ["log_ppl", "ppl_burstiness", "rank1_acc", "binoculars"]


def build_meta_features(component_probs):
    """Build enriched meta-features from component probabilities.

    Args:
        component_probs: list of 1-D arrays [lgb, svm] or [lgb, svm, deberta]

    Returns:
        2-D array with raw probs + interaction features (max, min, std, spread)
    """
    raw = np.column_stack(component_probs)
    interactions = np.column_stack([
        np.max(raw, axis=1),
        np.min(raw, axis=1),
        np.std(raw, axis=1),
        np.max(raw, axis=1) - np.min(raw, axis=1),
    ])
    return np.hstack([raw, interactions])


# Feature names for LGB when using stylometric + SVD features
def lgb_feature_names(n_char_svd=50, n_word_svd=30):
    names = list(FEATURE_NAMES)
    names += [f"char_svd_{i}" for i in range(n_char_svd)]
    names += [f"word_svd_{i}" for i in range(n_word_svd)]
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, nargs="+")
    parser.add_argument("--val", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--deberta-dir", type=Path, default=Path("models/deberta_onnx"))
    parser.add_argument("--seeds", type=str, default="42,123,456,789,1024")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]

    log.info("=" * 60)
    log.info("  VOIGHT-KAMPFF: Generative AI Detection Training")
    log.info("=" * 60)

    # Load training data
    records = []
    for data_path in args.data:
        log.info("\nLoading training data from %s...", data_path)
        records.extend(load_data(data_path))
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

    # Generate OOF predictions for SVM (avoid data leakage in weight search)
    svm_oof_probs = np.zeros(len(labels))
    skf_svm = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in skf_svm.split(X_svm_sparse, labels):
        fold_svm = CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=5000), cv=3, method="sigmoid"
        )
        fold_svm.fit(X_svm_sparse[train_idx], labels[train_idx])
        svm_oof_probs[val_idx] = fold_svm.predict_proba(X_svm_sparse[val_idx])[:, 1]
    svm_metrics = evaluate(labels, svm_oof_probs, 0.5)
    log.info("  SVM OOF AUC=%.4f, Brier=%.4f, F1=%.4f, Acc=%.4f",
             svm_metrics["auc"], svm_metrics["brier_complement"],
             svm_metrics["f1_macro"], svm_metrics["accuracy"])

    # Train final SVM on all data
    svm = CalibratedClassifierCV(
        LinearSVC(C=1.0, max_iter=5000), cv=5, method="sigmoid"
    )
    svm.fit(X_svm_sparse, labels)

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
    # Load DeBERTa (optional)
    # ================================================================
    deberta_session, deberta_tokenizer = load_deberta(args.deberta_dir)
    has_deberta = deberta_session is not None
    if has_deberta:
        log.info("\nLoaded DeBERTa ONNX from %s", args.deberta_dir)
    else:
        log.info("\nNo DeBERTa model found at %s, skipping", args.deberta_dir)

    # ================================================================
    # Learned stacker on val predictions (honest for all components)
    # ================================================================
    log.info("\n--- Training Stacker on Val Predictions ---")
    stacker = None
    calibrator = None
    ensemble_weights = [0.5, 0.5]
    meta_names = ["lgb", "svm"]

    if val_texts is not None:
        lgb_val_prob = np.mean(
            [m.predict(X_val_combined) for m in lgb_models], axis=0
        )
        X_svm_val_sparse = build_sparse_tfidf(
            val_texts, svm_char_vec, svm_word_vec, fit=False
        )
        svm_val_prob = svm.predict_proba(X_svm_val_sparse)[:, 1]

        val_meta_cols = [lgb_val_prob, svm_val_prob]
        if has_deberta:
            log.info("  Running DeBERTa inference on val set...")
            deberta_val_prob = predict_deberta(
                val_texts, deberta_session, deberta_tokenizer
            )
            val_meta_cols.append(deberta_val_prob)
            meta_names.append("deberta")

        X_meta_val = build_meta_features(val_meta_cols)
        meta_feat_names = meta_names + ["max", "min", "std", "spread"]

        # Train stacker on val (all component predictions are honest here)
        stacker = LogisticRegression(C=1.0, max_iter=1000)
        stacker.fit(X_meta_val, val_labels)

        blended_val = stacker.predict_proba(X_meta_val)[:, 1]
        val_pre = pan_mean(val_labels, blended_val, 0.5)
        log.info("  Stacker val (pre-cal): AUC=%.4f, Brier=%.4f, c@1=%.4f, F1=%.4f, F05u=%.4f, Mean=%.4f",
                 val_pre["roc_auc"], val_pre["brier"], val_pre["cat1"],
                 val_pre["f1"], val_pre["f05u"], val_pre["mean"])

        # Cross-val stacker predictions for honest calibration targets
        from sklearn.model_selection import cross_val_predict
        stacker_oof_val = cross_val_predict(
            LogisticRegression(C=1.0, max_iter=1000),
            X_meta_val, val_labels, cv=5, method="predict_proba",
        )[:, 1]

        calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        calibrator.fit(stacker_oof_val, val_labels)
        calibrated_val = calibrator.predict(blended_val)
        val_post = pan_mean(val_labels, calibrated_val, 0.5)
        log.info("  Stacker val (post-cal): AUC=%.4f, Brier=%.4f, c@1=%.4f, F1=%.4f, F05u=%.4f, Mean=%.4f",
                 val_post["roc_auc"], val_post["brier"], val_post["cat1"],
                 val_post["f1"], val_post["f05u"], val_post["mean"])

        opt_t, opt_margin, opt_mean = optimize_threshold_and_abstention(
            val_labels, calibrated_val
        )
        log.info("  Optimal threshold=%.3f, abstention_margin=%.3f, PAN mean=%.4f",
                 opt_t, opt_margin, opt_mean)

        coefs = dict(zip(meta_feat_names, stacker.coef_[0]))
        log.info("  Stacker coefficients: %s", coefs)

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

    if stacker is not None:
        joblib.dump(stacker, args.output_dir / "stacker.pkl")

    config = {
        "threshold": float(opt_t) if val_texts is not None else float(cv_thresh),
        "abstention_margin": float(opt_margin) if val_texts is not None else 0.0,
        "seeds": seeds,
        "n_stylo_features": len(FEATURE_NAMES),
        "n_char_svd": 50,
        "n_word_svd": 30,
        "feature_names": feat_names,
        "ensemble_weights": ensemble_weights,
        "has_stacker": stacker is not None,
        "stacker_components": meta_names,
        "has_calibrator": calibrator is not None,
        "cv_metrics_lgb": {k: float(v) for k, v in cv_metrics_lgb.items()},
    }
    if val_texts is not None:
        config["pan_metrics_val"] = {k: float(v) for k, v in val_post.items()}
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

"""TIRA entry point for Voight-Kampff Generative AI Detection.

Ensemble: ONNX DeBERTa + LightGBM + Calibrated SVM.

Usage (TIRA convention):
    python main.py $inputDataset/dataset.jsonl $outputDir
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from scipy.sparse import hstack as sparse_hstack

from features import extract_features_batch

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "models"


def load_all_models():
    with open(MODEL_DIR / "vk_config.json") as f:
        config = json.load(f)

    lgb_models = []
    for seed in config["seeds"]:
        lgb_models.append(lgb.Booster(model_file=str(MODEL_DIR / f"vk_seed{seed}.txt")))

    char_vec = joblib.load(MODEL_DIR / "tfidf_char.pkl")
    word_vec = joblib.load(MODEL_DIR / "tfidf_word.pkl")
    char_svd = joblib.load(MODEL_DIR / "svd_char.pkl")
    word_svd = joblib.load(MODEL_DIR / "svd_word.pkl")

    svm_char_vec = joblib.load(MODEL_DIR / "svm_tfidf_char.pkl")
    svm_word_vec = joblib.load(MODEL_DIR / "svm_tfidf_word.pkl")
    svm = joblib.load(MODEL_DIR / "svm_calibrated.pkl")

    calibrator = None
    if config.get("has_calibrator"):
        calibrator = joblib.load(MODEL_DIR / "calibrator.pkl")

    stacker = None
    if config.get("has_stacker"):
        stacker = joblib.load(MODEL_DIR / "stacker.pkl")

    deberta_session = None
    deberta_tokenizer = None
    deberta_dir = MODEL_DIR / "deberta_onnx"
    if deberta_dir.exists():
        import onnxruntime as ort
        from transformers import AutoTokenizer

        onnx_path = deberta_dir / "model_quantized.onnx"
        if not onnx_path.exists():
            onnx_candidates = list(deberta_dir.glob("*.onnx"))
            if onnx_candidates:
                onnx_path = onnx_candidates[0]
        deberta_session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        deberta_tokenizer = AutoTokenizer.from_pretrained(str(deberta_dir))
        log.info("Loaded DeBERTa ONNX model from %s", onnx_path.name)

    return {
        "lgb_models": lgb_models,
        "char_vec": char_vec,
        "word_vec": word_vec,
        "char_svd": char_svd,
        "word_svd": word_svd,
        "svm_char_vec": svm_char_vec,
        "svm_word_vec": svm_word_vec,
        "svm": svm,
        "calibrator": calibrator,
        "stacker": stacker,
        "deberta_session": deberta_session,
        "deberta_tokenizer": deberta_tokenizer,
        "config": config,
    }


def compute_gpt2_ppl(texts, max_tokens=512):
    """Compute GPT-2 perplexity features for each text."""
    import onnxruntime as ort
    from transformers import AutoTokenizer
    gpt2_path = MODEL_DIR / "gpt2-onnx" / "model.onnx"
    if not gpt2_path.exists():
        return None
    gpt2_session = ort.InferenceSession(str(gpt2_path), providers=["CPUExecutionProvider"])
    gpt2_tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR / "gpt2-onnx"))
    results = []
    for text in texts:
        tokens = gpt2_tokenizer(text, return_tensors="np", truncation=True, max_length=max_tokens)
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        position_ids = np.arange(input_ids.shape[1]).reshape(1, -1).astype(np.int64)
        logits = gpt2_session.run(None, {
            "input_ids": input_ids, "attention_mask": attention_mask,
            "position_ids": position_ids,
        })[0]
        ids = input_ids[0]
        log_probs = []
        cross_ents = []
        rank1_hits = 0
        for i in range(1, len(ids)):
            logit = logits[0, i - 1]
            shifted = logit - np.max(logit)
            probs = np.exp(shifted) / np.exp(shifted).sum()
            log_probs.append(np.log(probs[ids[i]] + 1e-10))
            cross_ents.append(-np.log(probs.max() + 1e-10))
            if np.argmax(logit) == ids[i]:
                rank1_hits += 1
        lp = np.array(log_probs)
        n_tokens = max(len(ids) - 1, 1)
        log_ppl = float(-np.mean(lp))
        mean_cross_ent = float(np.mean(cross_ents))
        results.append({
            "log_ppl": log_ppl,
            "burstiness": float(np.std(lp)),
            "rank1_acc": float(rank1_hits / n_tokens),
            "binoculars": log_ppl / (mean_cross_ent + 1e-10),
        })
    return results


def predict_deberta(texts, session, tokenizer, max_length=512, batch_size=16):
    """Run DeBERTa ONNX inference, return P(AI) for each text."""
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encoded = tokenizer(
            batch_texts, truncation=True, max_length=max_length,
            padding=True, return_tensors="np",
        )
        feed = {k: v for k, v in encoded.items() if k in {n.name for n in session.get_inputs()}}
        logits = session.run(None, feed)[0]
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        all_probs.append(probs[:, 1])
    return np.concatenate(all_probs)


def build_meta_features(component_probs):
    """Build enriched meta-features from component probabilities."""
    raw = np.column_stack(component_probs)
    interactions = np.column_stack([
        np.max(raw, axis=1),
        np.min(raw, axis=1),
        np.std(raw, axis=1),
        np.max(raw, axis=1) - np.min(raw, axis=1),
    ])
    return np.hstack([raw, interactions])


def main():
    if len(sys.argv) < 3:
        log.error("Usage: python main.py <input_jsonl> <output_dir>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading models...")
    m = load_all_models()
    config = m["config"]
    weights = config.get("ensemble_weights", [0.5, 0.5])

    log.info("Reading input from %s", input_path)
    records = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    log.info("  %d records", len(records))

    texts = [r.get("text", "") for r in records]
    ids = [r["id"] for r in records]

    # LGB: stylometric + TF-IDF SVD features
    log.info("Extracting features...")
    X_stylo = extract_features_batch(texts, show_progress=False)

    char_tfidf = m["char_vec"].transform(texts)
    word_tfidf = m["word_vec"].transform(texts)
    X_char_svd = m["char_svd"].transform(char_tfidf).astype(np.float32)
    X_word_svd = m["word_svd"].transform(word_tfidf).astype(np.float32)
    X_combined = np.hstack([X_stylo, X_char_svd, X_word_svd])

    lgb_prob = np.mean(
        [mdl.predict(X_combined) for mdl in m["lgb_models"]], axis=0
    )

    # SVM: raw sparse TF-IDF
    svm_char = m["svm_char_vec"].transform(texts)
    svm_word = m["svm_word_vec"].transform(texts)
    X_svm = sparse_hstack([svm_char, svm_word])
    svm_prob = m["svm"].predict_proba(X_svm)[:, 1]

    # DeBERTa inference (if available)
    deberta_prob = None
    if m["deberta_session"] is not None:
        log.info("Running DeBERTa inference...")
        deberta_prob = predict_deberta(
            texts, m["deberta_session"], m["deberta_tokenizer"]
        )

    # Ensemble: stacker (preferred) or fallback to weighted average
    if m["stacker"] is not None:
        meta_cols = [lgb_prob, svm_prob]
        if deberta_prob is not None and "deberta" in config.get("stacker_components", []):
            meta_cols.append(deberta_prob)
        X_meta = build_meta_features(meta_cols)
        blended = m["stacker"].predict_proba(X_meta)[:, 1]
        log.info("Using learned stacker (%s)", config.get("stacker_components"))
    else:
        blended = weights[0] * lgb_prob + weights[1] * svm_prob
        if deberta_prob is not None:
            w_deb = config.get("deberta_weight", 0.0)
            if w_deb > 0:
                blended = blended + w_deb * deberta_prob

    # Calibrate
    if m["calibrator"] is not None:
        blended = m["calibrator"].predict(blended)

    # GPT-2 perplexity safety net: catches unseen generators the ensemble misses
    ppl_features = compute_gpt2_ppl(texts)
    if ppl_features is not None:
        log_ppl = np.array([f["log_ppl"] for f in ppl_features])
        bino = np.array([f["binoculars"] for f in ppl_features])
        r1 = np.array([f["rank1_acc"] for f in ppl_features])

        # Convert to AI probability via sigmoid (lower ppl/bino = more AI-like)
        ppl_ai = 1.0 / (1.0 + np.exp(3.5 * (log_ppl - 3.1)))
        bino_ai = 1.0 / (1.0 + np.exp(8.0 * (bino - 2.40)))
        r1_ai = 1.0 / (1.0 + np.exp(-15.0 * (r1 - 0.38)))
        ppl_signal = 0.4 * ppl_ai + 0.35 * bino_ai + 0.25 * r1_ai

        # Only boost toward AI when ensemble is uncertain or says human
        # This catches unseen generators without hurting confident correct predictions
        boost = np.maximum(0, ppl_signal - blended) * 0.5
        n_boosted = int((boost > 0.01).sum())
        blended = np.clip(blended + boost, 0.0, 1.0)
        log.info("PPL safety net: boosted %d/%d predictions (mean_ppl=%.2f, mean_bino=%.2f)",
                 n_boosted, len(blended), log_ppl.mean(), bino.mean())

    # Post-processing: abstain on uncertain predictions
    abstention_margin = config.get("abstention_margin", 0.0)
    if abstention_margin > 0:
        uncertain = np.abs(blended - 0.5) < abstention_margin
        n_abstained = int(uncertain.sum())
        if n_abstained > 0:
            log.info("Abstaining on %d/%d predictions (margin=%.3f)",
                     n_abstained, len(blended), abstention_margin)
            blended[uncertain] = 0.5

    # Write predictions
    log.info("Writing predictions...")
    output_path = output_dir / "predictions.jsonl"
    with open(output_path, "w", encoding="utf-8") as out:
        for doc_id, prob in zip(ids, blended):
            prediction = {"id": doc_id, "label": round(float(prob), 4)}
            out.write(json.dumps(prediction) + "\n")

    log.info("Wrote %d predictions to %s", len(records), output_path)


if __name__ == "__main__":
    main()

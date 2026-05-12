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
        "deberta_session": deberta_session,
        "deberta_tokenizer": deberta_tokenizer,
        "config": config,
    }


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

    # Blend LGB + SVM
    blended = weights[0] * lgb_prob + weights[1] * svm_prob

    # DeBERTa (if available)
    if m["deberta_session"] is not None:
        log.info("Running DeBERTa inference...")
        deberta_prob = predict_deberta(
            texts, m["deberta_session"], m["deberta_tokenizer"]
        )
        w_deb = config.get("deberta_weight", 0.6)
        blended = w_deb * deberta_prob + (1 - w_deb) * blended

    # Calibrate
    if m["calibrator"] is not None:
        blended = m["calibrator"].predict(blended)

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

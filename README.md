<!-- repo-header:start -->
<img src="https://github.com/dcondrey.png?size=160" alt="Voight-Kampff: Cross-Genre AI-Generated Text Detection (PAN@CLEF 2026) logo" width="120" align="left">

<h1>Voight-Kampff: Cross-Genre AI-Generated Text Detection (PAN@CLEF 2026)</h1>

<p><strong>PAN@CLEF 2026 Voight-Kampff AI-generated text detector: calibrated DeBERTa + LightGBM + SVM ensemble over 44 domain-portable features (0.891 ROC-AUC).</strong></p>

<br clear="left">

[![Best Practices Evidence](https://img.shields.io/badge/best%20practices-evidence%20reviewed-6a4c93?style=flat-square&labelColor=20232a)](.bestpractices.json) [![License](https://img.shields.io/github/license/dcondrey/voight-kampff-clef2026?style=flat-square&labelColor=20232a&color=007ec6&label=license)](https://github.com/dcondrey/voight-kampff-clef2026/blob/main/LICENSE) [![GitHub Sponsors](https://img.shields.io/badge/GitHub%20Sponsors-Sponsor-EA4AAA?style=flat-square&labelColor=20232a)](https://github.com/sponsors/dcondrey)
<!-- repo-header:end -->

## Official Result

On the PAN 2026 test set, our best configuration (`large-rank`) scored:

| Metric | Score |
|---|---|
| ROC-AUC | **0.891** |
| C@1 | 0.853 |
| F1 | 0.902 |
| F0.5u | 0.899 |
| **Mean (ranking metric)** | **0.887 — rank 7** |

On the PAN 2025 backward-compatibility test set the same configuration scored **0.979**. We submitted 11 software configurations to TIRA; they share the same trained base models and differ only in ensemble composition (which classifiers are included, stacker vs. weighted-average fusion, isotonic calibration on/off, LightGBM seed-bag size, and the abstention margin).

## Approach

A calibrated ensemble of three complementary classifiers combined by learned stacking:

- **DeBERTa-v2** — fine-tuned, exported to ONNX with INT8 dynamic quantization for CPU inference.
- **Multi-seed LightGBM** — over 44 domain-portable stylometric features, concatenated with truncated-SVD projections of character (3–6) and word (1–2) n-gram TF-IDF, plus GPT-2 perplexity features.
- **Calibrated linear SVM** — Platt-scaled, over raw sparse character/word n-gram TF-IDF.

Component probabilities are combined via a logistic-regression stacker with interaction features (max, min, std, range), followed by isotonic-regression calibration; borderline predictions within a narrow margin of 0.5 abstain.

The 44 features are deliberately **domain-portable**. That choice came from our companion Reasoning Trajectory Detection analysis, where generator- and topic-specific features died under domain shift (0% fire rate out-of-domain) while vocabulary fingerprints and compression measures survived — so every feature here was selected for support overlap across genres, not for training-set effect size.

### Feature groups (44)

| Group | Count | Examples |
|---|---|---|
| Document structure | 7 | word/char/sentence/paragraph counts |
| Vocabulary richness | 8 | type-token ratio, hapax ratio, Yule's K, Heaps' exponent, MATTR |
| Sentence structure | 6 | mean length, length CV, sentence-start diversity |
| Readability | 4 | Flesch-Kincaid, Coleman-Liau, avg. syllables |
| Compression/entropy | 4 | zlib ratio, char entropy, repetition ratio |
| Style markers | 9 | punctuation ratio, quote density, contraction ratio |
| Discourse | 4 | transition diversity, connective formality |
| Distributional | 2 | burstiness, intrinsic dimensionality |
| Perplexity (GPT-2) | 4 | log-perplexity, burstiness, rank-1 accuracy, binoculars ratio |

## Quick Start

```bash
pip install -e .
python download_onnx.py            # fetch/convert the quantized DeBERTa ONNX model
python train.py                    # train LightGBM + SVM, fit the stacker + calibrator
python calibrate_ensemble.py       # isotonic calibration of the fused score
python main.py -i input/ -o output/  # predict (writes probabilities)
```

Docker (TIRA):

```bash
docker build -t voight-kampff-clef2026 .
docker run --rm -v /input:/input -v /output:/output voight-kampff-clef2026 -i /input -o /output
```

## Repository Structure

```
features.py             # the 44 domain-portable features + GPT-2 perplexity
train.py                # LightGBM (multi-seed) + SVM training
train_transformer.py    # DeBERTa-v2 fine-tuning
calibrate_ensemble.py   # learned stacking + isotonic calibration
download_onnx.py        # DeBERTa -> ONNX INT8 export
augment.py              # training-data augmentation
main.py                 # inference entrypoint
models/                 # trained base models, stacker, calibrator, vk_config.json
Dockerfile              # CPU/ONNX containerized inference
```

## Citation

```bibtex
@inproceedings{condrey2026pan,
  title     = {Writerslogic at {PAN} 2026: Process over Content for Robust
               Detection under Domain Shift},
  author    = {Condrey, David},
  booktitle = {Working Notes of CLEF 2026 -- Conference and Labs of the Evaluation Forum},
  series    = {CEUR Workshop Proceedings},
  year      = {2026},
  publisher = {CEUR-WS.org},
  note      = {Voight-Kampff is one of three PAN tasks in this paper; to appear}
}
```

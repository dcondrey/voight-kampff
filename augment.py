"""Homoglyph augmentation for AI-generated text detection training.

Applies Unicode confusable character substitutions and zero-width character
injection to a fraction of training texts, following the mdok approach.

Usage:
    uv run python augment.py --input data/train.jsonl --output data/train_augmented.jsonl
"""

import argparse
import json
import logging
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# Latin confusable mappings: ASCII -> visually similar Unicode
CONFUSABLES = {
    'a': ['\u0430', '\u00e0', '\u00e1', '\u1ea1'],  # Cyrillic а, à, á, ạ
    'c': ['\u0441', '\u00e7', '\u0188'],              # Cyrillic с, ç, ƈ
    'd': ['\u0501', '\u0257'],                         # Cyrillic ԁ, ɗ
    'e': ['\u0435', '\u00e8', '\u00e9', '\u0117'],    # Cyrillic е, è, é, ė
    'g': ['\u0261', '\u01e7'],                         # ɡ, ǧ
    'h': ['\u04bb', '\u0570'],                         # Cyrillic һ, Armenian հ
    'i': ['\u0456', '\u00ec', '\u00ed', '\u0131'],    # Cyrillic і, ì, í, ı
    'j': ['\u0458', '\u029d'],                         # Cyrillic ј, ʝ
    'k': ['\u043a'],                                   # Cyrillic к
    'l': ['\u006c', '\u0131', '\u04cf'],              # l, ı, Cyrillic ӏ
    'm': ['\u043c'],                                   # Cyrillic м
    'n': ['\u0578'],                                   # Armenian ո
    'o': ['\u043e', '\u00f2', '\u00f3', '\u0585'],    # Cyrillic о, ò, ó, Armenian օ
    'p': ['\u0440', '\u0271'],                         # Cyrillic р, ɱ
    'q': ['\u051b'],                                   # Cyrillic ԛ
    'r': ['\u0433'],                                   # Cyrillic г
    's': ['\u0455', '\u015f'],                         # Cyrillic ѕ, ş
    't': ['\u0442'],                                   # Cyrillic т
    'u': ['\u0446'],                                   # Cyrillic ц
    'v': ['\u0475', '\u03bd'],                         # Cyrillic ѵ, Greek ν
    'w': ['\u0461'],                                   # Cyrillic ѡ
    'x': ['\u0445', '\u04b3'],                         # Cyrillic х, ҳ
    'y': ['\u0443', '\u00fd'],                         # Cyrillic у, ý
    'z': ['\u0290'],                                   # ʐ
    'A': ['\u0410', '\u00c0', '\u00c1'],              # Cyrillic А, À, Á
    'B': ['\u0412', '\u0392'],                         # Cyrillic В, Greek Β
    'C': ['\u0421', '\u00c7'],                         # Cyrillic С, Ç
    'E': ['\u0415', '\u00c8', '\u00c9'],              # Cyrillic Е, È, É
    'H': ['\u041d', '\u0397'],                         # Cyrillic Н, Greek Η
    'I': ['\u0406', '\u00cc', '\u00cd'],              # Cyrillic І, Ì, Í
    'K': ['\u041a'],                                   # Cyrillic К
    'M': ['\u041c'],                                   # Cyrillic М
    'N': ['\u039d'],                                   # Greek Ν
    'O': ['\u041e', '\u00d2', '\u00d3'],              # Cyrillic О, Ò, Ó
    'P': ['\u0420', '\u03a1'],                         # Cyrillic Р, Greek Ρ
    'S': ['\u0405'],                                   # Cyrillic Ѕ
    'T': ['\u0422', '\u03a4'],                         # Cyrillic Т, Greek Τ
    'X': ['\u0425', '\u03a7'],                         # Cyrillic Х, Greek Χ
    'Y': ['\u0423'],                                   # Cyrillic У
    'Z': ['\u0396'],                                   # Greek Ζ
}

ZERO_WIDTH_CHARS = [
    '\u200b',  # zero-width space
    '\u200c',  # zero-width non-joiner
    '\u200d',  # zero-width joiner
    '\ufeff',  # zero-width no-break space
]


def apply_homoglyph(text, char_replace_prob=0.05, zwc_insert_prob=0.05, rng=None):
    """Apply homoglyph attack to text."""
    if rng is None:
        rng = random.Random()

    chars = list(text)
    result = []
    for c in chars:
        if c in CONFUSABLES and rng.random() < char_replace_prob:
            result.append(rng.choice(CONFUSABLES[c]))
        else:
            result.append(c)
        if rng.random() < zwc_insert_prob:
            result.append(rng.choice(ZERO_WIDTH_CHARS))

    return ''.join(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.10,
                        help="Fraction of AI texts to augment")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    records = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    log.info("Loaded %d records", len(records))
    ai_indices = [i for i, r in enumerate(records) if r["label"] == 1]
    n_augment = int(len(ai_indices) * args.fraction)
    augment_indices = set(rng.sample(ai_indices, n_augment))

    log.info("Augmenting %d of %d AI texts (%.0f%%)",
             n_augment, len(ai_indices), 100 * args.fraction)

    augmented = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for i, r in enumerate(records):
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
            if i in augment_indices:
                aug = dict(r)
                aug["id"] = r["id"] + "_homoglyph"
                aug["text"] = apply_homoglyph(r["text"], rng=rng)
                out.write(json.dumps(aug, ensure_ascii=False) + "\n")
                augmented += 1

    total = len(records) + augmented
    log.info("Wrote %d records (%d original + %d augmented) to %s",
             total, len(records), augmented, args.output)


if __name__ == "__main__":
    main()

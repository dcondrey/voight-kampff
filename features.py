"""Feature engineering for Voight-Kampff Generative AI Detection.

Domain-portable features for detecting AI-generated text across
essays, news, and fiction genres. No domain-specific features
(no LaTeX, no boxed answers) — purely structural and vocabulary
fingerprint signals.
"""

import math
import re
import string
import zlib
from collections import Counter

import numpy as np
from tqdm import tqdm


COORD_CONJUNCTIONS = frozenset({"and", "but", "so", "or", "yet", "for", "nor"})

FUNCTION_WORDS = frozenset(
    "the a an is are was were be been being have has had do does did will would "
    "shall should can could may might must to of in for on at by with from as "
    "into through during before after above below between under about this that "
    "these those and but or if while because so yet nor not very also then than "
    "i me my we our he him his she her it its they them their you your".split()
)

FORMAL_CONNECTIVES = frozenset({
    "thus", "hence", "consequently", "furthermore",
    "moreover", "nevertheless", "therefore", "whereby",
    "accordingly", "henceforth", "notwithstanding",
})

INFORMAL_CONNECTIVES = frozenset({
    "so", "but", "also", "anyway", "besides",
    "still", "yet", "though", "plus", "ok",
})

TRANSITIONS = frozenset({
    "therefore", "thus", "hence", "so", "consequently",
    "however", "but", "yet", "nevertheless", "although",
    "moreover", "furthermore", "additionally", "also",
    "since", "because", "given", "note", "recall",
    "first", "second", "next", "then", "finally",
    "similarly", "likewise", "instead", "otherwise",
    "specifically", "namely", "indeed", "clearly",
    "meanwhile", "regardless", "nonetheless",
})


def _compression_ratio(text):
    if not text:
        return 1.0
    raw = text.encode("utf-8")
    compressed = zlib.compress(raw, level=9)
    return len(compressed) / max(len(raw), 1)


def _entropy(text):
    if not text:
        return 0.0
    freq = Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in freq.values())


def _sentence_split(text):
    sentences = re.split(r'[.!?]+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def extract_features(text):
    """Extract feature vector from a single text.

    Returns a list of 37 floats.
    """
    tokens = text.split()
    tokens_lower = text.lower().split()
    word_count = len(tokens)
    char_count = len(text)
    sentences = _sentence_split(text)
    sent_count = max(len(sentences), 1)
    unique_words = set(tokens_lower)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Basic size
    word_count_log = math.log1p(word_count)
    char_count_log = math.log1p(char_count)
    sent_count_log = math.log1p(sent_count)

    # Vocabulary
    vocab_richness = len(unique_words) / max(word_count, 1)
    avg_word_length = sum(len(t) for t in tokens) / max(word_count, 1)

    # Character ratios
    digit_ratio = sum(c.isdigit() for c in text) / max(char_count, 1)
    punct_ratio = sum(c in string.punctuation for c in text) / max(char_count, 1)

    # Compression and entropy
    compression_ratio = _compression_ratio(text)
    char_entropy = _entropy(text)

    # Repetition
    if word_count >= 4:
        ngrams = [tuple(tokens_lower[i:i + 4]) for i in range(word_count - 3)]
        counts = Counter(ngrams)
        repeated = sum(c - 1 for c in counts.values() if c > 1)
        repetition_ratio = repeated / len(ngrams)
    else:
        repetition_ratio = 0.0

    # Sentence structure
    avg_sentence_length = word_count / sent_count
    sent_lengths = [len(s.split()) for s in sentences]
    if len(sent_lengths) >= 2:
        mean_sl = sum(sent_lengths) / len(sent_lengths)
        if mean_sl > 0:
            var_sl = sum((x - mean_sl) ** 2 for x in sent_lengths) / len(sent_lengths)
            sentence_length_cv = math.sqrt(var_sl) / mean_sl
        else:
            sentence_length_cv = 0.0
    else:
        sentence_length_cv = 0.0

    # Paragraph structure
    paragraph_count = len(paragraphs)
    words_per_paragraph = sum(len(p.split()) for p in paragraphs) / max(paragraph_count, 1)
    newline_ratio = text.count("\n") / max(char_count, 1)

    # Vocabulary fingerprints
    freq = Counter(tokens_lower)
    hapax = sum(1 for c in freq.values() if c == 1)
    hapax_ratio = hapax / max(len(freq), 1)

    # Yule's K
    n = word_count
    if n >= 2:
        freq_spectrum = Counter(freq.values())
        m2 = sum(i * i * v for i, v in freq_spectrum.items())
        yules_k = 10000.0 * (m2 - n) / (n * n) if n > 0 else 0.0
    else:
        yules_k = 0.0

    # Heaps' exponent
    if n >= 10:
        heaps_exponent = math.log(len(unique_words)) / math.log(n)
    else:
        heaps_exponent = 0.0

    # Function word ratio
    function_word_ratio = sum(1 for t in tokens_lower if t in FUNCTION_WORDS) / max(word_count, 1)

    # Sentence compression CV
    long_sentences = [s for s in sentences if len(s.strip()) > 20]
    if len(long_sentences) >= 3:
        ratios = [_compression_ratio(s) for s in long_sentences]
        mean_r = sum(ratios) / len(ratios)
        if mean_r > 0.01:
            var_r = sum((r - mean_r) ** 2 for r in ratios) / len(ratios)
            sentence_compression_cv = math.sqrt(var_r) / mean_r
        else:
            sentence_compression_cv = 0.0
    else:
        sentence_compression_cv = 0.0

    # Transition word features
    words_set = set(tokens_lower)
    transition_diversity = len(words_set & TRANSITIONS)
    formal_count = sum(1 for w in tokens_lower if w in FORMAL_CONNECTIVES)
    informal_count = sum(1 for w in tokens_lower if w in INFORMAL_CONNECTIVES)
    conn_total = formal_count + informal_count
    connective_formality = formal_count / conn_total if conn_total > 0 else 0.5

    # Pronoun features
    first_person = sum(1 for w in tokens_lower if w in {"i", "me", "my", "mine", "myself"})
    first_person_ratio = first_person / max(word_count, 1)
    we_ratio = sum(1 for w in tokens_lower if w in {"we", "us", "our", "ours"}) / max(word_count, 1)

    # Whitespace pattern
    single_nl = text.count("\n")
    double_nl = text.count("\n\n")
    whitespace_pattern = double_nl / single_nl if single_nl > 0 else 0.0

    # Quote and dialogue markers (genre-relevant for fiction)
    quote_count = text.count('"') + text.count('\u201c') + text.count('\u201d')
    quote_density = quote_count / max(char_count, 1)

    # Exclamation/question ratio (style signal)
    excl_count = text.count("!")
    quest_count = text.count("?")
    expressive_punct_ratio = (excl_count + quest_count) / max(char_count, 1)

    # Max word length
    max_word_length = max((len(t) for t in tokens), default=0)

    # Parenthesis density
    paren_density = (text.count("(") + text.count(")")) / max(char_count, 1)

    # Zipf coefficient
    if len(freq) >= 10:
        sorted_freqs = sorted(freq.values(), reverse=True)[:100]
        log_ranks = np.log(np.arange(1, len(sorted_freqs) + 1))
        log_freqs = np.log(np.array(sorted_freqs, dtype=np.float64))
        zipf_coeff = float(np.polyfit(log_ranks, log_freqs, 1)[0])
    else:
        zipf_coeff = -1.0

    # Burstiness
    burst_vals = []
    for word, cnt in freq.items():
        if 3 <= cnt <= 20:
            positions = [i for i, w in enumerate(tokens_lower) if w == word]
            gaps = [positions[j + 1] - positions[j] for j in range(len(positions) - 1)]
            if len(gaps) >= 2:
                mean_g = sum(gaps) / len(gaps)
                if mean_g > 0:
                    var_g = sum((g - mean_g) ** 2 for g in gaps) / len(gaps)
                    burst_vals.append(math.sqrt(var_g) / mean_g)
    burstiness = sum(burst_vals) / len(burst_vals) if burst_vals else 0.0

    # Sentence-start diversity
    if sent_count >= 2:
        first_words = [s.split()[0].lower() for s in sentences if s.split()]
        sent_start_diversity = len(set(first_words)) / len(first_words)
    else:
        sent_start_diversity = 0.0

    # Conjunction-start ratio
    if sent_count >= 2:
        conj_starts = sum(1 for s in sentences if s.split() and s.split()[0].lower() in COORD_CONJUNCTIONS)
        conjunction_start_ratio = conj_starts / sent_count
    else:
        conjunction_start_ratio = 0.0

    # Punctuation spacing CV
    punct_positions = [i for i, c in enumerate(text) if c in ",.;:"]
    if len(punct_positions) >= 3:
        punct_gaps = [punct_positions[j + 1] - punct_positions[j] for j in range(len(punct_positions) - 1)]
        mean_pg = sum(punct_gaps) / len(punct_gaps)
        if mean_pg > 0:
            var_pg = sum((g - mean_pg) ** 2 for g in punct_gaps) / len(punct_gaps)
            punct_spacing_cv = math.sqrt(var_pg) / mean_pg
        else:
            punct_spacing_cv = 0.0
    else:
        punct_spacing_cv = 0.0

    # Comma-to-period ratio
    period_count = text.count(".")
    comma_count = text.count(",")
    comma_period_ratio = comma_count / max(period_count, 1)

    # Average word frequency rank
    if word_count >= 5:
        rank_map = {}
        for rank, (w, _) in enumerate(freq.most_common(), 1):
            rank_map[w] = rank
        avg_word_rank = sum(rank_map[w] for w in tokens_lower) / word_count
    else:
        avg_word_rank = 0.0

    return [
        word_count_log,             # 0
        char_count_log,             # 1
        sent_count_log,             # 2
        vocab_richness,             # 3
        avg_word_length,            # 4
        digit_ratio,                # 5
        punct_ratio,                # 6
        compression_ratio,          # 7
        char_entropy,               # 8
        repetition_ratio,           # 9
        avg_sentence_length,        # 10
        sentence_length_cv,         # 11
        paragraph_count,            # 12
        words_per_paragraph,        # 13
        newline_ratio,              # 14
        hapax_ratio,                # 15
        yules_k,                    # 16
        heaps_exponent,             # 17
        function_word_ratio,        # 18
        sentence_compression_cv,    # 19
        transition_diversity,       # 20
        connective_formality,       # 21
        first_person_ratio,         # 22
        we_ratio,                   # 23
        whitespace_pattern,         # 24
        quote_density,              # 25
        expressive_punct_ratio,     # 26
        max_word_length,            # 27
        paren_density,              # 28
        max(word_count, 1),         # 29: raw word count (for length-aware splits)
        zipf_coeff,                 # 30
        burstiness,                 # 31
        sent_start_diversity,       # 32
        conjunction_start_ratio,    # 33
        punct_spacing_cv,           # 34
        comma_period_ratio,         # 35
        avg_word_rank,              # 36
    ]


FEATURE_NAMES = [
    "word_count_log", "char_count_log", "sent_count_log",
    "vocab_richness", "avg_word_length", "digit_ratio", "punct_ratio",
    "compression_ratio", "char_entropy", "repetition_ratio",
    "avg_sentence_length", "sentence_length_cv",
    "paragraph_count", "words_per_paragraph", "newline_ratio",
    "hapax_ratio", "yules_k", "heaps_exponent",
    "function_word_ratio", "sentence_compression_cv",
    "transition_diversity", "connective_formality",
    "first_person_ratio", "we_ratio", "whitespace_pattern",
    "quote_density", "expressive_punct_ratio",
    "max_word_length", "paren_density", "raw_word_count",
    "zipf_coeff", "burstiness", "sent_start_diversity",
    "conjunction_start_ratio", "punct_spacing_cv",
    "comma_period_ratio", "avg_word_rank",
]


def extract_features_batch(texts, show_progress=True):
    """Extract features for a list of texts.

    Returns (N, 37) numpy array.
    """
    iterator = tqdm(texts, desc="Extracting features") if show_progress else texts
    features = [extract_features(t) for t in iterator]
    return np.array(features, dtype=np.float32)

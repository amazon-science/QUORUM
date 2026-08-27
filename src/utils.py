from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
import matplotlib.pyplot as plt
import numpy as np
import json
import re
from scipy.stats import norm
import gdown
import subprocess

def download_data():
    gdown.download(id="1LBdZOkqyro2IH_nvt8P1LyfKUvRIeUia")
    subprocess.run(["unzip", "summaries.zip"], check=True)


def seed_everything(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across multiple libraries.

    Args:
        seed: Integer seed for random number generation
    """
    import random
    import numpy
    import torch

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def count_syllables(word):
    word = word.lower()
    word = re.sub(r"[^a-z]", "", word)
    vowels = "aeiouy"
    count = 0
    prev_vowel = False

    for c in word:
        if c in vowels and not prev_vowel:
            count += 1
        prev_vowel = c in vowels

    if word.endswith("e"):
        count = max(1, count - 1)

    return max(1, count)


def extract_linguistic_features(text):
    sents = sent_tokenize(text)
    words = [w for w in word_tokenize(text.lower()) if w.isalpha()]
    stop = set(stopwords.words("english"))
    words_ns = [w for w in words if w not in stop]

    if len(words_ns) == 0:
        return None

    avg_word_len = np.mean([len(w) for w in words_ns])
    ttr = len(set(words_ns)) / len(words_ns)
    avg_sent_len = np.mean([len(word_tokenize(s)) for s in sents])

    pos_tags = [p for _, p in pos_tag(words_ns)]
    noun_ratio = pos_tags.count("NN") / len(pos_tags)
    verb_ratio = pos_tags.count("VB") / len(pos_tags)

    syll = np.mean([count_syllables(w) for w in words_ns])
    flesch = 206.835 - 1.015 * avg_sent_len - 84.6 * syll
    flesch_norm = np.clip(1 - flesch / 100, 0, 1)

    return np.array([
        avg_word_len,
        ttr,
        avg_sent_len,
        flesch_norm,
        noun_ratio,
        verb_ratio
    ])


if __name__ == "__main__":
    difficult = "The implementation of adaptive routing enhances the efficiency of annotation workflows under constrained settings."
    easy = "Using smart routing methods greatly improves annotation work when resources are limited."
    print(f"Difficult score: {sum(extract_linguistic_features(difficult))}")
    print(f"Easy score: {sum(extract_linguistic_features(easy))}")
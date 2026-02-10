import re
import string
from typing import List, Dict, Any

def tokenize_sentences(sentences, config):
    default_config = {
        "lowercase": True,
        "remove_punctuation": True,
        "remove_stopwords": False,
        "strip_whitespace": True,
        "language": "en"
    }

    if config:
        default_config.update(config)
    cfg = default_config
    stop_words = {
        "en": {"a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with", "is", "are", "was", "were"}
    }.get(cfg["language"], set())

    processed_results = []

    for text in sentences:
        if cfg["strip_whitespace"]:
            text = text.strip()
        if cfg["lowercase"]:
            text = text.lower()
        if cfg["remove_punctuation"]:
            text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = text.split()
        if cfg["remove_stopwords"]:
            tokens = [word for word in tokens if word not in stop_words]
        processed_results.append(tokens)

    return processed_results


if __name__ == "__main__":
    sample_sentences = [
        "FinBERT is a powerful model for financial sentiment analysis!",
        "  Stock prices surged after the earnings report.  "
    ]
    user_config = {
        "lowercase": True,
        "remove_punctuation": True,
        "remove_stopwords": True
    }
    tokenized_output = tokenize_sentences(sample_sentences, user_config)
    print(f"{'PREPROCESSING RESULTS':^50}")
    print("=" * 50)
    for sent, tok in zip(sample_sentences, tokenized_output):
        print(f"Original: {sent.strip()}")
        print(f"Tokens:   {tok}")
        print("-" * 50)
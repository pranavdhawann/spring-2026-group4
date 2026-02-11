import re
import string
from typing import List, Dict, Any
from transformers import AutoTokenizer


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
    processed_sentences = []
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
        processed_sentences.append(" ".join(tokens))

    return processed_sentences

if __name__ == "__main__":
    sample_sentences = [
        "This is an example sentance to see Tokenization process for FinBERT "
    ]
    cleaned_texts = tokenize_sentences(sample_sentences, {"remove_stopwords": True})
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    inputs = tokenizer(
        cleaned_texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    print("="*50)
    print(f"Cleaned Text 1: {cleaned_texts[0]}")
    print(f"Token IDs:     {inputs['input_ids'][0][:10]}")
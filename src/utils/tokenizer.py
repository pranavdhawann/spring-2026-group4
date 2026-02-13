import string

from transformers import AutoTokenizer


def clean_sentences(sentences, cfg):
    stop_words = {
        "en": {
            "a",
            "an",
            "the",
            "and",
            "or",
            "in",
            "on",
            "at",
            "to",
            "for",
            "with",
            "is",
            "are",
            "was",
            "were",
        }
    }.get(cfg["language"], set())
    processed_sentences = []
    for text in sentences:
        if cfg["strip_whitespace"]:
            text = text.strip()
        if cfg["lowercase"]:
            text = text.lower()
        if cfg["remove_punctuation"]:
            text = text.translate(str.maketrans("", "", string.punctuation))
        tokens = text.split()
        if cfg["remove_stopwords"]:
            tokens = [word for word in tokens if word not in stop_words]
        processed_sentences.append(" ".join(tokens))

    return processed_sentences


def tokenize_sentences(sentences, config):
    default_config = {
        "lowercase": True,
        "remove_punctuation": True,
        "remove_stopwords": False,
        "strip_whitespace": True,
        "language": "en",
        "tokenizer_path": "ProsusAI/finbert",
        "padding": "max_length",
        "truncation": True,
        "max_length": 512,
    }
    if config:
        default_config.update(config)
    cfg = default_config
    cleaned_texts = clean_sentences(sample_sentences, cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_path"])
    inputs = tokenizer(
        cleaned_texts,
        padding=cfg["padding"],
        truncation=cfg["truncation"],
        max_length=cfg["max_length"],
        return_tensors="pt",
    )
    return cleaned_texts, inputs


if __name__ == "__main__":
    sample_sentences = [
        "This is an example sentance to see Tokenization process for FinBERT ",
        "Hello world! wr",
        "sdfsdcsdf",
        "unless they all have the same length.\
            So the tokenizer:\
            Pads shorter sequences\
            Truncates longer sequences\
            Ensures everything has equal length\
            🔹 What Is Happening In Your Case\
            You likely did something like:\
            tokenizer(\
                text,\
                padding='max_length',\
                truncation=True,\
                max_length=12,\
                return_tensors='pt'\
            )\
            So every output becomes length 12.\
            Example:\
            Short sentence\
            hello world",
    ]

    cleaned_texts, inputs = tokenize_sentences(
        sample_sentences, {"remove_stopwords": True, "max_length": 6}
    )
    print("=" * 50)
    for idx in range(len(cleaned_texts)):
        print(f"Cleaned Text {idx+1}: {cleaned_texts[idx]}")
        print(f"Token IDs:     {inputs['input_ids'][idx]}")

import time

import numpy as np

from src.utils.tokenizer import tokenize_sentences


def _replace_none_with_avg_np(arr):
    arr = np.array(arr, dtype=np.float32)
    mask = np.isnan(arr)
    avg = np.mean(arr[~mask]) if np.any(~mask) else 0.0
    arr[mask] = avg
    return arr


def _standardize_list_np(arr, mean=None, std=None):
    arr = np.array(arr, dtype=np.float32)
    if mean is None:
        mean = arr.mean()
        std = arr.std()
        std = std if std != 0 else 1.0
    arr = (arr - mean) / std
    return mean, std, arr


def preprocessFinbertMMBaseline(articles, dates, tokenizer, config, verbose=False):
    """
    Fast preprocessing for FinBERT Baseline
    """
    st_ = time.time()
    input_size = len(dates)

    news_texts = []
    for i in range(input_size):
        day_article = articles[i][0] if articles[i] else ""
        news_texts.append(day_article)

    _, inputs = tokenize_sentences(news_texts, tokenizer, config=config, verbose=False)

    if verbose:
        print(" Time to pre process : ", time.time() - st_)

    return [inputs["input_ids"], inputs["attention_mask"]]

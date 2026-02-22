import time

from src.utils.tokenizer import tokenize_sentences


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
    print(">>>", len(inputs["input_ids"]))
    print("+++", len(inputs["input_ids"][0]))

    if verbose:
        print(" Time to pre process : ", time.time() - st_)

    return [inputs["input_ids"], inputs["attention_mask"]]

import math

from src.utils.tokenizer import tokenize_sentences


class FinBertCollator:
    def __init__(self, collator_cfg):
        self.config = {}
        self.config.update(collator_cfg)

    def __call__(self, batch):
        return preProcessFinBertBaseline(batch, self.config)


def _replace_none_with_avg(lst):
    valid_values = [x for x in lst if x is not None]
    if len(valid_values) == 0:
        return [0] * len(lst)

    avg = sum(valid_values) / len(valid_values)

    return [avg if x is None else x for x in lst]


def _standardize_list(lst, mean=None, std=None):
    if len(lst) == 0:
        raise ValueError("List cannot be empty")
    if mean is None:
        mean = sum(lst) / len(lst)

        variance = sum((x - mean) ** 2 for x in lst) / len(lst)
        std = math.sqrt(variance)

        # 0 division
        if std == 0:
            standardized = [0 for _ in lst]
        else:
            standardized = [(x - mean) / std for x in lst]

        return mean, std, standardized
    else:
        lst = [(x - mean) / std for x in lst]
        return mean, std, lst


def preProcessFinBertBaseline(batch, prProcessCfg):
    cfg = {}
    cfg.update(prProcessCfg)
    X = []
    y = []
    input_size = max(0, len(batch[0]["dates"]))
    for b in batch:
        news_ = []
        closes_ = []
        for input in range(input_size):
            day_article = b["articles"][input]
            if day_article is not None:
                day_article = day_article[0]
            else:
                day_article = ""
            news_.append(day_article)

            time_series = b["time_series"][input]
            if time_series is not None:
                time_series = time_series["close"]
                closes_.append(time_series)
            else:
                closes_.append(None)
        _, inputs = tokenize_sentences(news_, config=prProcessCfg)
        news_ = inputs["input_ids"]
        closes_ = _replace_none_with_avg(closes_)
        mean, std, closes_ = _standardize_list(closes_)

        targets_ = b["target"]
        targets_ = _replace_none_with_avg(targets_)
        _, _, targets_ = _standardize_list(targets_, mean, std)
        X.append([news_, closes_, mean, std])
        y.append(targets_)

    return X, y

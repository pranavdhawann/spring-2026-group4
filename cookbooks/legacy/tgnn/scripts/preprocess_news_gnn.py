"""
preprocess_news_gnn.py — Offline FinBERT batch encoding of FinMultiTime news articles.

FinMultiTime news schema (JSONL):
    {TICKER_UPPERCASE}.jsonl — one JSON object per line:
    {
        "Date": "2025-04-15",
        "Url": "https://...",
        "Article": "full article text",
        "Stock_symbol": "a",          # lowercase
        "Article_title": "optional headline"
    }

Output: {cache_dir}/{TICKER}_news_embeddings.pt
    dict[date_str, tensor(num_articles, embed_dim)]
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils_gnn import (
    assign_news_to_trading_day,
    build_trading_calendar,
    DEFAULT_CONFIG_PATH,
    load_config,
    log_runtime_context,
    resolve_data_path,
    setup_logging,
)

logger = logging.getLogger(__name__)


def load_news_articles(news_dir: str) -> Dict[str, List[dict]]:
    """
    Load news articles from FinMultiTime JSONL files.

    File naming: {TICKER_UPPERCASE}.jsonl (e.g. AAPL.jsonl)
    Fields: Date, Url, Article, Stock_symbol, Article_title
    """
    all_news = {}

    if not os.path.exists(news_dir):
        logger.warning(f"News directory not found: {news_dir}")
        return all_news

    for fname in sorted(os.listdir(news_dir)):
        if not (fname.endswith(".jsonl") or fname.endswith(".json")):
            continue

        ticker = fname.split(".")[0].upper()
        fpath = os.path.join(news_dir, fname)
        articles = []

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        article = json.loads(line)
                        articles.append(article)
                    except json.JSONDecodeError:
                        if line_num <= 3:
                            logger.debug(f"Skipped invalid JSON in {fname} line {line_num}")

            if articles:
                all_news[ticker] = articles
        except Exception as e:
            logger.warning(f"Failed to load {fpath}: {e}")

    total = sum(len(v) for v in all_news.values())
    logger.info(f"Loaded {total} articles across {len(all_news)} tickers from {news_dir}")
    return all_news


def encode_articles_batch(texts, tokenizer, model, device, batch_size=32, max_length=512, pooling="mean"):
    """Batch-encode texts using a transformer model.

    Args:
        pooling: one of
            "mean"    — attention-mask-weighted mean of last_hidden_state
                        (default; best for regression on stock returns).
            "cls"     — last_hidden_state[:, 0, :] (classic [CLS] token).
            "pooler"  — outputs.pooler_output when available, else falls back
                        to mean.  For FinBERT this is the fine-tuned
                        sentiment head, which is NOT what we want.

    FIX E12: the previous implementation checked ``hasattr(outputs,
    "last_hidden_state")`` first, which is True for every HF model, so
    the pooler and fallback branches were dead code.  The default is now
    mean-pooling over the attention mask — well-studied to outperform [CLS]
    for sentence-embedding regression tasks.
    """
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        logger.debug("Encoding article batch %d-%d / %d", i + 1, min(i + batch_size, len(texts)), len(texts))

        encoded = tokenizer(
            batch_texts, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)

        if pooling == "cls" and hasattr(outputs, "last_hidden_state"):
            embeddings = outputs.last_hidden_state[:, 0, :]
        elif pooling == "pooler" and hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            embeddings = outputs.pooler_output
        else:
            # Default: attention-mask-weighted mean pooling over tokens.
            if hasattr(outputs, "last_hidden_state"):
                token_embeddings = outputs.last_hidden_state
            else:
                token_embeddings = outputs[0]
            attention_mask = encoded.get("attention_mask", torch.ones_like(encoded["input_ids"]))
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * mask_expanded, 1) / mask_expanded.sum(1).clamp(min=1e-9)

        all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)


def preprocess_news(config: dict):
    """Main pipeline: load articles → encode with FinBERT → cache per-ticker .pt files."""
    data_cfg = config["data"]
    data_dir = data_cfg["data_dir"]
    news_subdir = data_cfg.get("news_dir", "sp500_news")
    encoder_model_name = data_cfg.get("news_encoder_model", "ProsusAI/finbert")
    pooling = data_cfg.get("news_pooling", "mean")  # FIX E12: default "mean"
    cutoff_time = data_cfg.get("news_cutoff_time", "16:00")
    cutoff_hour, cutoff_minute = map(int, cutoff_time.split(":"))

    _, news_dir = resolve_data_path(
        data_dir,
        news_subdir,
        "sp500_news",
        kind="directory",
        aliases=["news"],
    )
    _, cache_dir = resolve_data_path(
        data_dir,
        data_cfg.get("news_cache_dir"),
        os.path.join("cache", "news_embeddings"),
        kind="directory",
    )
    os.makedirs(cache_dir, exist_ok=True)
    logger.info("News preprocessing paths | source=%s | cache=%s", os.path.abspath(news_dir), os.path.abspath(cache_dir))

    all_news = load_news_articles(news_dir)
    if not all_news:
        logger.warning("No news articles found.")
        return

    trading_cal = build_trading_calendar()

    logger.info(f"Loading encoder: {encoder_model_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(encoder_model_name)
    model = AutoModel.from_pretrained(encoder_model_name).to(device)
    model.eval()

    embed_dim = model.config.hidden_size
    logger.info(f"Encoder: {embed_dim}-dim embeddings on {device}")

    skipped_cached = 0
    encoded_tickers = 0
    for ticker, articles in tqdm(all_news.items(), desc="Encoding tickers"):
        cache_file = os.path.join(cache_dir, f"{ticker}_news_embeddings.pt")
        if os.path.exists(cache_file):
            skipped_cached += 1
            logger.debug("Skipping %s because cache already exists at %s", ticker, cache_file)
            continue

        # Build text from Article_title + Article (FinMultiTime schema)
        texts = []
        dates = []

        for article in articles:
            title = article.get("Article_title", "") or ""
            body = article.get("Article", "") or ""
            text = f"{title}. {body}" if body else title
            if not text.strip():
                continue

            date_str = article.get("Date", "")
            if not date_str:
                continue
            try:
                article_date = pd.Timestamp(date_str)
            except (ValueError, TypeError):
                continue

            texts.append(text[:2048])  # Truncate very long articles
            dates.append(article_date)

        if not texts:
            logger.debug("Skipping %s because no valid article text remained after cleaning", ticker)
            continue

        embeddings = encode_articles_batch(
            texts, tokenizer, model, device,
            batch_size=32, max_length=512, pooling=pooling,
        )

        # Group by trading day
        daily_embeddings = defaultdict(list)
        for emb, article_date in zip(embeddings, dates):
            try:
                trading_day = assign_news_to_trading_day(
                    article_date, trading_cal,
                    cutoff_hour=cutoff_hour, cutoff_minute=cutoff_minute,
                )
                daily_embeddings[str(trading_day.date())].append(emb)
            except (ValueError, IndexError):
                continue

        ticker_cache = {}
        for date_key, emb_list in daily_embeddings.items():
            ticker_cache[date_key] = torch.stack(emb_list)

        torch.save(ticker_cache, cache_file)
        encoded_tickers += 1
        logger.info(
            "Cached %s news embeddings | articles=%d | trading_days=%d | output=%s",
            ticker,
            len(texts),
            len(ticker_cache),
            cache_file,
        )

    logger.info(
        "Done. Embeddings cached to %s | encoded=%d | skipped_cached=%d",
        cache_dir,
        encoded_tickers,
        skipped_cached,
    )


def main():
    parser = argparse.ArgumentParser(description="Preprocess FinMultiTime news with FinBERT")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    log_path = setup_logging(config, command_name="preprocess_news", config_path=args.config, args=args)
    logger.info("Loaded config from %s", os.path.abspath(args.config))
    log_runtime_context("preprocess_news", config, extra={"preprocess_log_path": log_path})
    if args.force:
        _, cache_dir = resolve_data_path(
            config["data"]["data_dir"],
            config["data"].get("news_cache_dir"),
            os.path.join("cache", "news_embeddings"),
            kind="directory",
        )
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)
            logger.info("Removed existing news cache at %s because --force was set", os.path.abspath(cache_dir))
    preprocess_news(config)


if __name__ == "__main__":
    main()

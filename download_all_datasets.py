import os
import shutil

import kagglehub
import pandas as pd
import yfinance as yf
from datasets import load_dataset

# Create data directory
os.makedirs("data", exist_ok=True)

print("Step 1: Downloading stock prices...")
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "JPM", "BAC", "WMT", "V", "JNJ"]
stock_data = yf.download(
    tickers, start="2015-01-01", end="2024-12-31", group_by="ticker"
)
stock_data.to_csv("data/stock_prices.csv")
print("✓ Stock prices saved to data/stock_prices.csv")

print("\nStep 2: Downloading Financial PhraseBank...")
fp_dataset = load_dataset("financial_phrasebank", "sentences_allagree")
pd.DataFrame(fp_dataset["train"]).to_csv("data/financial_phrasebank.csv", index=False)
print("✓ Financial PhraseBank saved to data/financial_phrasebank.csv")

try:
    url = "https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests"
    print("\nStep 3: Downloading Kaggle Financial News...")
    path = kagglehub.dataset_download(
        "miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests"
    )
    if os.path.isdir(path):
        for item in os.listdir(path):
            src = os.path.join(path, item)
            dst = os.path.join("data", item)
            shutil.move(src, dst)
            print(f"✓ Kaggle Financial News saved to {dst}")
    shutil.rmtree(path)
except Exception as e:
    print(f"{e}: download manually from :  {url}")

try:
    url = "https://www.kaggle.com/datasets/sulphatet/twitter-financial-news"
    print("\nStep 4: Downloading Twitter Financial...")
    path = kagglehub.dataset_download("sulphatet/twitter-financial-news")
    if os.path.isdir(path):
        for item in os.listdir(path):
            src = os.path.join(path, item)
            dst = os.path.join("data", "twitter_financial_" + item)
            shutil.move(src, dst)
            print(f"✓ Twitter Financial saved to {dst}")
    shutil.rmtree(path)
except Exception as e:
    print(f"{e}: download manually from :  {url}")

print("\n✓ Automatic downloads complete!")
print("\nManual downloads needed:")
print("1. StockNet: git clone https://github.com/yumoxu/stocknet-dataset.git")
print("\nRun: kaggle datasets download -d <dataset-id> (after setting up Kaggle API)")

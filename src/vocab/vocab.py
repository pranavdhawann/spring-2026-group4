import json
import pandas as pd
from collections import Counter
import re
import os

class VocabPipeline:
    def __init__(self, csv_path, jsonl_dir, output_dir):
        self.csv_path = csv_path
        self.jsonl_dir = jsonl_dir
        self.output_dir = output_dir
        self.word_freq = Counter()
        self.special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']

    def get_available_stocks(self, k):
        df = pd.read_csv(self.csv_path)
        df.columns = df.columns.str.strip().str.replace('\ufeff', '')

        all_csv_stocks = df['ticker'].str.upper().tolist()

        if not os.path.exists(self.jsonl_dir):
            print(f"Error: Directory {self.jsonl_dir} not found. Please check your path.")
            return []

        files_in_folder = {f.replace('.jsonl', '') for f in os.listdir(self.jsonl_dir) if f.endswith('.jsonl')}

        available = [s for s in all_csv_stocks if s in files_in_folder]

        print(f"Total available files found: {len(available)}")
        return available[:k]

    def process_files(self, stock_list):
        for stock in stock_list:
            filename = f"{stock}.jsonl"
            file_path = os.path.join(self.jsonl_dir, filename)

            print(f"Reading: {filename}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line)
                        content = data.get('Article', '').lower()
                        # Tokenization
                        words = re.findall(r'\b\w+\b', content)
                        self.word_freq.update(words)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    def save_vocab(self, filename='vocab.txt'):
        output_path = os.path.join(self.output_dir, filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            for token in self.special_tokens:
                f.write(f"{token}\n")
            for word, _ in self.word_freq.most_common():
                f.write(f"{word}\n")

        print(f"\n--- DONE ---")
        print(f"Vocabulary saved to: {output_path}")
        print(f"Unique tokens extracted: {len(self.word_freq)}")



if __name__ == "__main__":

    CSV_FILE = 'data/stock_scores_news_1.csv'
    NEWS_DATA_DIR = 'data/sp500_news/'
    OUTPUT_DIR = '.'

    pipeline = VocabPipeline(CSV_FILE, NEWS_DATA_DIR, OUTPUT_DIR)

    try:
        k_val = int(input("Enter number of available stocks (k) to process: "))

        target_stocks = pipeline.get_available_stocks(k_val)

        if not target_stocks:
            print("No matching files found. Check your 'src/data/sp500_news/' folder.")
        else:
            print(f"Processing the following stocks: {target_stocks}")

            pipeline.process_files(target_stocks)

            pipeline.save_vocab()

    except ValueError:
        print("Invalid input. Please enter a number.")
    except Exception as e:
        print(f"An error occurred: {e}")
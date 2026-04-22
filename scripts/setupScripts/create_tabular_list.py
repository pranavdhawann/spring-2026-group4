import csv
import os

from src.utils import read_yaml

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
config_path = os.path.join(project_root, "config", "config.yaml")


def create_ticker_csv():
    config = read_yaml(config_path)

    tabular_data_rel_path = config.get("TABULAR_DATA")
    if not tabular_data_rel_path:
        print("Error: TABULAR_DATA not found in config.")
        return

    sp500_table_dir = os.path.join(project_root, tabular_data_rel_path)

    output_dir = os.path.dirname(sp500_table_dir)
    output_csv = os.path.join(output_dir, "table_tickers.csv")

    # Ensure the directory exists
    if not os.path.exists(sp500_table_dir):
        print(f"Error: Directory {sp500_table_dir} does not exist.")
        return

    tickers = [
        name
        for name in os.listdir(sp500_table_dir)
        if os.path.isdir(os.path.join(sp500_table_dir, name))
    ]

    tickers.sort()
    print(tickers[0])
    if tickers[0] == "TRUE":
        tickers = tickers[1:]
    print(f"Found {len(tickers)} tickers.")

    try:
        with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["ticker"])
            for ticker in tickers:
                writer.writerow([ticker])
        print(f"Successfully created {output_csv}")
    except Exception as e:
        print(f"Error writing to CSV: {e}")


if __name__ == "__main__":
    create_ticker_csv()

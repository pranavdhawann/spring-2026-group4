import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

# Add project root to sys path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.utils import load_stock_csv


def main():
    parser = argparse.ArgumentParser(
        description="Filter Chronos data by specific criteria"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to data directory containing CSVs",
    )
    parser.add_argument(
        "--output_manifest",
        type=str,
        required=True,
        help="Path to output JSON manifest",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.output_manifest)

    if not data_dir.exists():
        print(f"Error: Directory {data_dir} does not exist.")
        return

    # Find all CSV files in the data directory and subdirectories
    csv_files = list(data_dir.rglob("*.csv"))
    total_files = len(csv_files)

    if total_files == 0:
        print(f"No CSV files found in {data_dir}")
        return

    print(f"Scanning {total_files} CSV files in {data_dir}...")

    passed_tickers = []

    # Iterate over each CSV
    for csv_file in tqdm(csv_files, desc="Filtering CSVs"):
        ticker = csv_file.stem
        parent_dir = csv_file.parent

        try:
            # 1. Parse Date as datetime (handled gracefully by `load_stock_csv`)
            df = load_stock_csv(ticker, parent_dir)

            # 2. Count total rows
            total_rows = len(df)
            if total_rows == 0:
                continue

            # 3. Check NaN percentage in Close
            nan_count = df["Close"].isna().sum()
            nan_pct = nan_count / total_rows

            # 4. Compute date range span in calendar days
            min_date = df["Date"].min()
            max_date = df["Date"].max()
            date_span = (max_date - min_date).days

            # Check filtering conditions:
            # - minimum 500 rows
            # - less than 5% NaN in Close
            # - date span at least 2 years (~730 days)
            if total_rows >= 500 and nan_pct < 0.05 and date_span >= 730:
                passed_tickers.append(
                    {
                        "ticker": ticker,
                        "rows": int(total_rows),
                        "nan_pct_close": float(nan_pct),
                        "span_days": int(date_span),
                        "file_path": str(csv_file),
                    }
                )
        except Exception:
            # Skip file on read or parse error
            pass

    # Output tally and logging
    passed_count = len(passed_tickers)
    print("\n" + "=" * 50)
    print("Filtering complete.")
    print(f"Total CSVs parsed : {total_files}")
    print(f"Passed criteria   : {passed_count}")
    print(f"Failed criteria   : {total_files - passed_count}")
    print("=" * 50)

    # Save the manifest to JSON
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(passed_tickers, f, indent=2)

    print(f"\nManifest successfully written to {out_path}")


if __name__ == "__main__":
    main()

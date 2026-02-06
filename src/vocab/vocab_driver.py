from pathlib import Path
from vocab import (
    CharVocabConfig,
    setup_logging,
    generate_char_vocab,
)

CSV_PATH = Path("../data/stock_scores_news_1.csv")


def main() -> None:
    setup_logging()

    k = int(input("Enter K (number of stocks to include): ").strip())

    config = CharVocabConfig(
        csv_path=CSV_PATH,
        top_k=k,
        text_columns=("sector",),
        output_dir=Path(".."),
    )

    vocab_path = generate_char_vocab(config)
    print(f"Character vocab generated at: {vocab_path.resolve()}")


if __name__ == "__main__":
    main()

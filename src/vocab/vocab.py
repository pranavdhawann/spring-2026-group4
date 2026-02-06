from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence
from collections import Counter
import json
import logging

import pandas as pd


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CharVocabConfig:
    csv_path: Path
    output_dir: Path = Path("..")
    output_vocab_name: str = "char_vocab.txt"

    # row selection
    top_k: int = 10
    score_column: str = "score"

    # text source
    text_columns: Sequence[str] = ("sector",)
    lowercase: bool = True

    # special tokens
    special_tokens: Sequence[str] = ("[PAD]", "[UNK]")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def validate_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def select_top_k(df: pd.DataFrame, score_column: str, k: int) -> pd.DataFrame:
    if k <= 0:
        raise ValueError("top_k must be positive")
    return df.sort_values(by=score_column, ascending=False).head(k)


def iter_text(df: pd.DataFrame, columns: Sequence[str]) -> Iterable[str]:
    for col in columns:
        for v in df[col].dropna():
            s = str(v)
            if s:
                yield s.lower() if True else s


def count_characters(texts: Iterable[str]) -> Counter:
    c = Counter()
    for t in texts:
        c.update(list(t))
    return c


def build_char_vocab(
    char_counts: Counter,
    special_tokens: Sequence[str],
) -> List[str]:
    """
    Order:
    1. special tokens
    2. characters by frequency (desc)
    3. tiebreaker: unicode order
    """
    vocab = list(special_tokens)

    chars = [(ch, freq) for ch, freq in char_counts.items() if ch not in set(special_tokens)]
    chars.sort(key=lambda x: (-x[1], x[0]))

    for ch, _ in chars:
        vocab.append(ch)

    return vocab


def write_char_vocab(vocab: List[str], output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    path.write_text("\n".join(vocab) + "\n", encoding="utf-8")
    LOGGER.info("Wrote %s (%d symbols)", path, len(vocab))


def write_mappings(vocab: List[str], output_dir: Path) -> None:
    char_to_id: Dict[str, int] = {ch: i for i, ch in enumerate(vocab)}
    id_to_char: Dict[str, str] = {str(i): ch for i, ch in enumerate(vocab)}

    (output_dir / "char_to_id.json").write_text(
        json.dumps(char_to_id, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "id_to_char.json").write_text(
        json.dumps(id_to_char, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def generate_char_vocab(config: CharVocabConfig) -> Path:
    df = load_csv(config.csv_path)
    validate_columns(df, [config.score_column, *config.text_columns])

    df_k = select_top_k(df, config.score_column, config.top_k)

    texts = iter_text(df_k, config.text_columns)
    counts = count_characters(texts)

    vocab = build_char_vocab(counts, config.special_tokens)

    write_char_vocab(vocab, config.output_dir, config.output_vocab_name)
    write_mappings(vocab, config.output_dir)

    return config.output_dir / config.output_vocab_name

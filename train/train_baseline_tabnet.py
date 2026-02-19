# CRITICAL: Set CUDA env var BEFORE any PyTorch imports
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

"""
Train TabNet on baseline JSONL for stock price forecasting.
Uses NLP (text from articles) + time_series + table_data as tabular input.
Results: experiments/baseline/baseline_results_tabnet/
Metrics: MAE, MSE, RMSE, MAPE, SMAPE.
Plots: Date vs stock price; input price, predicted price, actual price (up to 10).
"""

import json

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch


from src.dataLoader.dataLoaderTabNet import build_tabnet_features

from src.models.tabnet_forecasting import TabNetForecasting
from src.utils import read_json_file, read_yaml, set_seed
from src.utils.metrics_utils import calculate_regression_metrics, print_metrics

# Patch set_seed to disable deterministic algorithms for TabNet (avoids CuBLAS error)
_original_set_seed = set_seed
def _safe_set_seed(seed=42):
    """Set seed without deterministic algorithms to avoid CuBLAS issues on GPU."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = False  # Disable for TabNet
    torch.backends.cudnn.benchmark = True  # Enable for performance
    # Don't call torch.use_deterministic_algorithms(True) - causes CuBLAS error
    print(f"Random seed set to {seed} (non-deterministic mode for TabNet)")

# Replace set_seed with safe version
import src.utils.utils as utils_module
utils_module.set_seed = _safe_set_seed
set_seed = _safe_set_seed



# Text encoding limits: more = better context but slower and more GPU memory
TEXT_MAX_ARTICLES = 50   # number of articles per sample to encode (was 20)
TEXT_MAX_LENGTH = 256    # max tokens per article (was 128)


def _make_text_encoder(embed_dim: int = 64, max_length: int = None, max_articles: int = None):
    """
    Build a text encoder that maps list of article strings -> vector of size embed_dim.
    Uses HuggingFace transformer + mean pool for NLP-derived features.
    """
    if max_length is None:
        max_length = TEXT_MAX_LENGTH
    if max_articles is None:
        max_articles = TEXT_MAX_ARTICLES
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        return None

    model_name = "ProsusAI/finbert"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    except Exception:
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    def _encode(texts):
        if not texts:
            return np.zeros(embed_dim, dtype=np.float32)
        texts = texts[:max_articles]
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc).last_hidden_state  # (batch, seq, 768)
            pooled = out.mean(dim=1)  # (batch, 768)
            vec = pooled.cpu().numpy().mean(axis=0).astype(np.float32)
        if len(vec) > embed_dim:
            vec = vec[:embed_dim]
        elif len(vec) < embed_dim:
            vec = np.pad(vec, (0, embed_dim - len(vec)))
        return vec

    return _encode


def _build_features_with_encoder(config, use_nlp_encoder: bool = True):
    """Build TabNet features; optionally use FinBERT-style text encoder."""
    text_encoder = None
    text_dim = 64
    if use_nlp_encoder:
        enc = _make_text_encoder(embed_dim=64, max_length=TEXT_MAX_LENGTH, max_articles=TEXT_MAX_ARTICLES)
        if enc is not None:
            text_encoder = enc
            text_dim = 64
    return build_tabnet_features(
        config, text_encoder=text_encoder, text_dim=text_dim
    )


def _save_metrics(metrics: dict, out_dir: Path):
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {path}")


def _generate_plots(
    meta_test: list,
    y_pred: np.ndarray,
    y_test: np.ndarray,
    out_dir: Path,
    max_plots: int = 10,
):
    """Generate up to 10 plots: date vs stock price; input, predicted, actual."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    n_plots = min(max_plots, len(meta_test or []), len(y_pred))
    if n_plots == 0:
        print("WARNING: No test samples for plots (meta_test or y_pred empty).")
        return

    for i in range(n_plots):
        meta = meta_test[i]
        dates = meta.get("dates", [])
        input_price = meta.get("input_price", np.nan)
        actual = y_test[i] if i < len(y_test) else np.array(meta.get("target", []))
        pred = y_pred[i] if i < len(y_pred) else np.full_like(actual, np.nan)

        # Build date axis: last history date + 7 future dates
        n_hist = len(dates)
        n_fut = len(actual)
        try:
            last_d = datetime.strptime(dates[-1], "%Y-%m-%d") if dates else datetime.now()
        except Exception:
            last_d = datetime.now()
        future_dates = [
            (last_d + timedelta(days=k)).strftime("%Y-%m-%d")
            for k in range(1, n_fut + 1)
        ]
        all_dates = dates + future_dates
        x_hist = list(range(n_hist))
        x_fut = list(range(n_hist, n_hist + n_fut))

        # Prices: history closes from meta (we don't have them in meta - use input_price at last point)
        # So we show: one point "input" at last hist date, then pred vs actual for future
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axvline(x=n_hist - 0.5, color="gray", linestyle="--", alpha=0.7)
        ax.scatter([n_hist - 1], [input_price], color="blue", s=80, zorder=5, label="Input (last close)")
        ax.plot(x_fut, pred, "o-", color="green", label="Predicted", markersize=6)
        ax.plot(x_fut, actual, "s-", color="red", label="Actual", markersize=6)
        ax.set_xticks(list(range(len(all_dates))))
        ax.set_xticklabels(all_dates, rotation=45, ha="right")
        ax.set_xlabel("Date")
        ax.set_ylabel("Stock price (Close)")
        ax.set_title(f"Sample {i+1} | Ticker: {meta.get('ticker_text', '')}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"plot_sample_{i+1}.png", dpi=150)
        plt.close()

    # One overview: date vs stock price for first sample (input + predicted + actual)
    if n_plots >= 1 and meta_test and len(meta_test[0].get("dates", [])) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        meta = meta_test[0]
        hist_dates = meta.get("dates", [])
        try:
            last_d = datetime.strptime(hist_dates[-1], "%Y-%m-%d")
        except Exception:
            last_d = datetime.now()
        future_dates = [
            (last_d + timedelta(days=k)).strftime("%Y-%m-%d")
            for k in range(1, len(y_pred[0]) + 1)
        ]
        all_dates = hist_dates + future_dates
        x_hist_last = len(hist_dates) - 1
        x_fut = list(range(len(hist_dates), len(all_dates)))
        ax.scatter([x_hist_last], [meta.get("input_price")], color="blue", s=80, label="Input price")
        ax.plot(x_fut, y_pred[0], "g-o", label="Predicted price", markersize=6)
        ax.plot(x_fut, y_test[0], "r-s", label="Actual price", markersize=6)
        ax.set_xticks(range(len(all_dates)))
        ax.set_xticklabels(all_dates, rotation=45, ha="right")
        ax.set_xlabel("Date")
        ax.set_ylabel("Stock price (Close)")
        ax.set_title("Date vs stock price (sample 0)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "plot_date_vs_price.png", dpi=150)
        plt.close()

    print(f"Saved up to {n_plots + 1} plots to {out_dir}")


def train(train_config: dict = None):
    # Set seed without deterministic algorithms to avoid CuBLAS issues
    import random
    random.seed(42)
    np.random.seed(42)
    # Don't call set_seed() as it enables deterministic algorithms which conflicts with CuBLAS on GPU
    train_config = train_config or {}
    project_root = Path(__file__).resolve().parent.parent

    config = {
        "yaml_config_path": str(project_root / "config" / "config.yaml"),
        "experiment_path": str(project_root / "experiments" / "baseline" / "baseline_results_tabnet"),
    }
    config.update(train_config)

    yaml_config = read_yaml(config["yaml_config_path"])
    config.update(yaml_config)

    ticker2idx = read_json_file(
        os.path.join(config["BASELINE_DATA_PATH"], config["TICKER2IDX"])
    )
    data_config = {
        "data_path": config["BASELINE_DATA_PATH"],
        "ticker2idx": ticker2idx,
        "test_train_split": 0.2,
        "random_seed": 42,
    }
    config.update(data_config)

    print("Building TabNet features from baseline JSONL...")
    X_train, y_train, X_test, y_test, meta_train, meta_test = _build_features_with_encoder(
        config, use_nlp_encoder=True
    )

    # Replace NaN in targets with 0 for training (and mask in eval if needed)
    y_train_nan_mask = np.isnan(y_train)
    y_train_fit = np.nan_to_num(y_train, nan=0.0)
    # Eval only on valid targets
    y_test_flat = y_test.flatten()
    y_test_valid = y_test_flat[~np.isnan(y_test_flat)]

    print(f"Train: X {X_train.shape}, y {y_train.shape}")
    print(f"Test:  X {X_test.shape}, y {y_test.shape}")

    model_config = {
        "output_dim": y_train.shape[1],
        "max_epochs": config.get("tabnet_max_epochs", 30),
        "patience": config.get("tabnet_patience", 10),
        "batch_size": config.get("tabnet_batch_size", 512),
    }
    model = TabNetForecasting(model_config)

    # Optional validation split from train
    n_train = len(X_train)
    val_size = max(1, int(0.15 * n_train))
    X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
    y_tr, y_val = y_train_fit[:-val_size], y_train_fit[-val_size:]

    # Ensure deterministic mode is disabled before TabNet training (avoids CuBLAS error)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
    print("Training TabNet...")
    model.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)

    out_dir = Path(config["experiment_path"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Output only metrics and plots (no model file)
    # model.save_model(str(out_dir / "tabnet_model"))  # disabled

    y_pred = model.predict(X_test)
    y_pred_flat = y_pred.flatten()
    y_test_flat = y_test.flatten()
    valid = ~np.isnan(y_test_flat)
    if np.any(valid):
        metrics = calculate_regression_metrics(
            y_test_flat[valid], y_pred_flat[valid]
        )
    else:
        metrics = {
            "mae": float("nan"), "mse": float("nan"), "rmse": float("nan"),
            "mape": float("nan"), "smape": float("nan"),
        }
    print_metrics(metrics, prefix="Test")

    # Always save metrics and plots (with error handling so we get something)
    try:
        _save_metrics(metrics, out_dir)
        print(f"Metrics written to {out_dir / 'metrics.json'}")
    except Exception as e:
        print(f"WARNING: Could not save metrics: {e}")

    try:
        _generate_plots(meta_test, y_pred, y_test, out_dir, max_plots=10)
        print(f"Plots written to {out_dir}")
    except Exception as e:
        print(f"WARNING: Could not save plots: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    train()

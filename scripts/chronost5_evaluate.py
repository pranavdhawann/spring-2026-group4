#!/usr/bin/env python3
"""
chronost5_evaluate.py - Unified Chronos T5 preprocessing, forecasting, and evaluation CLI.

Subcommands:
    preprocess  Clean raw per-ticker CSVs and compute log returns
    forecast    Run held-out backtest forecasts and save forecast CSVs/plots
    eval        Run held-out evaluation and save per-ticker/aggregate metrics
    compare     Compare two evaluation runs side by side
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 11,
    }
)

KEEP_COLS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def parse_quantiles(raw: str) -> list[int]:
    """Parse a comma-separated percentile list like ``10,50,90``."""
    parts = [part.strip() for part in str(raw).split(",")]
    if not parts or any(not part for part in parts):
        raise argparse.ArgumentTypeError(
            "Quantiles must be a comma-separated list like 10,50,90."
        )

    try:
        quantiles = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Quantiles must be whole-number percentiles like 10,50,90."
        ) from exc

    if any(q <= 0 or q >= 100 for q in quantiles):
        raise argparse.ArgumentTypeError("Quantiles must be between 1 and 99.")

    return sorted(set(quantiles))


def parse_plot_bands(raw: str) -> list[tuple[int, int]]:
    """Parse comma-separated percentile bands like ``10-90,30-40``."""
    parts = [part.strip() for part in str(raw).split(",")]
    if not parts or any(not part for part in parts):
        raise argparse.ArgumentTypeError(
            "Plot bands must be a comma-separated list like 10-90,30-40."
        )

    bands = []
    for part in parts:
        pieces = [piece.strip() for piece in part.split("-")]
        if len(pieces) != 2 or any(not piece for piece in pieces):
            raise argparse.ArgumentTypeError(
                "Each plot band must look like lower-upper, for example 10-90."
            )

        try:
            lower = int(pieces[0])
            upper = int(pieces[1])
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "Plot band endpoints must be whole-number percentiles."
            ) from exc

        if lower <= 0 or upper >= 100:
            raise argparse.ArgumentTypeError(
                "Plot band percentiles must be between 1 and 99."
            )
        if lower >= upper:
            raise argparse.ArgumentTypeError(
                "Each plot band must have a lower percentile smaller than the upper percentile."
            )

        bands.append((lower, upper))

    return bands


def quantile_label(quantile: int) -> str:
    """Return the column-friendly label for an integer percentile."""
    return f"Q{quantile}"


def compute_quantile_arrays(
    samples: np.ndarray, quantiles: list[int]
) -> dict[int, np.ndarray]:
    """Compute percentile paths from sampled forecast trajectories."""
    return {
        quantile: np.quantile(samples, quantile / 100.0, axis=0)
        for quantile in quantiles
    }


def merge_plot_band_quantiles(
    quantiles: list[int],
    plot_bands: list[tuple[int, int]] | None,
) -> list[int]:
    """Ensure any plotted band endpoints are also computed as forecast quantiles."""
    if not plot_bands:
        return list(quantiles)

    merged = set(quantiles)
    for lower, upper in plot_bands:
        merged.add(lower)
        merged.add(upper)
    return sorted(merged)


def resolve_plot_bands(
    quantiles: list[int],
    plot_bands: list[tuple[int, int]] | None,
) -> list[tuple[int, int]]:
    """Return the bands to shade on the plot."""
    if plot_bands is not None:
        return list(plot_bands)
    if len(quantiles) >= 2:
        return [(quantiles[0], quantiles[-1])]
    return []


def load_chronos_pipeline(model_path: str, device: str):
    """Load a ChronosPipeline from HF Hub or another compatible checkpoint."""
    from chronos import ChronosPipeline

    return ChronosPipeline.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.bfloat16,
    )


def lr_to_close(log_rets: np.ndarray, base_close: float) -> np.ndarray:
    """Convert a path of log returns into a close-price path."""
    return base_close * np.exp(np.cumsum(log_rets))


def clean_single_csv(fpath: Path) -> pd.DataFrame | None:
    """Read one ticker CSV and return Date + OHLCV + Log_Return."""
    try:
        df = pd.read_csv(fpath)
    except Exception as exc:
        print(f"  [SKIP] Cannot read {fpath.name}: {exc}")
        return None

    missing = [column for column in KEEP_COLS if column not in df.columns]
    if missing:
        print(f"  [SKIP] {fpath.name} missing required columns: {missing}")
        return None

    df = df[KEEP_COLS].copy()
    df["Date"] = (
        pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None).dt.normalize()
    )

    for column in ["Open", "High", "Low", "Close", "Volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["Volume"] = df["Volume"].fillna(0)
    df = (
        df.sort_values("Date")
        .drop_duplicates(subset="Date", keep="last")
        .reset_index(drop=True)
    )

    nan_before = int(df["Close"].isna().sum())
    df["Close"] = df["Close"].ffill(limit=3)
    nan_after = int(df["Close"].isna().sum())
    if nan_after > 0:
        df = df.dropna(subset=["Close"])
    if nan_before > 0:
        filled = nan_before - nan_after
        print(
            f"  [{fpath.name}] Forward-filled {filled} NaN Close values, dropped {nan_after}"
        )

    df = df[df["Close"] > 0].reset_index(drop=True)
    if len(df) < 2:
        print(f"  [SKIP] {fpath.name}: fewer than 2 valid Close values")
        return None

    close_arr = df["Close"].to_numpy(dtype=np.float64)
    df["Log_Return"] = np.concatenate(
        [[np.nan], np.log(close_arr[1:] / close_arr[:-1])]
    )
    df = df.iloc[1:].reset_index(drop=True)

    bad_mask = ~np.isfinite(df["Log_Return"].to_numpy(dtype=np.float64))
    if bad_mask.any():
        df = df.loc[~bad_mask].reset_index(drop=True)

    if df.empty:
        print(f"  [SKIP] {fpath.name}: no valid log-return rows")
        return None

    return df[KEEP_COLS + ["Log_Return"]]


def prepare_backtest_window(
    df: pd.DataFrame,
    *,
    context_length: int,
    horizon: int,
    history_window: int,
) -> dict | None:
    """Split a cleaned ticker frame into history, model context, and held-out future rows."""
    if len(df) < context_length + horizon:
        return None

    train_df = df.iloc[:-horizon].copy()
    actual_future_df = df.iloc[-horizon:].copy()
    if len(train_df) < context_length:
        return None

    return {
        "history_df": train_df.tail(history_window).copy(),
        "actual_future_df": actual_future_df,
        "context": train_df["Log_Return"].to_numpy(dtype=np.float64)[-context_length:],
        "base_close": float(train_df["Close"].iloc[-1]),
        "last_date": pd.Timestamp(train_df["Date"].iloc[-1]),
    }


def add_quantile_columns(
    frame_data: dict,
    *,
    quantiles: list[int],
    quantile_values: dict[int, np.ndarray],
    suffix: str,
) -> None:
    """Append dynamic quantile columns to an output row/frame mapping."""
    for quantile in quantiles:
        frame_data[f"{quantile_label(quantile)}_{suffix}"] = quantile_values[quantile]


def build_forecast_frame(result: dict) -> pd.DataFrame:
    """Return per-step forecast output in both spaces plus actuals and errors."""
    point_cum = np.cumsum(result["point_lr"])
    actual_cum = np.cumsum(result["actual_lr"])
    frame_data = {
        "Date": result["dates"],
        "Point_Forecast_LogReturn": result["point_lr"],
    }
    add_quantile_columns(
        frame_data,
        quantiles=result["quantiles"],
        quantile_values=result["quantile_lr"],
        suffix="LogReturn",
    )
    frame_data["Actual_LogReturn"] = result["actual_lr"]
    frame_data["Point_Cumulative_LogReturn"] = point_cum
    add_quantile_columns(
        frame_data,
        quantiles=result["quantiles"],
        quantile_values={
            quantile: np.cumsum(result["quantile_lr"][quantile])
            for quantile in result["quantiles"]
        },
        suffix="Cumulative_LogReturn",
    )
    frame_data["Actual_Cumulative_LogReturn"] = actual_cum
    frame_data["Point_Forecast_Close"] = result["point_close"]
    add_quantile_columns(
        frame_data,
        quantiles=result["quantiles"],
        quantile_values=result["quantile_close"],
        suffix="Close",
    )
    frame_data["Actual_Close"] = result["actual_close"]
    frame_data["Error_LogReturn"] = result["point_lr"] - result["actual_lr"]
    frame_data["Error_Close"] = result["point_close"] - result["actual_close"]

    return pd.DataFrame(frame_data)


def build_summary_row(ticker: str, result: dict) -> dict:
    """Return a compact final-horizon summary in both spaces plus actual comparison."""
    horizon = len(result["point_lr"])
    forecast_cum = float(np.cumsum(result["point_lr"])[-1])
    actual_cum = float(np.cumsum(result["actual_lr"])[-1])
    forecast_close_last = float(result["point_close"][-1])
    actual_close_last = float(result["actual_close"][-1])
    pct_chg = (forecast_close_last / result["last_close"] - 1) * 100

    row = {
        "ticker": ticker,
        "last_known_date": str(result["last_date"].date()),
        "last_known_close": round(result["last_close"], 4),
        "forecast_log_return_last_day": round(float(result["point_lr"][-1]), 8),
        "actual_log_return_last_day": round(float(result["actual_lr"][-1]), 8),
        "forecast_log_return_error_last_day": round(
            float(result["point_lr"][-1] - result["actual_lr"][-1]), 8
        ),
        "forecast_cumulative_log_return": round(forecast_cum, 8),
        "actual_cumulative_log_return": round(actual_cum, 8),
        f"forecast_close_day{horizon}": forecast_close_last,
        f"actual_close_day{horizon}": actual_close_last,
        "forecast_close_last_day": forecast_close_last,
        "actual_close_last_day": actual_close_last,
        "forecast_close_error_last_day": forecast_close_last - actual_close_last,
        "predicted_pct_change": round(pct_chg, 4),
        "direction": "up" if pct_chg > 0 else "down",
    }
    for quantile in result["quantiles"]:
        row[f"{quantile_label(quantile).lower()}_log_return_last_day"] = round(
            float(result["quantile_lr"][quantile][-1]),
            8,
        )
        row[f"{quantile_label(quantile).lower()}_close_last_day"] = float(
            result["quantile_close"][quantile][-1]
        )

    return row


def save_forecast_plot(
    ticker: str,
    history_df: pd.DataFrame,
    result: dict,
    output_path: Path,
    history_window: int,
    plot_bands: list[tuple[int, int]] | None = None,
) -> None:
    """Render a sample-style backtest plot in close-price space."""
    history_slice = history_df.tail(history_window)
    hist_dates = pd.to_datetime(history_slice["Date"]).dt.tz_localize(None)
    hist_close = history_slice["Close"].to_numpy(dtype=np.float64)
    act_dates = pd.DatetimeIndex(pd.to_datetime(result["dates"])).tz_localize(None)
    act_close = result["actual_close"]
    pred_close = result["point_close"]

    pivot_date = hist_dates.iloc[-1]
    pivot_price = hist_close[-1]
    all_act_dates = [pivot_date] + list(act_dates)
    all_pred_dates = [pivot_date] + list(act_dates)
    all_act_prices = np.concatenate([[pivot_price], act_close])
    all_pred_prices = np.concatenate([[pivot_price], pred_close])

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(hist_dates, hist_close, color="#4C72B0", linewidth=2.0, label="History")
    ax.plot(
        all_act_dates,
        all_act_prices,
        color="#2ca02c",
        linewidth=2.0,
        marker="o",
        markersize=5,
        label="Actual",
    )
    ax.plot(
        all_pred_dates,
        all_pred_prices,
        color="#DD8452",
        linewidth=2.0,
        marker="s",
        markersize=5,
        label="Predicted",
    )
    bands_to_plot = resolve_plot_bands(result["quantiles"], plot_bands)
    band_alphas = [0.12, 0.2, 0.28, 0.36]
    for index, (lower_quantile, upper_quantile) in enumerate(
        sorted(bands_to_plot, key=lambda band: band[1] - band[0], reverse=True)
    ):
        if (
            lower_quantile not in result["quantile_close"]
            or upper_quantile not in result["quantile_close"]
        ):
            raise ValueError(
                f"Plot band {lower_quantile}-{upper_quantile} requires matching quantiles in the forecast output."
            )
        lower = np.concatenate(
            [[pivot_price], result["quantile_close"][lower_quantile]]
        )
        upper = np.concatenate(
            [[pivot_price], result["quantile_close"][upper_quantile]]
        )
        ax.fill_between(
            all_pred_dates,
            lower,
            upper,
            alpha=band_alphas[min(index, len(band_alphas) - 1)],
            color="#DD8452",
            label=f"{quantile_label(lower_quantile)}-{quantile_label(upper_quantile)}",
        )
    ax.axvline(pivot_date, color="gray", linestyle=":", linewidth=1.2)
    ax.text(
        pivot_date,
        0.02,
        " forecast start",
        color="gray",
        fontsize=8,
        va="bottom",
        transform=ax.get_xaxis_transform(),
    )
    ax.set_title(
        f"{ticker.upper()} - {len(hist_close)}-Day History + {len(act_close)}-Day Forecast"
    )
    ax.set_ylabel("Price ($)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def forecast_from_samples(
    samples: np.ndarray,
    *,
    dates,
    actual_lr: np.ndarray,
    actual_close: np.ndarray,
    base_close: float,
    last_date: pd.Timestamp,
    quantiles: list[int],
    history_df: pd.DataFrame | None = None,
) -> dict:
    """Shape sampled forecast trajectories into a consistent result payload."""
    point_lr = np.median(samples, axis=0)
    quantile_lr = compute_quantile_arrays(samples, quantiles)

    result = {
        "dates": pd.to_datetime(dates).tz_localize(None)
        if not isinstance(dates, pd.Series)
        else dates.dt.tz_localize(None),
        "quantiles": quantiles,
        "point_lr": point_lr,
        "quantile_lr": quantile_lr,
        "actual_lr": np.asarray(actual_lr, dtype=np.float64),
        "point_close": lr_to_close(point_lr, base_close),
        "quantile_close": {
            quantile: lr_to_close(quantile_lr[quantile], base_close)
            for quantile in quantiles
        },
        "actual_close": np.asarray(actual_close, dtype=np.float64),
        "last_close": base_close,
        "last_date": last_date,
    }
    if history_df is not None:
        result["history_df"] = history_df
    return result


def forecast_ticker(
    pipeline,
    df: pd.DataFrame,
    *,
    context_length: int,
    forecast_horizon: int,
    num_samples: int,
    history_window: int,
    quantiles: list[int],
) -> dict | None:
    """Run held-out backtest forecast for one ticker. Returns dict with results or None."""
    prepared = prepare_backtest_window(
        df,
        context_length=context_length,
        horizon=forecast_horizon,
        history_window=history_window,
    )
    if prepared is None:
        return None

    context = prepared["context"]
    bad_mask = ~np.isfinite(context)
    if bad_mask.any():
        context = context.copy()
        context[bad_mask] = 0.0

    if len(context) < 60:
        return None

    context_tensor = torch.tensor(context, dtype=torch.float32)
    samples = pipeline.predict(
        context_tensor,
        prediction_length=forecast_horizon,
        num_samples=num_samples,
    )
    samples = samples.squeeze(0).numpy()

    actual_future_df = prepared["actual_future_df"]
    return forecast_from_samples(
        samples,
        dates=actual_future_df["Date"],
        actual_lr=actual_future_df["Log_Return"].to_numpy(dtype=np.float64),
        actual_close=actual_future_df["Close"].to_numpy(dtype=np.float64),
        base_close=prepared["base_close"],
        last_date=prepared["last_date"],
        quantiles=quantiles,
        history_df=prepared["history_df"],
    )


def forecast_from_context(
    pipeline,
    context: np.ndarray,
    *,
    horizon: int,
    num_samples: int,
    quantiles: list[int],
) -> dict:
    """Run Chronos on a 1-D log-return context array and return forecast stats."""
    bad = ~np.isfinite(context)
    if bad.any():
        context = context.copy()
        context[bad] = 0.0

    context_tensor = torch.tensor(context, dtype=torch.float32)
    samples = pipeline.predict(
        context_tensor, prediction_length=horizon, num_samples=num_samples
    )
    samples = samples.squeeze(0).numpy()

    return {
        "quantiles": quantiles,
        "point_lr": np.median(samples, axis=0),
        "quantile_lr": compute_quantile_arrays(samples, quantiles),
    }


def compute_metrics(predicted: np.ndarray, actual: np.ndarray) -> dict:
    """Compute common forecast accuracy metrics."""
    errors = predicted - actual
    abs_err = np.abs(errors)
    pct_err = abs_err / np.where(actual == 0, 1e-9, np.abs(actual))
    smape = 2 * abs_err / (np.abs(predicted) + np.abs(actual) + 1e-9)

    pred_dir = (
        np.sign(predicted - predicted[0]) if len(predicted) > 1 else np.array([0])
    )
    act_dir = np.sign(actual - actual[0]) if len(actual) > 1 else np.array([0])
    dir_acc = np.mean(pred_dir == act_dir) if len(pred_dir) > 1 else float("nan")

    return {
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(abs_err)),
        "mape": float(np.mean(pct_err) * 100),
        "smape": float(np.mean(smape) * 100),
        "direction_accuracy": float(dir_acc),
        "final_predicted": float(predicted[-1]),
        "final_actual": float(actual[-1]),
    }


def run_preprocess(args) -> int:
    """Run the CSV cleaning pipeline."""
    src_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(src_dir.glob("*.csv"))
    if args.max_tickers:
        csv_files = csv_files[: args.max_tickers]
    print(
        f"Found {len(csv_files)} CSV files in {src_dir}/"
        f"{f' (limited to {args.max_tickers})' if args.max_tickers else ''}\n"
    )

    if not csv_files:
        print(f"ERROR: No .csv files found in {src_dir}")
        return 1

    accepted = 0
    rejected = 0
    for fpath in csv_files:
        df = clean_single_csv(fpath)
        if df is None:
            rejected += 1
            continue

        df.to_csv(out_dir / f"{fpath.stem}.csv", index=False)
        accepted += 1

    print(f"\nDone. Accepted: {accepted}, Rejected: {rejected}")
    print(f"Cleaned CSVs written to {out_dir}/")
    return 0


def run_forecast(args) -> int:
    """Run held-out backtest forecasting and save forecast artifacts."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    effective_quantiles = merge_plot_band_quantiles(args.quantiles, args.plot_bands)
    plot_bands = resolve_plot_bands(effective_quantiles, args.plot_bands)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    ticker_dir = out_dir / "tickers"
    plot_dir = out_dir / "plots"
    ticker_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model:          {args.model}")
    print(f"Data dir:       {data_dir}")
    print(f"Output dir:     {out_dir}")
    print(f"Context length: {args.context_length}")
    print(f"Horizon:        {args.horizon}")
    print(f"Num samples:    {args.num_samples}")
    print(
        f"Quantiles:      {', '.join(quantile_label(quantile) for quantile in effective_quantiles)}"
    )
    if plot_bands:
        print(
            "Plot bands:     "
            + ", ".join(
                f"{quantile_label(lower)}-{quantile_label(upper)}"
                for lower, upper in plot_bands
            )
        )
    print()

    pipeline = load_chronos_pipeline(args.model, args.device)

    csv_files = sorted(data_dir.glob("*.csv"))
    if args.max_tickers:
        csv_files = csv_files[: args.max_tickers]
    print(
        f"Found {len(csv_files)} cleaned CSVs"
        f"{f' (limited to {args.max_tickers})' if args.max_tickers else ''}\n"
    )

    summary_rows = []
    for fpath in csv_files:
        ticker = fpath.stem
        df = pd.read_csv(fpath, parse_dates=["Date"])

        result = forecast_ticker(
            pipeline,
            df,
            context_length=args.context_length,
            forecast_horizon=args.horizon,
            num_samples=args.num_samples,
            history_window=args.history_days,
            quantiles=effective_quantiles,
        )
        if result is None:
            print(f"  [SKIP] {ticker}: not enough data for context + holdout")
            continue

        forecast_df = build_forecast_frame(result)
        forecast_df.to_csv(ticker_dir / f"{ticker}_forecast.csv", index=False)

        save_forecast_plot(
            ticker=ticker,
            history_df=result["history_df"],
            result=result,
            output_path=plot_dir / f"{ticker}_forecast.png",
            history_window=args.history_days,
            plot_bands=plot_bands,
        )

        summary_rows.append(build_summary_row(ticker, result))

        final_pred = result["point_close"][-1]
        final_actual = result["actual_close"][-1]
        print(
            f"  {ticker:8s}  pred={final_pred:>10.2f}  "
            f"actual={final_actual:>10.2f}  error={final_pred - final_actual:+.2f}"
        )

    pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)
    print(f"\nSummary: {out_dir / 'summary.csv'}")
    print(f"Tickers: {ticker_dir}/")
    print(f"Plots:   {plot_dir}/")
    print(f"Total:   {len(summary_rows)} tickers forecast")
    return 0


def run_evaluation(args) -> int:
    """Run held-out backtest evaluation and save forecast metrics."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    detail_dir = out_dir / "per_ticker_forecasts"
    detail_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model:     {args.model}")
    print(f"Data:      {data_dir}")
    print(f"Horizon:   {args.horizon}")
    print(f"Context:   {args.context_length}")
    print(f"Samples:   {args.num_samples}")
    print(
        f"Quantiles: {', '.join(quantile_label(quantile) for quantile in args.quantiles)}"
    )
    print()

    pipeline = load_chronos_pipeline(args.model, args.device)

    csv_files = sorted(data_dir.glob("*.csv"))
    if args.max_tickers:
        csv_files = csv_files[: args.max_tickers]
    print(
        f"Found {len(csv_files)} tickers"
        f"{f' (limited to {args.max_tickers})' if args.max_tickers else ''}\n"
    )

    all_metrics = []

    for fpath in csv_files:
        ticker = fpath.stem
        df = pd.read_csv(fpath, parse_dates=["Date"])

        if len(df) < args.context_length + args.horizon:
            print(f"  [SKIP] {ticker}: not enough data for context + holdout")
            continue

        train_df = df.iloc[: -args.horizon]
        test_df = df.iloc[-args.horizon :]
        context = train_df["Log_Return"].to_numpy(dtype=np.float64)[
            -args.context_length :
        ]
        base_close = float(train_df["Close"].iloc[-1])
        actual_close = test_df["Close"].to_numpy(dtype=np.float64)
        actual_dates = test_df["Date"].values

        fcast = forecast_from_context(
            pipeline,
            context,
            horizon=args.horizon,
            num_samples=args.num_samples,
            quantiles=args.quantiles,
        )
        pred_close = lr_to_close(fcast["point_lr"], base_close)
        quantile_close = {
            quantile: lr_to_close(fcast["quantile_lr"][quantile], base_close)
            for quantile in args.quantiles
        }

        metrics = compute_metrics(pred_close, actual_close)
        metrics["ticker"] = ticker
        metrics["base_close"] = base_close
        all_metrics.append(metrics)

        detail_data = {
            "Date": actual_dates,
            "Actual_Close": actual_close,
            "Predicted_Close": pred_close,
            "Error": pred_close - actual_close,
        }
        for quantile in args.quantiles:
            detail_data[f"{quantile_label(quantile)}_Close"] = quantile_close[quantile]
        pd.DataFrame(detail_data).to_csv(detail_dir / f"{ticker}.csv", index=False)

        pct_err = metrics["mape"]
        dir_str = (
            f"{metrics['direction_accuracy']:.0%}"
            if not np.isnan(metrics["direction_accuracy"])
            else "n/a"
        )
        print(
            f"  {ticker:8s}  MAPE={pct_err:6.2f}%  RMSE={metrics['rmse']:8.4f}  DirAcc={dir_str}"
        )

    if not all_metrics:
        print("\nNo tickers evaluated.")
        return 0

    metrics_df = pd.DataFrame(all_metrics)
    cols_order = [
        "ticker",
        "rmse",
        "mae",
        "mape",
        "smape",
        "direction_accuracy",
        "base_close",
        "final_predicted",
        "final_actual",
    ]
    metrics_df = metrics_df[
        [column for column in cols_order if column in metrics_df.columns]
    ]
    metrics_df.to_csv(out_dir / "per_ticker_metrics.csv", index=False)

    numeric_cols = ["rmse", "mae", "mape", "smape", "direction_accuracy"]
    metrics_df[numeric_cols].agg(["mean", "median", "std", "min", "max"]).to_csv(
        out_dir / "aggregate_metrics.csv"
    )

    print(f"\n{'=' * 60}")
    print("AGGREGATE METRICS")
    print(f"{'=' * 60}")
    print(f"  Tickers evaluated: {len(all_metrics)}")
    print(f"  Mean RMSE:   {metrics_df['rmse'].mean():.4f}")
    print(f"  Mean MAE:    {metrics_df['mae'].mean():.4f}")
    print(f"  Mean MAPE:   {metrics_df['mape'].mean():.2f}%")
    print(f"  Mean sMAPE:  {metrics_df['smape'].mean():.2f}%")
    print(f"  Mean DirAcc: {metrics_df['direction_accuracy'].mean():.1%}")
    print(f"\nResults saved to {out_dir}/")
    return 0


def run_comparison(files: list[str]) -> int:
    """Compare two per_ticker_metrics.csv files side by side."""
    dfs = []
    for fpath in files:
        df = pd.read_csv(fpath)
        label = Path(fpath).parent.name
        dfs.append(df.set_index("ticker").add_suffix(f"_{label}"))

    merged = pd.concat(dfs, axis=1)

    mape_cols = [column for column in merged.columns if column.startswith("mape_")]
    if len(mape_cols) == 2:
        labels = [column.replace("mape_", "") for column in mape_cols]
        merged["mape_improvement"] = merged[mape_cols[0]] - merged[mape_cols[1]]

        print(f"\n{'=' * 60}")
        print(f"COMPARISON: {labels[0]} vs {labels[1]}")
        print(f"{'=' * 60}")
        print(f"  Mean MAPE ({labels[0]}):  {merged[mape_cols[0]].mean():.2f}%")
        print(f"  Mean MAPE ({labels[1]}):  {merged[mape_cols[1]].mean():.2f}%")
        improvement = merged["mape_improvement"].mean()
        better = labels[1] if improvement > 0 else labels[0]
        print(f"  Mean improvement:     {abs(improvement):.2f}pp ({better} is better)")

        dir_cols = [
            column
            for column in merged.columns
            if column.startswith("direction_accuracy_")
        ]
        if len(dir_cols) == 2:
            print(f"  Dir accuracy ({labels[0]}): {merged[dir_cols[0]].mean():.1%}")
            print(f"  Dir accuracy ({labels[1]}): {merged[dir_cols[1]].mean():.1%}")

        print(
            f"\n  Tickers where {labels[1]} wins: {(merged['mape_improvement'] > 0).sum()}/{len(merged)}"
        )
        print(
            f"  Tickers where {labels[0]} wins: {(merged['mape_improvement'] < 0).sum()}/{len(merged)}"
        )

    out_path = Path("eval") / "comparison.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path)
    print(f"\n  Full comparison saved to {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the unified command-line parser."""
    parser = argparse.ArgumentParser(
        description="Unified Chronos T5 preprocessing, forecasting, and evaluation CLI"
    )
    sub = parser.add_subparsers(dest="command")

    preprocess_parser = sub.add_parser(
        "preprocess",
        help="Keep Date + OHLCV and compute Log_Return for Chronos forecasting",
    )
    preprocess_parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing raw per-ticker CSVs",
    )
    preprocess_parser.add_argument(
        "--output-dir",
        default="data/cleaned",
        help="Where to write cleaned CSVs (default: data/cleaned)",
    )
    preprocess_parser.add_argument(
        "--max-tickers",
        type=int,
        default=None,
        help="Limit to first N tickers alphabetically (default: no limit)",
    )

    forecast_parser = sub.add_parser(
        "forecast",
        help="Run held-out backtest forecasts on cleaned CSVs",
    )
    forecast_parser.add_argument(
        "--data-dir",
        default="data/cleaned",
        help="Directory of cleaned CSVs (default: data/cleaned)",
    )
    forecast_parser.add_argument(
        "--output-dir",
        default="results/zero_shot",
        help="Where to write forecast CSVs and plots (default: results/zero_shot)",
    )
    forecast_parser.add_argument(
        "--model",
        default="amazon/chronos-t5-large",
        help="HF model ID or compatible checkpoint path (default: amazon/chronos-t5-large)",
    )
    forecast_parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device (default: cuda)",
    )
    forecast_parser.add_argument(
        "--context-length",
        type=int,
        default=512,
        help="Number of historical log returns to feed (default: 512)",
    )
    forecast_parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="Held-out forecast horizon in trading days (default: 5)",
    )
    forecast_parser.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="Monte Carlo sample paths (default: 200)",
    )
    forecast_parser.add_argument(
        "--history-days",
        dest="history_days",
        type=int,
        default=30,
        help="Days of history to show before the forecast window (default: 30)",
    )
    forecast_parser.add_argument(
        "--plot-history",
        dest="history_days",
        type=int,
        help=argparse.SUPPRESS,
    )
    forecast_parser.add_argument(
        "--quantiles",
        type=parse_quantiles,
        default=[10, 90],
        help="Comma-separated forecast percentiles to export (default: 10,90)",
    )
    forecast_parser.add_argument(
        "--plot-bands",
        type=parse_plot_bands,
        default=None,
        help="Comma-separated percentile bands to shade like 10-90,30-40 (default: lowest-highest quantile)",
    )
    forecast_parser.add_argument("--seed", type=int, default=42)
    forecast_parser.add_argument(
        "--max-tickers",
        type=int,
        default=None,
        help="Limit to first N tickers alphabetically (default: no limit)",
    )

    eval_parser = sub.add_parser(
        "eval",
        help="Run backtested evaluation on cleaned CSVs",
    )
    eval_parser.add_argument("--data-dir", default="data/cleaned")
    eval_parser.add_argument("--output-dir", default="eval/zero_shot")
    eval_parser.add_argument("--model", default="amazon/chronos-t5-large")
    eval_parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    eval_parser.add_argument("--context-length", type=int, default=512)
    eval_parser.add_argument("--horizon", type=int, default=5)
    eval_parser.add_argument("--num-samples", type=int, default=200)
    eval_parser.add_argument(
        "--quantiles",
        type=parse_quantiles,
        default=[10, 90],
        help="Comma-separated forecast percentiles to export (default: 10,90)",
    )
    eval_parser.add_argument("--seed", type=int, default=42)
    eval_parser.add_argument(
        "--max-tickers",
        type=int,
        default=None,
        help="Limit to first N tickers alphabetically (default: no limit)",
    )

    compare_parser = sub.add_parser(
        "compare",
        help="Compare two evaluation runs",
    )
    compare_parser.add_argument(
        "files",
        nargs=2,
        help="Two per_ticker_metrics.csv files to compare",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "preprocess":
        return run_preprocess(args)
    if args.command == "forecast":
        return run_forecast(args)
    if args.command == "eval":
        return run_evaluation(args)
    if args.command == "compare":
        return run_comparison(args.files)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

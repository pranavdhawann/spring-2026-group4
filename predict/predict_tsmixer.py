"""TSMixer Prediction Script (V2 — matches train_tsmixer V2)

Changes from V1:
  - _inverse_transform_to_dollars: extracted as shared helper (mirrors train script)
  - _compute_directional_accuracy: V2 logic — last forecast day only, excludes exact zeros
  - save_prediction_plots: handles per_sample_ohlcv_norm (V2 hybrid normalization)
  - predict(): reads per_sample_ohlcv_norm from config and passes it through

Loads cached preprocessed data and best checkpoint, runs inference,
computes metrics, and saves prediction plots.
"""

import json
import os
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.models.tsmixer_model import TSMixerModel
from src.utils import read_yaml, set_seed
from src.utils.metrics_utils import calculate_regression_metrics


# ---------------------------------------------------------------------------
# Helpers (mirror train_tsmixer.py V2)
# ---------------------------------------------------------------------------

def _inverse_transform_to_dollars(normalized, anchors, target_type, target_mean, target_std):
    """Convert normalized predictions/targets back to dollar prices."""
    anchors_safe = np.where(
        (anchors == 0) | np.isnan(anchors), 1.0, anchors
    )

    if target_type == 'pct_change':
        pct = normalized * target_std + target_mean
        dollars = anchors_safe[:len(normalized), np.newaxis] * (1.0 + pct / 100.0)
    else:
        dollars = normalized * anchors_safe[:len(normalized), np.newaxis]

    return dollars


def _compute_directional_accuracy(all_preds, all_actuals, target_mean, target_std):
    """V2: Compute directional accuracy on raw pct_change (un-z-scored).

    Uses the last forecast day (day 7) to determine overall direction.
    Excludes exact zeros (no change) from accuracy calculation.
    """
    preds_pct = all_preds * target_std + target_mean
    actuals_pct = all_actuals * target_std + target_mean

    pred_dir = np.sign(preds_pct[:, -1])
    true_dir = np.sign(actuals_pct[:, -1])

    valid = (true_dir != 0)
    if valid.sum() > 0:
        acc = float((pred_dir[valid] == true_dir[valid]).mean())
    else:
        acc = 0.5

    return acc


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_prediction_scatter(y_true, y_pred, save_path, max_points=5000):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if len(y_true) > max_points:
        idx = np.random.choice(len(y_true), max_points, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]

    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.4, s=10, color='blue')

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        'r--',
        label='Perfect Prediction',
    )

    plt.xlabel('Actual Price ($)', fontsize=12)
    plt.ylabel('Predicted Price ($)', fontsize=12)
    plt.title('Predicted vs Actual Prices (TSMixer V2)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Scatter plot saved: {save_path}")


def save_prediction_plots(X_test, y_test, y_pred, test_dataset, scaler, save_dir,
                          num_plots=10, per_sample_ohlcv_norm=False):
    """Save per-sample prediction plots.

    V2: handles both per-sample OHLCV norm and legacy global scaler.
    When per_sample_ohlcv_norm=True, OHLCV cols are already divided by their
    per-sample anchor so a global inverse_transform is not valid. Instead we
    anchor the input window visually using the first actual dollar price.
    """
    os.makedirs(save_dir, exist_ok=True)

    close_idx = 3
    num_plots = min(num_plots, len(y_test))

    for i in range(num_plots):
        fig, ax = plt.subplots(figsize=(12, 6))
        input_seq = X_test[i].numpy()

        # Recover close prices for the input window
        if per_sample_ohlcv_norm:
            # Cols 0-3 are already per-sample normalized (divided by anchor).
            # We can't perfectly invert without knowing the original anchor
            # for this specific test sample, so anchor visually using the
            # first actual dollar price.
            input_close = input_seq[:, close_idx]
            actual = y_test[i].numpy() if hasattr(y_test[i], 'numpy') else y_test[i]
            if len(actual) > 0 and actual[0] > 0:
                input_close = input_close * actual[0]
        else:
            input_full = scaler.inverse_transform(input_seq)
            input_close = input_full[:, close_idx]
            actual = y_test[i].numpy() if hasattr(y_test[i], 'numpy') else y_test[i]

        seq_len = len(input_close)
        predicted = y_pred[i]

        try:
            sample = test_dataset[i]
            input_dates = sample['dates']
        except Exception:
            pass

        ax.plot(
            range(seq_len),
            input_close,
            label='Input (Historical Close)',
            color='blue',
            marker='o',
            markersize=3,
        )

        ax.plot(
            range(seq_len, seq_len + 7),
            actual,
            label='Actual Price',
            color='green',
            marker='o',
            markersize=5,
            linestyle='--',
        )

        ax.plot(
            range(seq_len, seq_len + 7),
            predicted,
            label='Predicted Price',
            color='red',
            marker='^',
            markersize=5,
            linestyle='--',
        )

        ax.axvline(
            x=seq_len - 1,
            color='gray',
            linestyle=':',
            label='Forecast Start',
        )

        ax.set_title(f'Stock Price Prediction (TSMixer V2) - Sample {i + 1}', fontsize=13)
        ax.set_xlabel('Time Step', fontsize=11)
        ax.set_ylabel('Price ($)', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'prediction_plot_{i + 1}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"  {num_plots} prediction plots saved to: {save_dir}")


# ---------------------------------------------------------------------------
# Main predict function
# ---------------------------------------------------------------------------

def predict(predict_config=None):
    config = {
        'yaml_config_path': 'config/config.yaml',
        'tsmixer_config_path': 'config/tsmixer_config.yaml',
        'rand_seed': 42,
        'verbose': True,
    }
    if predict_config:
        config.update(predict_config)

    set_seed(config['rand_seed'])

    print("=" * 70)
    print("TSMIXER PREDICTION (V2)")
    print("=" * 70)

    yaml_config = read_yaml(config['yaml_config_path'])
    tsmixer_config = read_yaml(config['tsmixer_config_path'])
    config.update(yaml_config)
    config.update(tsmixer_config)

    experiment_path = config['experiment_path']
    predict_dir = os.path.join(experiment_path, 'predict')
    os.makedirs(predict_dir, exist_ok=True)

    # Load preprocessed data from cache
    print("\nLoading cached preprocessed data...")
    preprocess_cache = os.path.join(experiment_path, 'preprocessed_data.pkl')
    with open(preprocess_cache, 'rb') as f:
        pp = pickle.load(f)
    X_test = pp['X_test']
    y_test = pp['y_test']
    scaler = pp['scaler']
    test_anchors = pp['test_anchors']
    target_type = pp.get('target_type', 'pct_change')
    target_mean = pp.get('target_mean', 0.0)
    target_std = pp.get('target_std', 1.0)

    # Load dataloaders cache (for prediction plots with dates)
    print("Loading cached dataloaders...")
    dataloader_cache = os.path.join(experiment_path, 'dataloaders.pkl')
    with open(dataloader_cache, 'rb') as f:
        dl = pickle.load(f)
    test_dataset = dl['test']

    print(f"  X_test:  {X_test.shape}")
    print(f"  y_test:  {y_test.shape}")

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    batch_size = config.get('training', {}).get('batch_size', 64)
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
    )

    # Load model
    model_config = config.get('model', {})
    model = TSMixerModel(model_config).to(device)

    best_model_path = os.path.join(
        experiment_path, 'checkpoints', 'best_model.pth'
    )
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded ({total_params:,} parameters)")

    # Predict
    print("\nRunning inference...")
    all_preds = []
    all_actuals = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            actuals = y_batch.numpy()
            all_preds.append(preds)
            all_actuals.append(actuals)

    all_preds = np.concatenate(all_preds, axis=0)      # normalized space
    all_actuals = np.concatenate(all_actuals, axis=0)  # normalized space

    # Inverse-transform to dollar space
    all_preds_dollars = _inverse_transform_to_dollars(
        all_preds, test_anchors, target_type, target_mean, target_std
    )
    all_actuals_dollars = _inverse_transform_to_dollars(
        all_actuals, test_anchors, target_type, target_mean, target_std
    )

    # Metrics (in dollar space)
    metrics = calculate_regression_metrics(
        all_actuals_dollars.flatten(),
        all_preds_dollars.flatten(),
    )

    # V2: Directional accuracy on raw pct_change, last forecast day, excl. zeros
    directional_acc = _compute_directional_accuracy(
        all_preds, all_actuals, target_mean, target_std
    )
    metrics['directional_accuracy'] = directional_acc

    print("\n  Test Metrics (dollar space):")
    print(f"    MAE:   {metrics['mae']:.4f}")
    print(f"    MSE:   {metrics['mse']:.4f}")
    print(f"    RMSE:  {metrics['rmse']:.4f}")
    print(f"    MAPE:  {metrics['mape']:.2f}%")
    print(f"    SMAPE: {metrics['smape']:.2f}%")
    print(f"    Dir Acc: {directional_acc:.1%}")

    # Save results
    print("\nSaving results...")

    with open(os.path.join(predict_dir, 'predict_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    print("  predict_metrics.json saved!")

    save_prediction_scatter(
        all_actuals_dollars,
        all_preds_dollars,
        os.path.join(predict_dir, 'predictions_scatter.png'),
        config.get('evaluation', {}).get('max_scatter_points', 5000),
    )

    per_sample_ohlcv = config.get('preprocessing', {}).get('per_sample_ohlcv_norm', False)
    num_plots = config.get('evaluation', {}).get('num_plots', 10)

    save_prediction_plots(
        X_test,
        torch.tensor(all_actuals_dollars),
        all_preds_dollars,
        test_dataset,
        scaler,
        predict_dir,
        num_plots=num_plots,
        per_sample_ohlcv_norm=per_sample_ohlcv,
    )

    print("\nPREDICTION COMPLETE!")
    print(f"All results saved to: {predict_dir}")


if __name__ == "__main__":
    predict()
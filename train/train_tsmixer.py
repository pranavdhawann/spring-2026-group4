"""TSMixer Training Script (V2 — after audit fixes)

Changes from V1:
  - Added validation set (early stopping on val, not test)
  - Variance penalty in loss (anti-flatness)
  - Fixed directional accuracy (computed on raw pct_change, not z-scored)
  - Loss plot includes val loss curve

Follows the 8-stage structure:
  [1/8] Load configs
  [2/8] Load data (cached)
  [3/8] Preprocess (cached)
  [4/8] Create DataLoaders (train / val / test)
  [5/8] Create TSMixer model
  [6/8] Setup training
  [7/8] Training loop (early stop on val)
  [8/8] Evaluate on test & save
"""

import json
import os
import pickle
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from src.dataLoader.dataLoaderBaseline import getTrainTestDataLoader
from src.models.tsmixer_model import TSMixerModel
from src.preProcessing.tsmixer_preprocessing import preprocess_for_tsmixer
from src.utils import read_json_file, read_yaml, set_seed
from src.utils.metrics_utils import calculate_regression_metrics


class MSEPlusDiffLoss(nn.Module):
    """Combined MSE loss on values + MSE on temporal differences + variance penalty.

    V2 changes:
      - Added variance_weight to penalize flat predictions (predictions whose
        variance is much lower than target variance).
      - Default diff_weight reduced to 0.3 (was 0.7) to balance shape vs level.
    """

    def __init__(self, diff_weight=0.3, variance_weight=0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.diff_weight = diff_weight
        self.variance_weight = variance_weight

    def forward(self, pred, target):
        # Standard MSE on values (level accuracy)
        value_loss = self.mse(pred, target)

        # MSE on day-to-day differences (shape accuracy)
        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        diff_loss = self.mse(pred_diff, target_diff)

        total = (1.0 - self.diff_weight - self.variance_weight) * value_loss \
              + self.diff_weight * diff_loss

        # Variance penalty: encourage predictions to match target variance
        # This directly penalizes flat/constant predictions
        if self.variance_weight > 0:
            pred_var = pred.var(dim=1).mean()
            target_var = target.var(dim=1).mean().detach()
            var_loss = (pred_var - target_var).abs()
            total = total + self.variance_weight * var_loss

        return total


def save_losses_plot(train_losses, val_losses, test_losses, save_path):
    """V2: 3-curve loss plot (train + val + test)."""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o', color='blue')
    if val_losses:
        plt.plot(epochs, val_losses, label='Val Loss', marker='s', color='green')
    if test_losses:
        plt.plot(epochs, test_losses, label='Test Loss', marker='^', color='orange')
    plt.title('Epoch vs Loss (TSMixer V2)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Loss plot saved: {save_path}")


def save_prediction_plots(X_test, y_test, y_pred, test_dataset, scaler, save_dir,
                          num_plots=10, per_sample_ohlcv_norm=False):
    """Save per-sample prediction plots.

    V2: handles both per-sample OHLCV norm and legacy global scaler.
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
            # for this specific test sample, so just show the normalized prices.
            # The prediction plots still show dollar-space actuals/preds.
            input_close = input_seq[:, close_idx]
            # Scale back roughly using the range of actual prices
            actual = y_test[i].numpy() if hasattr(y_test[i], 'numpy') else y_test[i]
            # Use the first actual price to anchor the input visually
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


def save_scatter_plot(y_true, y_pred, save_path, max_points=5000):
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
    """
    # Reverse z-score to get raw pct_change
    preds_pct = all_preds * target_std + target_mean
    actuals_pct = all_actuals * target_std + target_mean

    # Overall direction: sign of pct_change at day 7 (last forecast day)
    pred_dir = np.sign(preds_pct[:, -1])
    true_dir = np.sign(actuals_pct[:, -1])

    # Exclude exact zeros (no change) from accuracy calculation
    valid = (true_dir != 0)
    if valid.sum() > 0:
        acc = float((pred_dir[valid] == true_dir[valid]).mean())
    else:
        acc = 0.5

    return acc


def train(train_config=None):
    config = {
        'yaml_config_path': 'config/config.yaml',
        'tsmixer_config_path': 'config/tsmixer_config.yaml',
        'rand_seed': 42,
        'verbose': True,
    }
    if train_config:
        config.update(train_config)

    set_seed(config['rand_seed'])

    print("=" * 70)
    print("TSMIXER TRAINING (V2)")
    print("=" * 70)
    print("\n[1/8] Loading configurations...")
    yaml_config = read_yaml(config['yaml_config_path'])
    tsmixer_config = read_yaml(config['tsmixer_config_path'])

    config.update(yaml_config)
    config.update(tsmixer_config)

    experiment_path = config['experiment_path']
    os.makedirs(experiment_path, exist_ok=True)
    os.makedirs(os.path.join(experiment_path, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(experiment_path, 'predictions'), exist_ok=True)

    print("\n[2/8] Loading data...")

    dataloader_cache = os.path.join(experiment_path, 'dataloaders.pkl')

    if os.path.exists(dataloader_cache):
        print("  Loading cached dataloaders...")
        with open(dataloader_cache, 'rb') as f:
            dl = pickle.load(f)
        train_dataset = dl['train']
        test_dataset = dl['test']
    else:
        ticker2idx = read_json_file(
            os.path.join(config['BASELINE_DATA_PATH'], config['TICKER2IDX'])
        )
        data_config = {
            'data_path': config['BASELINE_DATA_PATH'],
            'ticker2idx': ticker2idx,
            'test_train_split': 0.2,
            'random_seed': config['rand_seed'],
        }
        train_dataset, test_dataset = getTrainTestDataLoader(data_config)

        with open(dataloader_cache, 'wb') as f:
            pickle.dump({'train': train_dataset, 'test': test_dataset}, f)

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples:  {len(test_dataset)}")

    print("\n[3/8] Preprocessing data...")

    preprocess_cache = os.path.join(experiment_path, 'preprocessed_data.pkl')

    if os.path.exists(preprocess_cache):
        print("  Loading cached preprocessed data...")
        with open(preprocess_cache, 'rb') as f:
            pp = pickle.load(f)
        X_train = pp['X_train']
        y_train = pp['y_train']
        X_test = pp['X_test']
        y_test = pp['y_test']
        scaler = pp['scaler']
        train_anchors = pp['train_anchors']
        test_anchors = pp['test_anchors']
        target_type = pp.get('target_type', 'pct_change')
        target_mean = pp.get('target_mean', 0.0)
        target_std = pp.get('target_std', 1.0)
    else:
        print("  Running preprocessing...")
        X_train, y_train, X_test, y_test, scaler, train_anchors, test_anchors, target_type, target_mean, target_std = preprocess_for_tsmixer(
            train_dataset,
            test_dataset,
            config,
            verbose=config['verbose'],
        )
        with open(preprocess_cache, 'wb') as f:
            pickle.dump({
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test,
                'scaler': scaler,
                'train_anchors': train_anchors,
                'test_anchors': test_anchors,
                'target_type': target_type,
                'target_mean': target_mean,
                'target_std': target_std,
            }, f)
        print("  Preprocessed data cached!")

    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_test:  {y_test.shape}")

    print("\n[4/8] Creating DataLoaders...")

    training_config = config.get('training', {})
    batch_size = training_config.get('batch_size', 64)

    sample_fraction = training_config.get('sample_fraction', 1.0)
    if sample_fraction < 1.0:
        n_train = int(len(X_train) * sample_fraction)
        n_test = int(len(X_test) * sample_fraction)
        X_train = X_train[:n_train]
        y_train = y_train[:n_train]
        X_test = X_test[:n_test]
        y_test = y_test[:n_test]
        train_anchors = train_anchors[:n_train]
        test_anchors = test_anchors[:n_test]
        print(f"  Using {sample_fraction * 100}% of data:")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_test:  {X_test.shape}")

    # V2: Split training data into train + validation
    val_ratio = training_config.get('val_ratio', 0.15)
    val_size = int(len(X_train) * val_ratio)
    if val_size > 0:
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        val_anchors = train_anchors[-val_size:]

        X_train = X_train[:-val_size]
        y_train = y_train[:-val_size]
        train_anchors = train_anchors[:-val_size]

        val_loader = DataLoader(
            TensorDataset(X_val, y_val),
            batch_size=batch_size,
            shuffle=False,
        )
        print(f"  V2: Split off {val_size} validation samples ({val_ratio:.0%})")
    else:
        val_loader = None
        print("  WARNING: No validation set (val_ratio=0)")

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
    )

    print(f"  Train batches: {len(train_loader)}")
    if val_loader:
        print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")

    print("\n[5/8] Creating TSMixer model...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    model_config = config.get('model', {})
    model = TSMixerModel(model_config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    print("\n[6/8] Setting up training...")

    max_epochs = training_config.get('epochs', 30)
    patience = training_config.get('patience', 10)
    learning_rate = training_config.get('learning_rate', 0.001)
    gradient_clip_val = training_config.get('gradient_clip_val', 1.0)
    weight_decay = training_config.get('weight_decay', 0.0001)

    loss_fn_name = training_config.get('loss_function', 'mse_plus_diff')
    diff_loss_weight = training_config.get('diff_loss_weight', 0.3)
    variance_loss_weight = training_config.get('variance_loss_weight', 0.1)

    if loss_fn_name == 'mse_plus_diff':
        criterion = MSEPlusDiffLoss(
            diff_weight=diff_loss_weight,
            variance_weight=variance_loss_weight,
        )
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=training_config.get('lr_patience', 5),
        factor=training_config.get('lr_factor', 0.5),
    )

    print(f"  Epochs:            {max_epochs}")
    print(f"  Batch size:        {batch_size}")
    print(f"  Learning rate:     {learning_rate}")
    print(f"  Weight decay:      {weight_decay}")
    print(f"  Early stopping:    patience={patience} (on val loss)")
    print(f"  Gradient clip:     {gradient_clip_val}")
    print(f"  Loss function:     {loss_fn_name}")
    if loss_fn_name == 'mse_plus_diff':
        print(f"    diff_weight:     {diff_loss_weight}")
        print(f"    variance_weight: {variance_loss_weight}")
    print(f"  Target type:       {target_type}")
    print(f"  LR scheduler:      ReduceLROnPlateau")

    print("\n[7/8] Training model...")

    train_losses = []
    val_losses = []
    test_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(experiment_path, 'checkpoints', 'best_model.pth')

    start_time = time.time()

    for epoch in range(1, max_epochs + 1):

        # ── Train ──
        model.train()
        epoch_train_loss = 0.0

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch [{epoch:3d}/{max_epochs}] Train",
            leave=False
        )

        for X_batch, y_batch in train_bar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
            optimizer.step()

            epoch_train_loss += loss.item()
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = epoch_train_loss / len(train_loader)

        # ── Validate ── (V2: early stopping on val, not test)
        model.eval()
        avg_val_loss = 0.0

        if val_loader is not None:
            epoch_val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    predictions = model(X_batch)
                    loss = criterion(predictions, y_batch)
                    epoch_val_loss += loss.item()
            avg_val_loss = epoch_val_loss / len(val_loader)

        # ── Test (monitoring only, NOT used for early stopping) ──
        epoch_test_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                epoch_test_loss += loss.item()
        avg_test_loss = epoch_test_loss / len(test_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        test_losses.append(avg_test_loss)

        # Step LR scheduler on val loss
        monitor_loss = avg_val_loss if val_loader else avg_test_loss
        scheduler.step(monitor_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Epoch summary
        print(
            f"  Epoch [{epoch:3d}/{max_epochs}] "
            f"Train: {avg_train_loss:.4f} | "
            f"Val: {avg_val_loss:.4f} | "
            f"Test: {avg_test_loss:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # Early stopping on VALIDATION loss (V2: not test)
        if monitor_loss < best_val_loss:
            best_val_loss = monitor_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"             Best model saved! (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"             No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"\n  Early stopping triggered at epoch {epoch}!")
            print(f"  Best val loss: {best_val_loss:.4f}")
            break

    training_time = time.time() - start_time
    print(f"\n  Training completed in {training_time / 60:.2f} minutes")

    # ══════════════════════════════════════════════════════════════════
    print("\n[8/8] Evaluating best model on TEST set...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    all_preds = []
    all_actuals = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            actuals = y_batch.numpy()

            all_preds.append(preds)
            all_actuals.append(actuals)

    all_preds = np.concatenate(all_preds, axis=0)
    all_actuals = np.concatenate(all_actuals, axis=0)

    # Inverse-transform to dollar space
    all_preds_dollars = _inverse_transform_to_dollars(
        all_preds, test_anchors, target_type, target_mean, target_std
    )
    all_actuals_dollars = _inverse_transform_to_dollars(
        all_actuals, test_anchors, target_type, target_mean, target_std
    )

    metrics = calculate_regression_metrics(
        all_actuals_dollars.flatten(),
        all_preds_dollars.flatten(),
    )

    # V2: Fixed directional accuracy (on raw pct_change, not z-scored)
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

    print("\nSaving results...")

    with open(os.path.join(experiment_path, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    print("  metrics.json saved!")

    per_sample_ohlcv = config.get('preprocessing', {}).get('per_sample_ohlcv_norm', False)

    hyperparams = {
        'n_epochs': max_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'patience': patience,
        'gradient_clip_val': gradient_clip_val,
        'input_size': model_config.get('input_size', 23),
        'output_size': model_config.get('output_size', 7),
        'seq_len': model_config.get('seq_len', 14),
        'd_model': model_config.get('d_model', 64),
        'd_ff': model_config.get('d_ff', 128),
        'd_ff_time': model_config.get('d_ff_time', None),
        'num_blocks': model_config.get('num_blocks', 3),
        'dropout': model_config.get('dropout', 0.15),
        'output_strategy': model_config.get('output_strategy', 'temporal_proj'),
        'loss_function': loss_fn_name,
        'diff_loss_weight': diff_loss_weight,
        'variance_loss_weight': variance_loss_weight,
        'val_ratio': val_ratio,
        'per_sample_ohlcv_norm': per_sample_ohlcv,
        'sample_fraction': sample_fraction,
        'training_time_mins': round(training_time / 60, 2),
        'version': 'V2',
    }

    with open(os.path.join(experiment_path, 'hyperparameters.json'), 'w') as f:
        json.dump(hyperparams, f, indent=4)
    print("  hyperparameters.json saved!")

    model_summary_path = os.path.join(experiment_path, 'model_summary.txt')
    with open(model_summary_path, 'w') as f:
        f.write("TSMixer Model Summary (V2)\n")
        f.write(f"Architecture: TSMixer (Time-Series Mixer)\n")
        f.write(f"Total Parameters:     {total_params:,}\n")
        f.write(f"Trainable Parameters: {trainable_params:,}\n\n")
        f.write("Configuration:\n")
        for k, v in model_config.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nBest Val Loss: {best_val_loss:.4f}\n")
        f.write(f"Training Time:  {training_time / 60:.2f} minutes\n")
        f.write(f"\nModel Structure:\n{model}\n")
    print("  model_summary.txt saved!")

    save_losses_plot(
        train_losses,
        val_losses,
        test_losses,
        os.path.join(experiment_path, 'epoch_vs_loss.png'),
    )

    save_scatter_plot(
        all_actuals_dollars,
        all_preds_dollars,
        os.path.join(experiment_path, 'predictions_scatter.png'),
        config.get('evaluation', {}).get('max_scatter_points', 5000),
    )

    num_plots = config.get('evaluation', {}).get('num_plots', 10)
    save_prediction_plots(
        X_test,
        torch.tensor(all_actuals_dollars),
        all_preds_dollars,
        test_dataset,
        scaler,
        os.path.join(experiment_path, 'predictions'),
        num_plots=num_plots,
        per_sample_ohlcv_norm=per_sample_ohlcv,
    )

    print("TRAINING COMPLETE!")
    print(f"\nAll results saved to: {experiment_path}")
    print(f"  ├── metrics.json")
    print(f"  ├── hyperparameters.json")
    print(f"  ├── model_summary.txt")
    print(f"  ├── epoch_vs_loss.png")
    print(f"  ├── predictions_scatter.png")
    print(f"  ├── checkpoints/")
    print(f"  │   └── best_model.pth")
    print(f"  └── predictions/")
    print(f"      └── prediction_plot_1-{num_plots}.png")


if __name__ == "__main__":
    train()

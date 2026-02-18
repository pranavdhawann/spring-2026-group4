"""TCN Baseline Training Script"""

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
from src.models.tcn_model import TCNModel
from src.preProcessing.tcn_baseline_preprocessing import preprocess_for_tcn
from src.utils import read_json_file, read_yaml, set_seed
from src.utils.metrics_utils import calculate_regression_metrics


def save_losses_plot(train_losses, test_losses, save_path):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o', color='blue')
    plt.plot(epochs, test_losses, label='Test Loss', marker='o', color='orange')
    plt.title('Epoch vs Train/Test Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Loss plot saved: {save_path}")


def save_prediction_plots(X_test, y_test, y_pred, test_dataset, scaler, save_dir, num_plots=10, ):
    os.makedirs(save_dir, exist_ok=True)

    close_idx = 3
    num_plots = min(num_plots, len(y_test))

    for i in range(num_plots):

        fig, ax = plt.subplots(figsize=(12, 6))
        input_seq = X_test[i].numpy()
        input_full = scaler.inverse_transform(input_seq)
        input_close = input_full[:, close_idx]
        seq_len = len(input_close)

        try:
            sample = test_dataset[i]
            input_dates = sample['dates']
            last_date = datetime.strptime(input_dates[-1], '%Y-%m-%d')
            future_dates = [
                f"Day+{d + 1}" for d in range(7)
            ]
            x_input = input_dates
        except Exception:
            x_input = list(range(seq_len))
            future_dates = [f"Day+{d + 1}" for d in range(7)]

        actual = y_test[i].numpy()
        predicted = y_pred[i]

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

        ax.set_title(f'Stock Price Prediction - Sample {i + 1}', fontsize=13)
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
    plt.title('Predicted vs Actual Prices', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Scatter plot saved: {save_path}")


def train(train_config=None):
    config = {
        'yaml_config_path': 'config/config.yaml',
        'tcn_config_path': 'config/tcn_config.yaml',
        'rand_seed': 42,
        'verbose': True,
    }
    if train_config:
        config.update(train_config)

    set_seed(config['rand_seed'])

    print("=" * 70)
    print("TCN BASELINE TRAINING")
    print("=" * 70)
    print("\n[1/8] Loading configurations...")
    yaml_config = read_yaml(config['yaml_config_path'])
    tcn_config = read_yaml(config['tcn_config_path'])

    config.update(yaml_config)
    config.update(tcn_config)

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
    else:
        print("  Running preprocessing...")
        X_train, y_train, X_test, y_test, scaler = preprocess_for_tcn(
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
        print(f"  Using {sample_fraction * 100}% of data:")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_test:  {X_test.shape}")

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
    print(f"  Test batches:  {len(test_loader)}")

    print("\n[5/8] Creating TCN model...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    model_config = config.get('model', {})
    model = TCNModel(model_config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    print("\n[6/8] Setting up training...")

    max_epochs = training_config.get('epochs', 10)
    patience = training_config.get('patience', 5)
    learning_rate = training_config.get('learning_rate', 0.001)
    gradient_clip_val = training_config.get('gradient_clip_val', 1.0)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"  Epochs:            {max_epochs}")
    print(f"  Batch size:        {batch_size}")
    print(f"  Learning rate:     {learning_rate}")
    print(f"  Early stopping:    patience={patience}")
    print(f"  Gradient clip:     {gradient_clip_val}")

    print("\n[7/8] Training model...")

    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
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
            leave=False  # Cleans up after each epoch
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

            # Update bar with current loss
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = epoch_train_loss / len(train_loader)

        # ── Evaluate ──
        model.eval()
        epoch_test_loss = 0.0

        test_bar = tqdm(
            test_loader,
            desc=f"Epoch [{epoch:3d}/{max_epochs}] Test ",
            leave=False
        )

        with torch.no_grad():
            for X_batch, y_batch in test_bar:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                epoch_test_loss += loss.item()

                test_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_test_loss = epoch_test_loss / len(test_loader)

        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)

        # Epoch summary
        print(
            f"  Epoch [{epoch:3d}/{max_epochs}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Test Loss: {avg_test_loss:.4f}"
        )

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"             Best model saved! (Test Loss: {best_test_loss:.4f})")
        else:
            patience_counter += 1
            print(f"             No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"\n  Early stopping triggered at epoch {epoch}!")
            print(f"  Best test loss: {best_test_loss:.4f}")
            break

    training_time = time.time() - start_time
    print(f"\n  Training completed in {training_time / 60:.2f} minutes")

    print("\n[8/8] Evaluating best model...")
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

    metrics = calculate_regression_metrics(
        all_actuals.flatten(),
        all_preds.flatten(),
    )

    print("\n  Test Metrics:")
    print(f"    MAE:   {metrics['mae']:.4f}")
    print(f"    MSE:   {metrics['mse']:.4f}")
    print(f"    RMSE:  {metrics['rmse']:.4f}")
    print(f"    MAPE:  {metrics['mape']:.2f}%")
    print(f"    SMAPE: {metrics['smape']:.2f}%")

    print("\nSaving results...")

    with open(os.path.join(experiment_path, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    print("  metrics.json saved!")

    hyperparams = {
        'n_epochs': max_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'patience': patience,
        'gradient_clip_val': gradient_clip_val,
        'input_size': model_config.get('input_size', 12),
        'output_size': model_config.get('output_size', 7),
        'num_channels': model_config.get('num_channels', [64, 64, 128, 128]),
        'kernel_size': model_config.get('kernel_size', 3),
        'dropout': model_config.get('dropout', 0.1),
        'hidden_size': model_config.get('hidden_size', 64),
        'sample_fraction': sample_fraction,
        'training_time_mins': round(training_time / 60, 2),
    }

    with open(os.path.join(experiment_path, 'hyperparameters.json'), 'w') as f:
        json.dump(hyperparams, f, indent=4)
    print("  hyperparameters.json saved!")

    model_summary_path = os.path.join(experiment_path, 'model_summary.txt')
    with open(model_summary_path, 'w') as f:
        f.write("TCN Baseline Model Summary\n")
        f.write(f"Architecture: TCN (Temporal Convolutional Network)\n")
        f.write(f"Total Parameters:     {total_params:,}\n")
        f.write(f"Trainable Parameters: {trainable_params:,}\n\n")
        f.write("Configuration:\n")
        for k, v in model_config.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nBest Test Loss: {best_test_loss:.4f}\n")
        f.write(f"Training Time:  {training_time / 60:.2f} minutes\n")
        f.write(f"\nModel Structure:\n{model}\n")
    print("  model_summary.txt saved!")

    save_losses_plot(
        train_losses,
        test_losses,
        os.path.join(experiment_path, 'epoch_vs_loss.png'),
    )

    save_scatter_plot(
        all_actuals,
        all_preds,
        os.path.join(experiment_path, 'predictions_scatter.png'),
        config.get('evaluation', {}).get('max_scatter_points', 5000),
    )

    num_plots = config.get('evaluation', {}).get('num_plots', 10)
    save_prediction_plots(
        X_test,
        torch.tensor(all_actuals),
        all_preds,
        test_dataset,
        scaler,
        os.path.join(experiment_path, 'predictions'),
        num_plots=num_plots,
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

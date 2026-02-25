from __future__ import annotations
import copy
import contextlib
import importlib
import json
import math
import os
import pickle
import random
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.dataLoader import getTrainTestDataLoader
from src.preProcessing.preProcessingMultiModalBaseline import MultiModalPreProcessing
from src.models import FinBertForecastingBL
from src.utils import (
    calculate_regression_metrics,
    read_json_file,
    read_yaml,
    set_seed,
)

try:
    from torch.amp import GradScaler as AmpGradScaler

    scaler = AmpGradScaler("cuda", enabled=torch.cuda.is_available())
except Exception:
    from torch.cuda.amp import GradScaler

    scaler = GradScaler(enabled=torch.cuda.is_available())

def import_from_path(path: str):
    if ":" in path:
        module_name, attr = path.split(":", 1)
    else:
        module_name, attr = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def _resolve_class(value: Any, default_class: type) -> type:
    """Use default class if value is None; else use value if it's a class, else import from string path."""
    if value is None:
        return default_class
    if isinstance(value, type):
        return value
    return import_from_path(str(value))


def _to_tensor(v: Any, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if isinstance(v, torch.Tensor):
        return v.to(dtype=dtype, device=device)
    return torch.as_tensor(v, dtype=dtype, device=device)


def batch_to_model_inputs_finbert_mm(
    X: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    pad_id: int = 0,
    include_time_series: bool = False,
) -> Dict[str, Any]:
    first_tokens = X[0]["tokenized_news_"]
    t0 = torch.as_tensor(first_tokens)
    max_w = t0.shape[0]
    token_len = t0.shape[1]
    for s in X[1:]:
        t = torch.as_tensor(s["tokenized_news_"])
        max_w = max(max_w, t.shape[0])
        token_len = t.shape[1]  # should be constant (e.g. 512)

    def pad_2d_to_max_w(seq2d, *, pad_value):
        t = torch.as_tensor(seq2d, dtype=torch.long)
        W, L = t.shape
        if W == max_w and L == token_len:
            return t.to(device=device)
        out = torch.full((max_w, token_len), pad_value, dtype=torch.long)
        copy_w = min(W, max_w)
        copy_l = min(L, token_len)
        out[:copy_w, :copy_l] = t[:copy_w, :copy_l]
        return out.to(device=device)

    def pad_1d_to_max_w(seq1d, *, pad_value):
        t = torch.as_tensor(seq1d, dtype=torch.float32).view(-1)
        W = t.numel()
        out = torch.full((max_w,), pad_value, dtype=torch.float32)
        out[: min(W, max_w)] = t[: min(W, max_w)]
        return out.to(device=device)

    input_ids = torch.stack(
        [pad_2d_to_max_w(s["tokenized_news_"], pad_value=pad_id) for s in X]
    )

    if "attention_mask_news_" in X[0]:
        attention_mask = torch.stack(
            [pad_2d_to_max_w(s["attention_mask_news_"], pad_value=0) for s in X]
        )
        if attention_mask.max() > 1:
            attention_mask = (attention_mask != 0).long()
    else:
        attention_mask = (input_ids != pad_id).long()

    closes = torch.stack(
        [pad_1d_to_max_w(s["closes_"], pad_value=0.0) for s in X]
    )
    extra_features = torch.tensor(
        [[float(s.get("mean_closes_", 0.0)), float(s.get("std_closes_", 1.0))] for s in X],
        dtype=torch.float32,
        device=device,
    )

    out: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "closes": closes,
        "extra_features": extra_features,
    }

    if include_time_series and "time_series_features_" in X[0]:
        out["time_series_features"] = torch.stack(
            [_to_tensor(s["time_series_features_"], dtype=torch.float32, device=device) for s in X]
        )

    for k in ("ticker_id_", "sector_", "ticker_text_"):
        if k in X[0]:
            out[k.rstrip("_")] = [s.get(k) for s in X]

    return out

def build_batch_adapter(config: Mapping[str, Any]) -> Callable[[Any, torch.device], Dict[str, Any]]:
    adapter = config.get("batch_adapter", "finbert_mm")
    if callable(adapter):
        return lambda X, device: adapter(X, device=device)

    adapter = str(adapter).lower()
    if adapter in ("finbert_mm", "default"):
        include_ts = bool(config.get("include_time_series_features", False))
        pad_id = int(config.get("pad_id", 0))
        return lambda X, device: batch_to_model_inputs_finbert_mm(
            X, device=device, pad_id=pad_id, include_time_series=include_ts
        )

    raise ValueError(f"Unknown batch_adapter: {adapter}")

def build_model(config: Mapping[str, Any], device: torch.device) -> nn.Module:
    ModelCls = _resolve_class(config.get("model_class"), FinBertForecastingBL)
    model_cfg = copy.deepcopy(dict(config))
    model_cfg["device"] = device
    model = ModelCls(model_cfg)

    if config.get("load_pre_trained", False):
        model_path = os.path.join(config["experiment_path"], "best_model.pth")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(" Loaded Pre Trained Model")
    model.to(device)

    if config.get("use_torch_compile", False):
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"Warning: torch.compile failed, continuing without compile. Error: {e}")

    return model


def build_optimizer(config: Mapping[str, Any], model: nn.Module):
    weight_decay = float(config.get("weight_decay", 0.01))
    lr_cfg = config.get("lr", 1e-3)

    if isinstance(lr_cfg, Mapping) and all(hasattr(model, k) for k in ("finbert", "news_projection", "mlp_regressor")):
        return torch.optim.AdamW(
            [
                {
                    "params": model.finbert.parameters(),
                    "lr": float(lr_cfg.get("bert", 1e-5)),
                    "weight_decay": float(config.get("weight_decay_bert", 0.01)),
                },
                {"params": model.news_projection.parameters(), "lr": float(lr_cfg.get("mlp", 1e-3))},
                {"params": model.mlp_regressor.parameters(), "lr": float(lr_cfg.get("mlp", 1e-3))},
            ],
            weight_decay=weight_decay,
        )

    if isinstance(lr_cfg, Mapping):
        lr = float(lr_cfg.get("default", next(iter(lr_cfg.values()))))
    else:
        lr = float(lr_cfg)

    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )


def forward_model(model: nn.Module, inputs: Dict[str, Any], config: Mapping[str, Any]):
    mode = str(config.get("model_forward", "auto")).lower()
    if mode == "dict":
        return model(inputs)
    if mode == "kwargs":
        return model(**inputs)
    try:
        return model(inputs)
    except TypeError:
        return model(**inputs)


def get_fraction_subset(dataset, fraction=0.2, random_sample=True, seed=42):
    dataset_size = len(dataset)
    subset_size = max(1, math.ceil(dataset_size * fraction))
    if random_sample:
        rng = random.Random(seed)
        indices = rng.sample(range(dataset_size), subset_size)
    else:
        indices = list(range(subset_size))
    return Subset(dataset, indices)


def save_losses_plot(train_losses, test_losses, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Train Loss", marker="o", color="blue")
    plt.plot(test_losses, label="Test Loss", marker="o", color="orange")
    plt.title("Training and Test Loss Over Epochs", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"     Loss plot saved to {save_path}")


def save_prediction_scatter(
    y_true, y_pred, save_path="pred_vs_true.png", max_points=5000
):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    if len(y_true) > max_points:
        idx = np.random.choice(len(y_true), max_points, replace=False)
        y_true, y_pred = y_true[idx], y_pred[idx]
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Predicted vs True Scatter Plot")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"     Prediction scatter plot saved to {save_path}")


def train(train_config: Optional[Mapping[str, Any]] = None):
    train_config = dict(train_config or {})
    config: Dict[str, Any] = {
        "yaml_config_path": "config/config.yaml",
        "experiment_path": "experiments/baseline/finbert_multimodal",
        "load_pre_trained": False,
        "epochs": 10,
        "batch_size": 64,
        "max_length": 512,
        "padding": "max_length",
        "truncation": True,
        "lr": {"bert": 1e-5, "mlp": 1e-3},
        "tokenizer_path": "ProsusAI/finbert",
        "local_files_only": True,
        "sample_fraction": 1,
        "patience": 5,
        "number_of_epochs_ran": 0,
        "y_true_vs_y_pred_max_points": 5000,
        "rand_seed": 42,
        "verbose": False,
        "mlp_hidden_dims": [128, 64],
        "news_embedding_dim": 256,
        "freeze_bert": False,
        "max_window_size": 14,
        "use_torch_compile": False,
        # Generalization knobs (None = use imported defaults; or pass class/string path)
        "collator_class": None,
        "model_class": None,
        "batch_adapter": "finbert_mm",
        "include_time_series_features": False,
        "model_forward": "auto",
    }
    config.update(train_config)
    set_seed(config["rand_seed"])

    yaml_config = read_yaml(config["yaml_config_path"])
    config.update(yaml_config)
    ticker2idx = read_json_file(
        os.path.join(config["BASELINE_DATA_PATH"], config["TICKER2IDX"])
    )
    data_config = {
        "data_path": config["BASELINE_DATA_PATH"],
        "ticker2idx": ticker2idx,
        "test_train_split": 0.2,
    }
    config.update(data_config)

    os.makedirs(config["experiment_path"], exist_ok=True)

    if not os.path.exists(os.path.join(config["experiment_path"], "dataloaders.pkl")):
        train_dataloader, test_dataloader = getTrainTestDataLoader(config)
        with open(
            os.path.join(config["experiment_path"], "dataloaders.pkl"), "wb"
        ) as f:
            pickle.dump({"train": train_dataloader, "test": test_dataloader}, f)
        print("Dataloaders saved!")
    else:
        with open(
            os.path.join(config["experiment_path"], "dataloaders.pkl"), "rb"
        ) as f:
            dataloaders = pickle.load(f)
        train_dataloader = dataloaders["train"]
        test_dataloader = dataloaders["test"]
        print("Dataloaders loaded successfully!")

    train_dataloader = get_fraction_subset(
        train_dataloader, fraction=config["sample_fraction"]
    )
    test_dataloader = get_fraction_subset(
        test_dataloader, fraction=config["sample_fraction"]
    )

    CollatorCls = _resolve_class(config.get("collator_class"), MultiModalPreProcessing)
    collator = CollatorCls(config)
    train_loader = DataLoader(
        train_dataloader,
        collate_fn=collator,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    test_loader = DataLoader(
        test_dataloader,
        collate_fn=collator,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = build_model(config, device=device)
    optimizer = build_optimizer(config, model)
    criterion: nn.Module = nn.MSELoss()
    batch_adapter = build_batch_adapter(config)
    amp_ctx = (
        autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )

    best_rmse = np.inf
    epochs_without_improvement = 0
    train_losses, test_losses = [], []

    for epoch in range(config["epochs"]):
        total_loss = 0
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False)
        y_true, y_pred = [], []
        for X, y in loop:
            y = torch.as_tensor(np.array(y), dtype=torch.float32, device=device)
            inputs = batch_adapter(X, device)
            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                y_hat = forward_model(model, inputs, config)
                loss = criterion(y_hat.squeeze(), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")
            y_true.extend(y.detach().view(-1).cpu().numpy())
            y_pred.extend(y_hat.detach().view(-1).cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        train_metrics = calculate_regression_metrics(y_true, y_pred)
        print("     Train Metrics: ", train_metrics)
        print(f"Epoch {epoch + 1} | Loss: {total_loss / len(train_loader):.4f}")
        train_losses.append(total_loss / len(train_loader))
        config["train_samples"] = len(train_loader) * config["batch_size"]

        model.eval()
        test_loss = 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            test_loop = tqdm(test_loader, desc=f"Epoch {epoch + 1} | Test", leave=False)
            for X_test, y_test in test_loop:
                y_test = torch.as_tensor(np.array(y_test), dtype=torch.float32, device=device)
                inputs_test = batch_adapter(X_test, device)
                with amp_ctx:
                    y_pred_test = forward_model(model, inputs_test, config)
                    loss_test = criterion(y_pred_test.squeeze(), y_test)
                test_loss += loss_test.item()
                test_loop.set_postfix(loss=f"{loss_test.item():.4f}")
                y_true.extend(y_test.detach().view(-1).cpu().numpy())
                y_pred.extend(y_pred_test.detach().view(-1).cpu().numpy())

        config["test_samples"] = len(test_loader) * config["batch_size"]
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        test_metrics = calculate_regression_metrics(y_true, y_pred)
        print("     Test Metrics: ", test_metrics)
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        score = test_metrics.get("rmse", avg_test_loss)
        if score < best_rmse:
            best_rmse = score
            torch.save(
                model.state_dict(),
                os.path.join(config["experiment_path"], "best_model.pth"),
            )
            epochs_without_improvement = 0
            print("     Best model saved!")
        else:
            epochs_without_improvement += 1

        save_losses_plot(
            train_losses, test_losses,
            os.path.join(config["experiment_path"], "losses.png"),
        )
        save_prediction_scatter(
            y_true, y_pred,
            os.path.join(config["experiment_path"], "predictions.png"),
            config["y_true_vs_y_pred_max_points"],
        )
        with open(os.path.join(config["experiment_path"], "metrics.json"), "w") as f:
            json.dump({"train_": train_metrics, "test_": test_metrics}, f, indent=4)
        config["number_of_epochs_ran"] = epoch + 1
        with open(os.path.join(config["experiment_path"], "config.json"), "w") as f:
            json.dump(config, f, indent=4)

        if epochs_without_improvement >= config["patience"]:
            print("     Early stopping!")
            config["early_stopping_initiated"] = True
            break
        print(f"     Avg Test Loss: {avg_test_loss:.4f}\n")

if __name__ == "__main__":
    train()
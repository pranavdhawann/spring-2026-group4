"""

Script to train TabNet-based text classification model (next-day direction).

This mirrors `train_baseline_tabnet.py` but:
- Uses TabNet in classification mode (`task="classification"`).
- Builds a binary label for next-day direction from the first target horizon.

Label definition (per sample):
- y_raw: standardized targets vector of length H (from collator).
- We use only horizon 0 (next day) and define:
    label = 1 if y_raw[0] > 0  else 0

So the model learns to classify "next-day above standardized mean" vs "not".
"""
import copy
import json
import math
import os
import pickle
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.dataLoader import getTrainTestDataLoader
from src.models import tabet_forcasting
from src.preProcessing import TabNetCollator
from src.utils import read_json_file, read_yaml, set_seed

scaler = GradScaler()
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
torch.use_deterministic_algorithms(False)

def convert_X_to_tensors(X, device="cpu"):
    # sample[0]: list/array of shape (W_ids_i, L)
    input_seqs = [torch.as_tensor(sample[0], dtype=torch.long) for sample in X]
    # sample[2]: list/array of shape (W_close_i,)
    close_seqs = [torch.as_tensor(sample[2], dtype=torch.float32) for sample in X]

    # ---- Pad input_ids (tokens) ----
    max_w_ids = max(seq.shape[0] for seq in input_seqs)
    L = input_seqs[0].shape[1]  # token length (e.g. 512)

    padded_inputs = []
    for inp in input_seqs:
        w = inp.shape[0]
        if w < max_w_ids:
            pad_size = (0, 0, 0, max_w_ids - w)  # pad rows
            inp = torch.nn.functional.pad(inp, pad_size, value=0)
        padded_inputs.append(inp)

    input_ids = torch.stack(padded_inputs)  # (B, max_w_ids, L)
    attention_mask = (input_ids != 0).long()

    # ---- Pad closes separately ----
    max_w_close = max(seq.shape[0] for seq in close_seqs)

    padded_closes = []
    for clo in close_seqs:
        w = clo.shape[0]
        if w < max_w_close:
            clo = torch.nn.functional.pad(clo, (0, max_w_close - w), value=0.0)
        padded_closes.append(clo)

    closes = torch.stack(padded_closes)  # (B, max_w_close)

    extra_features = torch.tensor(
        [[sample[3], sample[4]] for sample in X], dtype=torch.float32
    )

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "closes": closes.to(device),
        "extra_features": extra_features.to(device),
    }



def get_fraction_subset(dataset, fraction=0.2, random_sample=True, seed=42):
    dataset_size = len(dataset)
    subset_size = max(1, math.ceil(dataset_size * fraction))

    if random_sample:
        rng = random.Random(seed)
        indices = rng.sample(range(dataset_size), subset_size)
    else:
        indices = list(range(subset_size))

    subset_dataset = Subset(dataset, indices)
    return subset_dataset


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


def train(train_config):
    config = {
        "yaml_config_path": "config/config.yaml",
        "experiment_path": "experiments/baseline/tabnetClassification",
        "load_pre_trained": False,
        "epochs": 20,
        "batch_size": 64,
        "max_length": 512,
        "lr": {
            "bert": 1e-5,
            "tabnet": 1e-3,
        },
        "tokenizer_path": "ProsusAI/finbert",
        "local_files_only": True,
        "sample_fraction": 0.2,
        "patience": 5,
        "number_of_epochs_ran": 0,
        "rand_seed": 42,
        "verbose": False,
        "news_embedding_dim": 256,
        "freeze_bert": False,
        "max_window_size": 14,
    }

    config.update(train_config)
    set_seed(config["rand_seed"])
    torch.use_deterministic_algorithms(False)

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
    collator = TabNetCollator(config)

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
    model_config = copy.deepcopy(config)
    model_config["device"] = device
    model_config["task"] = "classification"
    model = tabet_forcasting(model_config)

    if config["load_pre_trained"]:
        model_path = os.path.join(config["experiment_path"], "best_model.pth")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(" Loaded Pre Trained Model")

    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.finbert.parameters(),
                "lr": config["lr"]["bert"],
                "weight_decay": config.get("weight_decay_bert", 0.01),
            },
            {
                "params": model.news_projection.parameters(),
                "lr": config["lr"]["tabnet"],
            },
            {
                "params": model.tabnet.parameters(),
                "lr": config["lr"]["tabnet"],
            },
        ],
        weight_decay=config.get("weight_decay", 0.01),
    )

    best_val_acc = 0.0
    epochs_without_improvement = 0

    train_losses = []
    test_losses = []

    for epoch in range(config["epochs"]):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False)

        for X, y in loop:
            if config["verbose"]:
                print("     time taken to retrive one batch: ", time.time())

            # y: list of standardized targets (B, H). Use horizon 0 and threshold at 0.
            y_arr = np.array(y)  # (B, H)
            y_next = y_arr[:, 0]  # next-day standardized target
            labels = torch.as_tensor((y_next > 0).astype(np.float32), device=device)

            X_tensors = convert_X_to_tensors(X, device=device)

            optimizer.zero_grad()

            with autocast(device_type="cuda", dtype=torch.float16):
                logits = model(X_tensors).view(-1)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

            preds = (torch.sigmoid(logits) >= 0.5).long()
            total_correct += (preds == labels.long()).sum().item()
            total_samples += labels.numel()

        train_loss = total_loss / len(train_loader)
        train_acc = total_correct / max(1, total_samples)
        train_losses.append(train_loss)

        print(
            f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f}"
        )

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_samples = 0

        with torch.no_grad():
            val_loop = tqdm(test_loader, desc=f"Epoch {epoch + 1} | Val", leave=False)
            for X_val, y_val in val_loop:
                y_arr = np.array(y_val)
                y_next = y_arr[:, 0]
                labels = torch.as_tensor(
                    (y_next > 0).astype(np.float32), device=device
                )
                X_tensors = convert_X_to_tensors(X_val, device=device)

                with autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(X_tensors).view(-1)
                    loss = criterion(logits, labels)

                val_loss += loss.item()
                val_loop.set_postfix(loss=f"{loss.item():.4f}")

                preds = (torch.sigmoid(logits) >= 0.5).long()
                val_correct += (preds == labels.long()).sum().item()
                val_samples += labels.numel()

        avg_val_loss = val_loss / len(test_loader)
        val_acc = val_correct / max(1, val_samples)
        test_losses.append(avg_val_loss)

        print(
            f"     Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # Track best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(),
                os.path.join(config["experiment_path"], "best_model.pth"),
            )
            epochs_without_improvement = 0
            print("     Best model (by val acc) saved!")
        else:
            epochs_without_improvement += 1

        save_losses_plot(
            train_losses,
            test_losses,
            os.path.join(config["experiment_path"], "losses.png"),
        )

        metrics = {
            "train_loss": train_losses,
            "val_loss": test_losses,
            "best_val_acc": best_val_acc,
        }
        with open(os.path.join(config["experiment_path"], "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        config["number_of_epochs_ran"] = epoch
        with open(os.path.join(config["experiment_path"], "config.json"), "w") as f:
            json.dump(config, f, indent=4)

        if epochs_without_improvement >= config["patience"]:
            print("     Early stopping!")
            config["early_stopping_initiated"] = True
            break


if __name__ == "__main__":
    train_config = {"load_pre_trained": False}
    train(train_config)


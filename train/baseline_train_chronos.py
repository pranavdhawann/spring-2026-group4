"""
Chronos-2 Fine-tuning Training Script.

Usage:
    python -m train.baseline_train_chronos --config config/chronos_debug.yaml
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed as acc_set_seed
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoConfig,
    T5Config,
    AutoTokenizer,
    get_cosine_schedule_with_warmup
)
from chronos import ChronosPipeline, ChronosConfig, ChronosTokenizer

from src.dataset.chronos_dataset import ChronosFineTuningDataset
from src.preProcessing.data_preprocessing_chronos import load_all_tickers
from src.utils.metrics_utils import calculate_regression_metrics, print_metrics
from src.utils.utils import read_yaml, set_seed, read_json_file

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Chronos-2 using LoRA")
    parser.add_argument("--config", type=str, default="config/chronos_debug.yaml", help="Path to YAML config")
    # Quick overrides
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def load_config(args):
    # Base params
    config = {
        "model_name": "amazon/chronos-2-large",
        "context_length": 512,
        "prediction_length": 64,
        "stride": 1,
        "target_col": "Close",
        
        "learning_rate": 1e-4,
        "num_epochs": 10,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 2,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
        
        "logging_steps": 50,
        "save_steps": 500,
        "eval_steps": 500,
        "num_samples_for_eval": 20,
        
        "fp16": True,
        "seed": 42,
        "output_dir": "experiments/baseline/chronos_finetune"
    }
    
    # Override from YAML
    if args.config and os.path.exists(args.config):
        yaml_config = read_yaml(args.config)
        config.update(yaml_config)

    # Override from CLI
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate
    if args.num_epochs is not None:
        config["num_epochs"] = args.num_epochs
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    return config


def compute_mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    """Mean Absolute Scaled Error (MASE)."""
    # naive forecast error (in-sample absolute differences)
    if len(y_train) > 1:
        naive_error = np.mean(np.abs(y_train[1:] - y_train[:-1]))
    else:
        naive_error = 1.0 # fallback

    if naive_error == 0:
        naive_error = 1e-8
        
    mae = np.mean(np.abs(y_true - y_pred))
    return mae / naive_error


def compute_wql(y_true: np.ndarray, forecast_samples: np.ndarray, quantiles=[0.1, 0.5, 0.9]) -> float:
    """Weighted Quantile Loss (WQL)."""
    # forecast_samples: [num_samples, horizon]
    # y_true: [horizon]
    total_wql = 0.0
    denominator = np.sum(np.abs(y_true))
    if denominator == 0:
        denominator = 1e-8

    for q in quantiles:
        q_forecast = np.quantile(forecast_samples, q, axis=0) # [horizon]
        errors = y_true - q_forecast
        loss = np.maximum(q * errors, (q - 1) * errors)
        total_wql += np.sum(loss) / denominator
        
    return total_wql / len(quantiles)


def evaluate(
    accelerator,
    model: nn.Module,
    val_dataset: ChronosFineTuningDataset,
    val_raw_data: list,
    config: dict,
    tokenizer: ChronosTokenizer
):
    """
    Zero-shot generation evaluation on a fixed number of samples using the pipeline.
    Because `ChronosPipeline` wraps generation and sampling logic elegantly, 
    we construct a pipeline using our fine-tuned model and original tokenizer.
    """
    model.eval()
    logger.info("Starting evaluation...")

    metrics_list = []
    
    # We create a temporary pipeline to utilize its convenient `.predict()` method
    # Note: ChronosPipeline.from_pretrained usually expects a config string. 
    # Since we have the model in memory, we construct it manually.
    unwrapped = accelerator.unwrap_model(model)
    if not hasattr(unwrapped.config, "prediction_length"):
        unwrapped.config.prediction_length = config["prediction_length"]
        
    pipeline = ChronosPipeline(
        tokenizer=tokenizer,
        model=unwrapped
    )

    # To keep eval fast, we only evaluate the first `n` items where the last window is the prediction target.
    # Take up to 20 raw tickers for evaluation
    eval_count = min(len(val_raw_data), 20)
    
    with torch.no_grad():
        for item in tqdm(val_raw_data[:eval_count], desc="Evaluating"):
            values = item["values"]
            context_len = config["context_length"]
            pred_len = config["prediction_length"]
            
            # Ensure enough data
            if len(values) < (context_len + pred_len):
                continue
                
            # Get the exact last window split: context and actual target
            context = values[-(context_len + pred_len) : -pred_len]
            y_true = values[-pred_len:]
            
            # Predict
            context_tensor = torch.tensor(context, dtype=torch.float32, device=accelerator.device).unsqueeze(0)
            
            forecast_samples = pipeline.predict(
                context_tensor,
                prediction_length=pred_len,
                num_samples=config["num_samples_for_eval"]
            ) # shape [1, num_samples, pred_len]
            
            forecast_samples = forecast_samples.squeeze(0).cpu().numpy() # [num_samples, pred_len]
            y_pred_median = np.median(forecast_samples, axis=0)
            
            # Regression metrics
            reg_metrics = calculate_regression_metrics(y_true, y_pred_median)
            
            # MASE
            train_context = values[:-(context_len + pred_len)]
            if len(train_context) < 2:
                train_context = context # fallback
            mase = compute_mase(y_true, y_pred_median, train_context)
            
            # WQL
            wql = compute_wql(y_true, forecast_samples)
            
            item_metrics = {
                **reg_metrics,
                "mase": float(mase),
                "wql": float(wql)
            }
            metrics_list.append(item_metrics)

    # Aggregate
    if not metrics_list:
        return {"loss": 9999.0}

    agg_metrics = {}
    for k in metrics_list[0].keys():
        agg_metrics[k] = np.mean([m[k] for m in metrics_list])
        
    return agg_metrics


def main():
    args = parse_args()
    config = load_config(args)
    
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(out_dir / "training.log"),
            logging.StreamHandler(sys.stdout)
        ],
        level=logging.INFO
    )

    logger.info("=== Chronos-2 Fine-tuning ===")
    logger.info(json.dumps(config, indent=2))

    set_seed(config["seed"])
    acc_set_seed(config["seed"])

    # --- 1. Accelerator setup ---
    accelerator = Accelerator(
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        mixed_precision="fp16" if config.get("fp16") else "no"
    )

    # --- 2. Load Chronos Tokenizer ---
    logger.info(f"Loading tokenizer from {config['model_name']}...")
    # ChronosTokenizer can be initialized via its config class.
    # Chronos is based on T5 and uses the HuggingFace T5 configs augmented with chronos_config.
    hf_config = AutoConfig.from_pretrained(config["model_name"])
    # If chronos_config exists inside hf_config, extract it.
    if hasattr(hf_config, "chronos_config"):
        chronos_cfg_dict = hf_config.chronos_config
        tokenizer_class = chronos_cfg_dict.get("tokenizer_class", "MeanScaleUniformBins")
        tokenizer_kwargs = chronos_cfg_dict.get("tokenizer_kwargs", {})
        chronos_config = ChronosConfig(
            tokenizer_class=tokenizer_class,
            tokenizer_kwargs=tokenizer_kwargs,
            n_tokens=chronos_cfg_dict.get("n_tokens", 4096),
            n_special_tokens=chronos_cfg_dict.get("n_special_tokens", 2),
            pad_token_id=chronos_cfg_dict.get("pad_token_id", 0),
            eos_token_id=chronos_cfg_dict.get("eos_token_id", 1),
            use_eos_token=chronos_cfg_dict.get("use_eos_token", True),
            model_type=chronos_cfg_dict.get("model_type", "seq2seq"),
            context_length=config["context_length"],
            prediction_length=config["prediction_length"],
            # defaults for inference
            num_samples=20,
            temperature=1.0,
            top_k=50,
            top_p=1.0,
        )
    else:
        # Fallback to standard
        logger.warning(f"No chronos_config found in {config['model_name']}, using defaults.")
        chronos_config = ChronosConfig(
            tokenizer_class="MeanScaleUniformBins",
            tokenizer_kwargs={"low_limit": -15.0, "high_limit": 15.0},
            n_tokens=4096,
            n_special_tokens=2,
            pad_token_id=0,
            eos_token_id=1,
            use_eos_token=True,
            model_type="seq2seq",
            context_length=config["context_length"],
            prediction_length=config["prediction_length"],
            num_samples=20,
            temperature=1.0,
            top_k=50,
            top_p=1.0,
        )
        
    tokenizer = chronos_config.create_tokenizer()

    # --- 3. Data Loading ---
    logger.info("Loading Data...")
    manifest_path = config.get("manifest_path", "data/chronos_filtered_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    # Read data list
    all_data = load_all_tickers(manifest_path, config.get("data_dir", "data"), target_col=config["target_col"])
    
    limit = config.get("ticker_limit")
    if limit is not None:
        logger.info(f"Applying ticker limit: {limit}")
        all_data = list(all_data)[:limit]

    # Split: simple train/val
    split_idx = int(0.9 * len(all_data))
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    if not train_data:
        train_data = all_data
        
    if not val_data:
        val_data = all_data

    logger.info(f"Train tickers: {len(train_data)}, Val tickers: {len(val_data)}")

    train_dataset = ChronosFineTuningDataset(
        data_list=train_data,
        tokenizer=tokenizer,
        context_length=config["context_length"],
        prediction_length=config["prediction_length"],
        stride=config["stride"],
        mode="training"
    )
    
    val_dataset = ChronosFineTuningDataset(
        data_list=val_data,
        tokenizer=tokenizer,
        context_length=config["context_length"],
        prediction_length=config["prediction_length"],
        stride=config["prediction_length"], # non-overlapping for val 
        mode="validation"
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["per_device_train_batch_size"], 
        shuffle=True, 
        drop_last=True
    )
    # val_loader not strictly needed since we evaluate via raw series arrays

    logger.info(f"Train Dataset size: {len(train_dataset)} windows")

    # --- 4. Model Loading & PEFT Setup ---
    logger.info(f"Loading Base Model {config['model_name']}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config["model_name"],
        torch_dtype=torch.float16 if config.get("fp16") else torch.float32,
        low_cpu_mem_usage=True
    )
    
    # Store chronos config in the model for saving later
    model.config.chronos_config = chronos_config.__dict__

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["lora_target_modules"],
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # --- 5. Optimizer & Scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config["learning_rate"], 
        weight_decay=config["weight_decay"]
    )
    
    # Calculate steps
    num_update_steps_per_epoch = len(train_loader) // config["gradient_accumulation_steps"]
    if num_update_steps_per_epoch == 0:
        num_update_steps_per_epoch = 1
        
    total_steps = num_update_steps_per_epoch * config["num_epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )

    # --- 6. Accelerate Prepare ---
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    # --- 7. Training Loop ---
    logger.info(f"Starting Training: {config['num_epochs']} epochs, {total_steps} total steps")
    
    global_step = 0
    best_val_loss = float("inf")
    start_time = time.time()
    
    model.train()
    for epoch in range(1, config["num_epochs"] + 1):
        total_loss = 0.0
        
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}", disable=not accelerator.is_local_main_process)):
            with accelerator.accumulate(model):
                try:
                    # Forward
                    outputs = model(**batch)
                    loss = outputs.loss
                    
                    # Backward
                    accelerator.backward(loss)
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    total_loss += loss.item()
                    global_step += 1
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.warning(f"CUDA Out of Memory at step {global_step}. Skipping batch.")
                        optimizer.zero_grad()
                        accelerator.free_memory() # Clear cache
                        continue
                    else:
                        raise e

                # Logging
                if global_step % config["logging_steps"] == 0 and accelerator.sync_gradients:
                    avg_loss = total_loss / config["logging_steps"]
                    logger.info(f"Epoch {epoch} | Step {global_step} | Train Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
                    total_loss = 0.0
                
                # Eval
                if global_step % config["eval_steps"] == 0 and accelerator.sync_gradients:
                    val_metrics = evaluate(
                        accelerator, model, val_dataset, val_data, config, tokenizer
                    )
                    logger.info(f"Epoch {epoch} | Step {global_step} | Val Metrics: {json.dumps(val_metrics)}")
                    
                    val_loss = val_metrics.get("mase", 9999.0) # Using MASE as pseudo-loss to track best
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_dir = out_dir / "best_model"
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            logger.info(f"New best MASE: {best_val_loss:.4f}. Saving wrapper to {best_dir}")
                            # Unwrap & Save
                            unwrapped_model = accelerator.unwrap_model(model)
                            unwrapped_model.save_pretrained(best_dir)
                            
                    model.train() # Resume training
                    
                # Save
                if global_step % config["save_steps"] == 0 and accelerator.sync_gradients:
                    save_dir = out_dir / f"checkpoint-{global_step}"
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(save_dir)
                        logger.info(f"Checkpoint saved at {save_dir}")

    # --- 8. Final Eval & Save ---
    val_metrics = evaluate(accelerator, model, val_dataset, val_data, config, tokenizer)
    logger.info(f"Final Val Metrics: {json.dumps(val_metrics)}")
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = out_dir / "checkpoint-final"
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(final_dir)
        logger.info(f"Final model saved to {final_dir}")
        
    total_time_mins = (time.time() - start_time) / 60
    logger.info(f"Training completed in {total_time_mins:.2f} minutes.")
    logger.info("Done!")

if __name__ == "__main__":
    main()

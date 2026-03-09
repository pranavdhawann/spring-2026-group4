"""
src/models/chronos_t5.py

Configurable Chronos-T5 model wrapper for time-series forecasting.
Supports:
- Hugging Face model loading via ChronosPipeline
- Optional LoRA adapter merging via PEFT
- Standardized PyTorch nn.Module initialization via `config` dict
- Configurable prediction horizons and sample counts
"""

from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn

try:
    from chronos import ChronosPipeline
except ImportError:
    ChronosPipeline = None


class ChronosT5Model(nn.Module):
    """
    Generic Chronos T5 wrapper supporting inference and integration 
    with PyTorch-based pipelines.
    """

    def __init__(self, config: Dict):
        super(ChronosT5Model, self).__init__()

        # -------- Defaults --------
        self.config = {
            "model_name_or_path": "amazon/chronos-t5-large",
            "prediction_length": 7,
            "num_samples": 20,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "torch_dtype": torch.float32,
            "adapter_path": None,  # If provided, loads base model + adapter
        }

        # Override defaults
        self.config.update(config)

        self.device = self.config["device"]
        self.prediction_length = self.config["prediction_length"]
        self.num_samples = self.config["num_samples"]

        if ChronosPipeline is None:
            raise ImportError(
                "The 'chronos' package is required. Install it with: "
                "pip install chronos-forecasting"
            )

        # -------- Load model --------
        adapter_path = self.config.get("adapter_path")
        
        if adapter_path and Path(adapter_path).exists():
            # Load LoRA architecture
            import json
            from peft import PeftModel

            adapter_config_path = Path(adapter_path) / "adapter_config.json"
            if not adapter_config_path.exists():
                raise ValueError(f"adapter_config.json not found in {adapter_path}")

            with open(adapter_config_path, "r") as f:
                adapter_cfg = json.load(f)

            base_model_name = adapter_cfg.get("base_model_name_or_path")
            if not base_model_name:
                raise ValueError("adapter_config.json missing 'base_model_name_or_path'")

            print(f"Loading Base Chronos Pipeline: {base_model_name}")
            self.pipeline = ChronosPipeline.from_pretrained(
                base_model_name,
                device_map=self.device,
                torch_dtype=self.config["torch_dtype"],
            )

            print(f"Merging LoRA Adapter from: {adapter_path}")
            self.pipeline.model.model = PeftModel.from_pretrained(
                self.pipeline.model.model, str(adapter_path)
            )
            self.pipeline.model.model = self.pipeline.model.model.merge_and_unload()

        else:
            # Load standard base model
            model_path = self.config["model_name_or_path"]
            print(f"Loading Chronos Pipeline: {model_path}")
            self.pipeline = ChronosPipeline.from_pretrained(
                model_path,
                device_map=self.device,
                torch_dtype=self.config["torch_dtype"],
            )

        # The underlying model inside the pipeline
        self.model = self.pipeline.model.model

        # Trainable by default if needed (though typically frozen for zero-shot)
        for p in self.model.parameters():
            p.requires_grad = True

    # ------------------------------------------------------------------
    def forward(
        self, 
        context: torch.Tensor, 
        prediction_length: Optional[int] = None, 
        num_samples: Optional[int] = None
    ) -> torch.Tensor:
        """
        Forward pass / Inference.

        Parameters
        ----------
        context : torch.Tensor
            Historical context tensor of shape (batch_size, context_length).
        prediction_length : int, optional
            Override the default prediction length config.
        num_samples : int, optional
            Override the default number of samples drawn.

        Returns
        -------
        torch.Tensor
            Sampled forecasts of shape (batch_size, num_samples, prediction_length).
        """
        pred_length = prediction_length if prediction_length else self.prediction_length
        samples = num_samples if num_samples else self.num_samples

        context = context.to(self.device, dtype=self.config["torch_dtype"])

        # ChronosPipeline handles the tokenization and autoregressive logic
        # Output shape: (batch_size, num_samples, prediction_length)
        forecast_samples = self.pipeline.predict(
            context,
            prediction_length=pred_length,
            num_samples=samples,
        )

        return forecast_samples

    # ------------------------------------------------------------------
    def predict_median(self, context: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Convenience method to return only the median point-forecast instead of all samples.
        Returned shape: (batch_size, prediction_length)
        """
        samples = self.forward(context, **kwargs)
        # Median across the samples dimension (dim 1)
        median_forecast = torch.quantile(samples, 0.5, dim=1)
        return median_forecast


# ======================================================================
# Example Driver Code
# ======================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = {
        "model_name_or_path": "amazon/chronos-t5-large",
        "prediction_length": 7,
        "num_samples": 20,
        "device": device,
    }

    print("Initializing ChronosT5Model...")
    model = ChronosT5Model(config)
    model.eval()

    # Create dummy time series data: 2 independent series, each with 60 historical context points
    batch_size = 2
    context_length = 60
    
    # Chronos expects simple float tensors
    dummy_context = torch.rand((batch_size, context_length))
    
    print(f"\nInput context shape: {dummy_context.shape}")

    with torch.no_grad():
        print("Running forward pass (probabilistic samples)...")
        outputs = model(dummy_context)
        print(f"Output samples shape: {outputs.shape}")  
        # Expected: (2, 20, 7)
        
        print("\nRunning median point-forecast...")
        median_preds = model.predict_median(dummy_context)
        print(f"Median output shape: {median_preds.shape}")  
        # Expected: (2, 7)
    
    print("\nTest finished successfully!")

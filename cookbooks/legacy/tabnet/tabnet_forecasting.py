from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.utils import set_seed

try:
    from pytorch_tabnet.tab_model import TabNetRegressor
    _TABNET_AVAILABLE = True
except ImportError:
    TabNetRegressor = None
    _TABNET_AVAILABLE = False


class TabNetForecasting:

    def __init__(self, config: Dict):
        if not _TABNET_AVAILABLE:
            raise ImportError(
                "pytorch-tabnet is required. Install with: pip install pytorch-tabnet"
            )
        import random
        seed = config.get("seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        # No in-code defaults: config must come from tabnet_config.yaml (model + training)
        self.config = dict(config)
        self.config.setdefault("input_dim", None)
        self.model: Optional[TabNetRegressor] = None
        self._fitted = False

    def _get_model(self, input_dim: int, output_dim: int) -> TabNetRegressor:
        return TabNetRegressor(
            n_d=self.config["n_d"],
            n_a=self.config["n_a"],
            n_steps=self.config["n_steps"],
            gamma=self.config["gamma"],
            n_independent=self.config["n_independent"],
            n_shared=self.config["n_shared"],
            lambda_sparse=self.config["lambda_sparse"],
            seed=self.config["seed"],
            device_name=self.config["device_name"],
            verbose=1,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "TabNetForecasting":

        input_dim = X_train.shape[1]
        output_dim = y_train.shape[1] if y_train.ndim > 1 else 1
        if output_dim == 1 and y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        self.config["input_dim"] = input_dim
        self.config["output_dim"] = output_dim
        self.model = self._get_model(input_dim, output_dim)

        eval_set = None
        if X_val is not None and y_val is not None:
            if y_val.ndim == 1:
                y_val = y_val.reshape(-1, 1)
            eval_set = [(X_val, y_val)]

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            eval_metric=self.config["eval_metric"],
            max_epochs=self.config["max_epochs"],
            patience=self.config["patience"],
            batch_size=self.config["batch_size"],
            virtual_batch_size=self.config["virtual_batch_size"],
        )
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted or self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict(X)

    def save_model(self, path: str) -> str:
        if not self._fitted or self.model is None:
            raise RuntimeError("No model to save.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return self.model.save_model(path)

    def load_model(self, path: str) -> "TabNetForecasting":
        if self.model is None:
            self.model = TabNetRegressor()
        self.model.load_model(path)
        self._fitted = True
        return self

if __name__ == "__main__":
    np.random.seed(42)
    n, d, out = 200, 50, 7
    X = np.random.randn(n, d).astype(np.float32)
    y = np.random.randn(n, out).astype(np.float32)
    config = {"max_epochs": 5, "patience": 2}
    model = TabNetForecasting(config)
    model.fit(X[:160], y[:160], X[160:], y[160:])
    pred = model.predict(X[160:])
    print("Pred shape:", pred.shape)

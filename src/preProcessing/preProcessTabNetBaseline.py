from typing import Any, Dict, List, Tuple

from .preProcessFinBertBaseline import FinBertCollator


class TabNetCollator(FinBertCollator):
    """
    Collator for the TabNet forecasting baseline.

    This currently reuses the exact preprocessing logic of `FinBertCollator`
    so that the TabNet forecaster can plug into the same data pipeline as the
    FinBERT baseline, keeping the code simple and consistent.
    """

    def __init__(self, collator_cfg: Dict[str, Any]):
        super().__init__(collator_cfg)

    def __call__(self, batch: List[Dict[str, Any]]) -> Tuple[Any, Any]:
        return self.preprocess(batch, verbose=False)


__all__ = ["TabNetCollator"]


from __future__ import annotations

from typing import Dict, Protocol

import torch
import torch.nn as nn


class _BinaryClassifier(Protocol):
    def __call__(self, inputs: Dict) -> torch.Tensor: ...


class HardVotingEnsemble(nn.Module):
    """
    Hard-voting ensemble for binary direction classification.

    Expected:
    - Each base model returns either:
      - logits shaped (B,) or (B,1), OR
      - hard labels shaped (B,) in {0,1}

    Output:
    - hard label (B,) in {0,1} via majority vote.
    """

    def __init__(
        self,
        tcn_model: _BinaryClassifier,
        tabnet_model: _BinaryClassifier,
        finbert_model: _BinaryClassifier,
        logits_to_label: bool = True,
        threshold: float = 0.5,
    ):
        super().__init__()
        self.tcn_model = tcn_model
        self.tabnet_model = tabnet_model
        self.finbert_model = finbert_model
        self.logits_to_label = logits_to_label
        self.threshold = threshold

    @staticmethod
    def _to_labels(x: torch.Tensor, logits_to_label: bool, threshold: float) -> torch.Tensor:
        x = x.view(-1) if x.dim() > 1 else x
        if not logits_to_label:
            return x.long()
        probs = torch.sigmoid(x)
        return (probs >= threshold).long()

    def forward(self, inputs: Dict) -> torch.Tensor:
        tcn_out = self.tcn_model(inputs)
        tab_out = self.tabnet_model(inputs)
        fin_out = self.finbert_model(inputs)

        tcn_lbl = self._to_labels(tcn_out, self.logits_to_label, self.threshold)
        tab_lbl = self._to_labels(tab_out, self.logits_to_label, self.threshold)
        fin_lbl = self._to_labels(fin_out, self.logits_to_label, self.threshold)

        votes = tcn_lbl + tab_lbl + fin_lbl  # (B,)
        return (votes >= 2).long()


__all__ = ["HardVotingEnsemble"]


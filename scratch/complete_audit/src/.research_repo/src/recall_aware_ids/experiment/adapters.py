"""
Concrete defense and model adapters for the evaluation runner.

Each adapter wraps:
  - a constructed defense object
  - a frozen RF model
  - validation of defense diagnostics

All adapters call model.predict() and model.predict_proba() on the *defended* output,
never on the original input.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, Any

from recall_aware_ids.defenses.base import DefenseResult


class DefenseAdapterError(RuntimeError):
    """Raised when a defense produces invalid output."""


def _validate_result(result: DefenseResult, name: str):
    """Abort if defense damaged any protected feature or produced invalid values."""
    if result.protected_feature_modification_count > 0:
        raise DefenseAdapterError(
            f"{name}: {result.protected_feature_modification_count} protected features modified"
        )
    if result.final_bounds_violation_count > 0:
        raise DefenseAdapterError(
            f"{name}: {result.final_bounds_violation_count} cells violate training bounds after defense"
        )
    if result.final_nan_count > 0:
        raise DefenseAdapterError(f"{name}: {result.final_nan_count} NaN values after defense")
    if result.final_inf_count > 0:
        raise DefenseAdapterError(f"{name}: {result.final_inf_count} Inf values after defense")


class AFPDefenseAdapter:
    """
    Adapter for Adaptive Feature Poisoning.
    Calls AFP.defend() then model.predict / predict_proba on the defended matrix.
    """
    def __init__(self, afp, model, epsilon_base: float, alpha: float):
        self.afp = afp
        self.model = model
        self.epsilon_base = epsilon_base
        self.alpha = alpha

    @property
    def defense_name(self) -> str:
        return "afp"

    def defend_batch(
        self, X: np.ndarray, intensity: float, seed: int, attack_scenario: str, batch_id: int
    ) -> Tuple[np.ndarray, DefenseResult, np.ndarray]:
        """
        Returns (predictions, DefenseResult, positive_scores).
        intensity is used as epsilon_base; alpha is the calibrated constant.
        """
        X_def, result, _ = self.afp.defend(
            X, epsilon_base=intensity, alpha=self.alpha,
            seed=seed, attack_scenario=attack_scenario, batch_id=batch_id
        )
        _validate_result(result, "AFP")
        preds = self.model.predict(X_def)
        scores = self.model.predict_proba(X_def)[:, 1]
        return preds.astype(int), result, scores.astype(float)


class FSDefenseAdapter:
    """
    Adapter for Feature Squeezing.
    Captures effective_d from FS and exposes it via last_effective_d.
    The controller log retains continuous intensity; effective_d is a diagnostics field.
    """
    def __init__(self, fs, model):
        self.fs = fs
        self.model = model
        self._last_effective_d: int = -1

    @property
    def defense_name(self) -> str:
        return "feature_squeezing"

    @property
    def last_effective_d(self) -> int:
        return self._last_effective_d

    def defend_batch(
        self, X: np.ndarray, intensity: float, seed: int, attack_scenario: str, batch_id: int
    ) -> Tuple[np.ndarray, DefenseResult, np.ndarray]:
        """
        Returns (predictions, DefenseResult, positive_scores).
        effective_d = max(0, 6 - int(intensity)) as defined in FeatureSqueezing.defend().
        """
        X_def, result, effective_d = self.fs.defend(
            X, intensity=intensity, seed=seed, attack_scenario=attack_scenario, batch_id=batch_id
        )
        self._last_effective_d = int(effective_d)
        _validate_result(result, "FS")
        preds = self.model.predict(X_def)
        scores = self.model.predict_proba(X_def)[:, 1]
        return preds.astype(int), result, scores.astype(float)


class RSDefenseAdapter:
    """
    Adapter for Randomized Smoothing.
    Uses the 11-member ensemble semantics from RandomizedSmoothing.predict_ensemble():
      - hard prediction = majority vote
      - positive score = positive_votes / 11
      - identical standard variates when (seed, attack_scenario, batch_id) are equal
    Does NOT call model.predict_proba() directly; scores are derived from the vote tally.
    """
    def __init__(self, rs, predict_func, chunk_size: int = 100):
        self.rs = rs
        self.predict_func = predict_func  # hard-label callable (BlackBoxOracle or direct)
        self.chunk_size = chunk_size

    @property
    def defense_name(self) -> str:
        return "randomized_smoothing"

    def defend_batch(
        self, X: np.ndarray, intensity: float, seed: int, attack_scenario: str, batch_id: int
    ) -> Tuple[np.ndarray, DefenseResult, np.ndarray]:
        """
        Returns (hard_preds, DefenseResult, positive_vote_fraction).
        sigma = intensity; scores = positive_votes / ensemble_size.
        """
        preds, result, scores = self.rs.predict_ensemble(
            X, sigma=intensity, seed=seed,
            attack_scenario=attack_scenario, batch_id=batch_id,
            predict_func=self.predict_func,
            chunk_size=self.chunk_size,
            return_scores=True
        )
        _validate_result(result, "RS")
        return preds.astype(int), result, scores.astype(float)

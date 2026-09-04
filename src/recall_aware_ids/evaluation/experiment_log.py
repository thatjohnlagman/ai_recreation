"""
src/recall_aware_ids/evaluation/experiment_log.py
Batch-level experiment log builder.

Generates the structured CSV log required by Thesis Table 7 (Chapter 3).
Every run emits one row per batch with all required fields.
Log is used to produce Tables B1–B4 and the paired t-test.

Run ID format: {attack}_{defense}_{config_id}_{seed}
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from recall_aware_ids.evaluation.metrics import BatchMetrics
from recall_aware_ids.controller.recall_controller import ControllerLogEntry


@dataclass
class ExperimentLogRow:
    """
    One row of the batch-level experiment log.
    All fields map directly to Thesis Table 7 (Chapter 3).
    """
    # Run identification
    run_id: str
    seed: int
    attack_scenario: str
    defense_mechanism: str
    configuration: str        # "Base" or "C1"..."C7"
    controller_active: bool

    # Batch identity
    batch_id: int
    sample_count: int

    # Confusion matrix
    tp: int
    fp: int
    tn: int
    fn: int

    # Derived metrics
    precision: float
    recall: float
    f1_score: float
    precision_defined: bool
    recall_defined: bool

    # Controller fields (null in base config)
    controller_state: Optional[str]
    perturbation_intensity: float
    controller_config_id: Optional[str]
    sliding_window_size: Optional[int]
    Rcritical: Optional[float]
    Rmin: Optional[float]
    fast_decay: Optional[float]
    slow_decay: Optional[float]
    growth_factor: Optional[float]
    adjustment_policy: Optional[str]
    rolling_recall: Optional[float]
    window_tp: Optional[int]
    window_fn: Optional[int]
    at_intensity_min: Optional[bool]
    at_intensity_max: Optional[bool]


class ExperimentLogger:
    """
    Accumulates batch-level log entries for one run and writes to CSV/Parquet.

    Usage:
        logger = ExperimentLogger(run_id, attack, defense, config, seed)
        for batch_id, (y_true, y_pred) in enumerate(batches):
            metrics = compute_batch_metrics(y_true, y_pred, batch_id)
            ctrl_entry = controller.update(batch_id, metrics.tp, metrics.fn)  # or None
            logger.log_batch(metrics, ctrl_entry, current_intensity)
        logger.save(output_dir)
    """

    def __init__(
        self,
        run_id: str,
        attack_scenario: str,
        defense_mechanism: str,
        configuration: str,
        seed: int,
        controller_active: bool,
    ) -> None:
        self.run_id = run_id
        self.attack_scenario = attack_scenario
        self.defense_mechanism = defense_mechanism
        self.configuration = configuration
        self.seed = seed
        self.controller_active = controller_active
        self._rows: list[ExperimentLogRow] = []

    def log_batch(
        self,
        metrics: BatchMetrics,
        intensity_used: float,          # Intensity BEFORE batch t was seen (for logging)
        ctrl_entry: Optional[ControllerLogEntry] = None,
    ) -> None:
        """Append one batch row. ctrl_entry=None for base configuration runs."""
        row = ExperimentLogRow(
            run_id=self.run_id,
            seed=self.seed,
            attack_scenario=self.attack_scenario,
            defense_mechanism=self.defense_mechanism,
            configuration=self.configuration,
            controller_active=self.controller_active,
            batch_id=metrics.batch_id,
            sample_count=metrics.sample_count,
            tp=metrics.tp,
            fp=metrics.fp,
            tn=metrics.tn,
            fn=metrics.fn,
            precision=metrics.precision,
            recall=metrics.recall,
            f1_score=metrics.f1_score,
            precision_defined=metrics.precision_defined,
            recall_defined=metrics.recall_defined,
            perturbation_intensity=intensity_used,
            controller_state=ctrl_entry.controller_state if ctrl_entry else None,
            controller_config_id=ctrl_entry.config_id if ctrl_entry else None,
            sliding_window_size=ctrl_entry.window_size if ctrl_entry else None,
            Rcritical=ctrl_entry.Rcritical if ctrl_entry else None,
            Rmin=ctrl_entry.Rmin if ctrl_entry else None,
            fast_decay=ctrl_entry.fast_decay if ctrl_entry else None,
            slow_decay=ctrl_entry.slow_decay if ctrl_entry else None,
            growth_factor=ctrl_entry.growth_factor if ctrl_entry else None,
            adjustment_policy=ctrl_entry.config_id if ctrl_entry else None,
            rolling_recall=ctrl_entry.rolling_recall if ctrl_entry else None,
            window_tp=ctrl_entry.window_tp if ctrl_entry else None,
            window_fn=ctrl_entry.window_fn if ctrl_entry else None,
            at_intensity_min=ctrl_entry.at_min_bound if ctrl_entry else None,
            at_intensity_max=ctrl_entry.at_max_bound if ctrl_entry else None,
        )
        self._rows.append(row)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(r) for r in self._rows])

    def save(self, output_dir: Path, format: str = "parquet") -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        if format == "parquet":
            path = output_dir / f"{self.run_id}.parquet"
            df.to_parquet(path, index=False)
        else:
            path = output_dir / f"{self.run_id}.csv"
            df.to_csv(path, index=False)
        return path


def make_run_id(
    attack: str,
    defense: str,
    config_id: str,
    seed: int,
) -> str:
    """Generate a deterministic run ID for a given (attack, defense, config, seed) tuple."""
    a = attack.lower().replace(" ", "_").replace("-", "_")
    d = defense.lower().replace(" ", "_").replace("-", "_")
    return f"{a}__{d}__{config_id.lower()}__{seed:02d}"

"""
ExperimentRunner for Phase 10.

Execution order per batch:
  1. get_intensity(t)         — before any label access
  2. get_batch_data(t)        — cache (no labels involved)
  3. defend_batch(...)        — defense + model predict on defended output
  4. get_labels(...)          — labels accessed ONLY after prediction
  5. calculate_metrics(...)   — TP/FN computed
  6. submit_observations(...) — RA only, after batch complete; Base never receives TP/FN

Output write order:
  config.json → confusion.json → scores.json → run_summary.json → completion.json

completion.json is written last and only after all others succeed.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import json
import dataclasses
import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import shutil
from sklearn.metrics import average_precision_score

from recall_aware_ids.experiment.schemas import (
    BatchConfigLog, BatchConfusionLog, CompletionMarker,
    RunSummary, FailureRecord, _REQUIRED_PROVENANCE_KEYS, _is_hex64,
)
from recall_aware_ids.experiment.metrics import calculate_metrics

_PLACEHOLDER_STRINGS = frozenset({"dummy", "hash", "placeholder", "todo", "none", "null", "xxx"})


def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


def _validate_run_outputs(run_tmp_dir: Path, summary: RunSummary) -> None:
    """
    Reopen and validate serialized outputs before completion.json is created.
    """
    config_path = run_tmp_dir / "config.json"
    confusion_path = run_tmp_dir / "confusion.json"
    scores_path = run_tmp_dir / "scores.json"
    summary_path = run_tmp_dir / "run_summary.json"

    for p, name in [
        (config_path, "config.json"),
        (confusion_path, "confusion.json"),
        (scores_path, "scores.json"),
        (summary_path, "run_summary.json"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Run output missing before completion: {name}")

    try:
        with open(config_path, "r") as f:
            config_data = json.load(f)
    except Exception as e:
        raise ValueError(f"config.json is not valid JSON: {e}")

    try:
        with open(confusion_path, "r") as f:
            confusion_data = json.load(f)
    except Exception as e:
        raise ValueError(f"confusion.json is not valid JSON: {e}")

    try:
        with open(scores_path, "r") as f:
            scores_data = json.load(f)
    except Exception as e:
        raise ValueError(f"scores.json is not valid JSON: {e}")

    try:
        with open(summary_path, "r") as f:
            summary_dict = json.load(f)
        reconstructed_summary = RunSummary(**summary_dict)
    except Exception as e:
        raise ValueError(f"run_summary.json validation failed: {e}")

    if len(config_data) != 144:
        raise ValueError(f"config.json must have 144 records, got {len(config_data)}")
    if len(confusion_data) != 144:
        raise ValueError(f"confusion.json must have 144 records, got {len(confusion_data)}")
    if len(scores_data) != 144:
        raise ValueError(f"scores.json must have 144 records, got {len(scores_data)}")

    expected_batch_ids = list(range(144))
    if [c.get("batch_id") for c in config_data] != expected_batch_ids:
        raise ValueError("config.json batch_ids must be exactly 0..143 in order")
    if [c.get("batch_id") for c in confusion_data] != expected_batch_ids:
        raise ValueError("confusion.json batch_ids must be exactly 0..143 in order")
    if [s.get("batch_id") for s in scores_data] != expected_batch_ids:
        raise ValueError("scores.json batch_ids must be exactly 0..143 in order")

    # Reconstruct every configuration and confusion record
    for c in config_data:
        BatchConfigLog(**c)
    for cf in confusion_data:
        BatchConfusionLog(**cf)

    # Scores, predictions, labels contain 500 aligned elements per batch
    all_scores = []
    all_labels = []
    all_preds = []
    total_elements = 0

    for s_rec in scores_data:
        b_id = s_rec.get("batch_id")
        b_scores = s_rec.get("scores")
        b_preds = s_rec.get("predictions", s_rec.get("preds"))
        b_labels = s_rec.get("labels", s_rec.get("y_true"))

        if not isinstance(b_scores, list) or len(b_scores) != 500:
            raise ValueError(f"Batch {b_id} scores must be a list of 500 elements")
        if not isinstance(b_preds, list) or len(b_preds) != 500:
            raise ValueError(f"Batch {b_id} predictions must be a list of 500 elements")
        if not isinstance(b_labels, list) or len(b_labels) != 500:
            raise ValueError(f"Batch {b_id} labels must be a list of 500 elements")

        for sc in b_scores:
            if isinstance(sc, bool) or not isinstance(sc, (int, float)) or not math.isfinite(sc) or not (0.0 <= sc <= 1.0):
                raise ValueError(f"Batch {b_id} contains invalid score: {sc!r}")
        for pr in b_preds:
            if isinstance(pr, bool) or pr not in (0, 1):
                raise ValueError(f"Batch {b_id} contains non-binary prediction: {pr!r}")
        for lb in b_labels:
            if isinstance(lb, bool) or lb not in (0, 1):
                raise ValueError(f"Batch {b_id} contains non-binary label: {lb!r}")

        total_elements += 500
        all_scores.extend(b_scores)
        all_preds.extend(b_preds)
        all_labels.extend(b_labels)

    if total_elements != 72000:
        raise ValueError(f"Concatenated records total {total_elements} != 72000")

    # Reconstructed global totals match RunSummary
    recon_tp = sum(c["tp"] for c in confusion_data)
    recon_fp = sum(c["fp"] for c in confusion_data)
    recon_tn = sum(c["tn"] for c in confusion_data)
    recon_fn = sum(c["fn"] for c in confusion_data)
    recon_eligible = sum(c["eligible_count"] for c in confusion_data)
    recon_attempted = sum(c["attempted_count"] for c in confusion_data)
    recon_successful = sum(c["successful_count"] for c in confusion_data)

    if recon_tp + recon_fp + recon_tn + recon_fn != 72000:
        raise ValueError(f"Reconstructed confusion sum != 72000: {recon_tp + recon_fp + recon_tn + recon_fn}")

    if (summary.tp != recon_tp or
        summary.fp != recon_fp or
        summary.tn != recon_tn or
        summary.fn != recon_fn or
        summary.total_eligible != recon_eligible or
        summary.total_attempted != recon_attempted or
        summary.total_successful != recon_successful):
        raise ValueError("Reconstructed global totals from confusion records do not match RunSummary")

    # Global PR-AUC recomputation matches the stored value
    if len(np.unique(all_labels)) > 1:
        recomputed_pr_auc = float(average_precision_score(all_labels, all_scores))
    else:
        recomputed_pr_auc = 0.0

    if not math.isclose(summary.pr_auc_average_precision, recomputed_pr_auc, rel_tol=1e-7, abs_tol=1e-7):
        raise ValueError(
            f"Recomputed global PR-AUC {recomputed_pr_auc} does not match RunSummary {summary.pr_auc_average_precision}"
        )


def _validate_provenance(hashes: Dict[str, str]):
    if not hashes:
        raise ValueError("provenance_hashes must be non-empty")
    missing = _REQUIRED_PROVENANCE_KEYS - set(hashes.keys())
    if missing:
        raise ValueError(f"provenance_hashes missing required keys: {sorted(missing)}")
    for key, val in hashes.items():
        if not isinstance(key, str) or not isinstance(val, str):
            raise TypeError(f"provenance key/value must be strings: {key!r}: {val!r}")
        if key.lower() in _PLACEHOLDER_STRINGS or val.lower() in _PLACEHOLDER_STRINGS:
            raise ValueError(f"provenance_hashes contains placeholder: {key!r}: {val!r}")
        if not _is_hex64(val):
            raise ValueError(
                f"provenance_hashes[{key!r}] must be a 64-char lowercase hex string, got {val!r}"
            )


class AttackCacheProvider:
    def get_batch_data(self, batch_id: int) -> Dict[str, np.ndarray]:
        raise NotImplementedError


class LabelProvider:
    def __init__(self, y_measurement: np.ndarray):
        self._y = y_measurement
        self.access_log = []

    def get_labels(self, batch_indices: np.ndarray, batch_id: int) -> np.ndarray:
        self.access_log.append({"event": "label_access", "batch_id": batch_id})
        return self._y[batch_indices]


class ExperimentRunner:
    def __init__(
        self,
        resolved_batches: pd.DataFrame,
        label_provider: LabelProvider,
        attack_cache: AttackCacheProvider,
        defense_adapter,
        policy_controller,
        output_dir: Path,
        run_id: str,
        seed: int,
        attack_scenario: str,
        defense_name: str,
        config_id: str,
        provenance_hashes: Dict[str, str],
    ):
        # Basic validation
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a nonempty string")
        if ".." in run_id or "/" in run_id:
            raise ValueError("run_id cannot escape directory")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be int")

        # Validate provenance
        _validate_provenance(provenance_hashes)

        # Batch structure
        batch_counts = resolved_batches["batch_id"].value_counts()
        if len(batch_counts) != 144:
            raise ValueError(f"Expected 144 batches, got {len(batch_counts)}")
        if not all(batch_counts == 500):
            raise ValueError("Not all batches have exactly 500 records")
        if set(batch_counts.index.tolist()) != set(range(144)):
            raise ValueError("Batch IDs must be exactly 0..143")

        # Policy interface
        if not hasattr(policy_controller, "requires_feedback"):
            raise ValueError("Policy must have requires_feedback property")

        self.resolved_batches = resolved_batches
        self.label_provider = label_provider
        self.attack_cache = attack_cache
        self.defense_adapter = defense_adapter
        self.policy_controller = policy_controller
        self.output_dir = Path(output_dir)
        self.run_id = run_id
        self.seed = seed
        self.attack_scenario = attack_scenario
        self.defense_name = defense_name
        self.config_id = config_id
        self.provenance_hashes = dict(provenance_hashes)

    def _get_batch_indices(self, batch_id: int) -> np.ndarray:
        batch_df = self.resolved_batches[self.resolved_batches["batch_id"] == batch_id]
        batch_df = batch_df.sort_values("eval_position")
        return batch_df["measurement_idx"].values

    def execute_run(self):
        final_dir = self.output_dir / self.run_id
        if final_dir.exists():
            raise FileExistsError(f"Run '{self.run_id}' already exists. Will not overwrite.")

        run_tmp_dir = self.output_dir / f"{self.run_id}_tmp"
        if run_tmp_dir.exists():
            ts = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
            quarantine_dir = self.output_dir / f"{self.run_id}_stale_quarantined_{ts}"
            run_tmp_dir.rename(quarantine_dir)

        run_tmp_dir.mkdir(parents=True, exist_ok=False)

        # Accumulators for global PR-AUC
        all_y_true = []
        all_scores = []

        config_logs = []
        confusion_logs = []
        score_records = []
        
        # Accumulators for audit metrics
        total_queries = 0
        status_counts = {}
        all_l0, all_l1, all_l2, all_linf = [], [], [], []

        self.policy_controller.reset()

        asr_applicable = (self.attack_scenario not in ("Silent Probing", "SilentProbing"))

        try:
            for batch_id in range(144):
                # ---- 1. TIMING: intensity BEFORE label access ----
                decision = self.policy_controller.get_intensity(batch_id)
                intensity = decision.intensity

                # ---- 2. Cache data (no labels) ----
                cache_data = self.attack_cache.get_batch_data(batch_id)
                X_attacked = cache_data["X_attacked"]

                if X_attacked.shape != (500, 78):
                    raise ValueError(f"X_attacked shape {X_attacked.shape} != (500, 78)")
                if not np.all(np.isfinite(X_attacked)):
                    raise ValueError("X_attacked contains non-finite values")

                # Audit aggregations
                total_queries += int(np.sum(cache_data.get("queries", 0)))
                status_codes = cache_data.get("status_codes")
                if status_codes is None:
                    status_codes = ["NOT_APPLICABLE"] * 500
                for code in status_codes:
                    status_counts[code] = status_counts.get(code, 0) + 1
                    
                success_mask = cache_data["successful"]
                if np.any(success_mask):
                    if "l0" in cache_data: all_l0.append(cache_data["l0"][success_mask])
                    if "l1" in cache_data: all_l1.append(cache_data["l1"][success_mask])
                    if "l2" in cache_data: all_l2.append(cache_data["l2"][success_mask])
                    if "linf" in cache_data: all_linf.append(cache_data["linf"][success_mask])

                # ---- 3. Defense + model predict on defended output ----
                defended_preds, defense_result, defended_scores = self.defense_adapter.defend_batch(
                    X=X_attacked,
                    intensity=intensity,
                    seed=self.seed,
                    attack_scenario=self.attack_scenario,
                    batch_id=batch_id,
                )

                if len(defended_preds) != 500 or len(defended_scores) != 500:
                    raise ValueError("Defended predictions length must be 500")

                # ---- 4. Label access AFTER prediction ----
                batch_indices = self._get_batch_indices(batch_id)
                y_batch = self.label_provider.get_labels(batch_indices, batch_id)

                # Accumulate for global PR-AUC
                all_y_true.extend(y_batch.tolist())
                all_scores.extend(defended_scores.tolist())

                # ---- 5. Calculate metrics ----
                batch_metrics = calculate_metrics(
                    y_true=y_batch,
                    y_pred=defended_preds,
                    positive_scores=defended_scores,
                    eligible_mask=cache_data["eligible"],
                    attempted_mask=cache_data["attempted"],
                    successful_mask=cache_data["successful"],
                    asr_applicable=asr_applicable,
                )

                tp = batch_metrics["tp"]
                fn = batch_metrics["fn"]

                # ---- 6. RA feedback (after batch completion) or Base log ----
                fs_effective_d = None
                if hasattr(self.defense_adapter, "last_effective_d"):
                    fs_effective_d = self.defense_adapter.last_effective_d

                if self.policy_controller.requires_feedback:
                    # RA: submit observations
                    policy_update = self.policy_controller.submit_observations(batch_id, tp, fn)
                    c_log = BatchConfigLog(
                        run_id=self.run_id,
                        batch_id=batch_id,
                        config_id=self.config_id,
                        intensity=float(intensity),
                        state=policy_update.state,
                        multiplier=float(policy_update.multiplier),
                        rolling_recall=float(policy_update.rolling_recall),
                        hit_min_bound=bool(policy_update.hit_min_bound),
                        hit_max_bound=bool(policy_update.hit_max_bound),
                        zero_denominator=bool(policy_update.zero_denominator),
                        tp=int(policy_update.tp),
                        fn=int(policy_update.fn),
                        window_start_batch_id=(
                            int(policy_update.window_start_batch_id)
                            if policy_update.window_start_batch_id is not None else None
                        ),
                        window_end_batch_id=(
                            int(policy_update.window_end_batch_id)
                            if policy_update.window_end_batch_id is not None else None
                        ),
                        configured_window_size=int(policy_update.configured_window_size),
                        window_batch_count=int(policy_update.window_batch_count),
                        window_tp_sum=int(policy_update.window_tp_sum),
                        window_fn_sum=int(policy_update.window_fn_sum),
                        unclipped_next_intensity=float(policy_update.unclipped_next_intensity),
                        clipped_next_intensity=float(policy_update.clipped_next_intensity),
                        fs_effective_d=int(fs_effective_d) if fs_effective_d is not None else None,
                    )
                else:
                    # Base: never receives TP/FN; all RA-only fields must be None
                    c_log = BatchConfigLog(
                        run_id=self.run_id,
                        batch_id=batch_id,
                        config_id=self.config_id,
                        intensity=float(intensity),
                        state="Base",
                        multiplier=1.0,
                        rolling_recall=0.0,
                        hit_min_bound=False,
                        hit_max_bound=False,
                        zero_denominator=False,
                        # All RA-only fields explicitly None
                        tp=None,
                        fn=None,
                        window_start_batch_id=None,
                        window_end_batch_id=None,
                        configured_window_size=None,
                        window_batch_count=None,
                        window_tp_sum=None,
                        window_fn_sum=None,
                        unclipped_next_intensity=None,
                        clipped_next_intensity=None,
                        fs_effective_d=int(fs_effective_d) if fs_effective_d is not None else None,
                    )

                cf_log = BatchConfusionLog(
                    run_id=self.run_id,
                    batch_id=batch_id,
                    tp=int(tp),
                    fp=int(batch_metrics["fp"]),
                    tn=int(batch_metrics["tn"]),
                    fn=int(fn),
                    accuracy=float(batch_metrics["accuracy"]),
                    recall=float(batch_metrics["recall"]),
                    precision=float(batch_metrics["precision"]),
                    f1=float(batch_metrics["f1"]),
                    balanced_accuracy=float(batch_metrics["balanced_accuracy"]),
                    pr_auc_average_precision=float(batch_metrics["pr_auc_average_precision"]),
                    eligible_count=int(batch_metrics["eligible_count"]),
                    attempted_count=int(batch_metrics["attempted_count"]),
                    successful_count=int(batch_metrics["successful_count"]),
                    asr_applicable=bool(batch_metrics["asr_applicable"]),
                    asr=batch_metrics["asr"],
                )

                config_logs.append(dataclasses.asdict(c_log))
                confusion_logs.append(dataclasses.asdict(cf_log))
                score_records.append({
                    "batch_id": batch_id,
                    "y_true": y_batch.tolist(),
                    "labels": y_batch.tolist(),
                    "scores": defended_scores.tolist(),
                    "preds": defended_preds.tolist(),
                    "predictions": defended_preds.tolist(),
                    "defense_final_invalid_count": (
                        defense_result.final_nan_count + defense_result.final_inf_count
                        + defense_result.final_bounds_violation_count
                    ),
                    "protected_feature_modification_count": defense_result.protected_feature_modification_count,
                    "projected_cell_count": defense_result.projected_cell_count,
                })

            # ---- Write outputs in strict order ----
            # 1. config.json
            with open(run_tmp_dir / "config.json", "w") as f:
                json.dump(config_logs, f, indent=2)

            # 2. confusion.json
            with open(run_tmp_dir / "confusion.json", "w") as f:
                json.dump(confusion_logs, f, indent=2)

            # 3. scores.json
            with open(run_tmp_dir / "scores.json", "w") as f:
                json.dump(score_records, f)

            # 4. run_summary.json — global PR-AUC from concatenated scores
            agg_y = np.array(all_y_true, dtype=int)
            agg_scores = np.array(all_scores, dtype=float)

            total_tp = sum(r["tp"] for r in (dataclasses.asdict(BatchConfusionLog(**c))
                                              for c in (json.loads(json.dumps(cf))
                                                        for cf in confusion_logs)))
            # Simpler aggregation from confusion_logs directly
            agg_tp = sum(c["tp"] for c in confusion_logs)
            agg_fp = sum(c["fp"] for c in confusion_logs)
            agg_tn = sum(c["tn"] for c in confusion_logs)
            agg_fn = sum(c["fn"] for c in confusion_logs)
            agg_eligible = sum(c["eligible_count"] for c in confusion_logs)
            agg_attempted = sum(c["attempted_count"] for c in confusion_logs)
            agg_successful = sum(c["successful_count"] for c in confusion_logs)

            denom = agg_tp + agg_fp + agg_tn + agg_fn
            global_accuracy = (agg_tp + agg_tn) / denom if denom > 0 else 0.0
            global_recall = agg_tp / (agg_tp + agg_fn) if (agg_tp + agg_fn) > 0 else 0.0
            global_precision = agg_tp / (agg_tp + agg_fp) if (agg_tp + agg_fp) > 0 else 0.0
            if (global_precision + global_recall) > 0:
                global_f1 = 2 * global_precision * global_recall / (global_precision + global_recall)
            else:
                global_f1 = 0.0
            specificity = agg_tn / (agg_tn + agg_fp) if (agg_tn + agg_fp) > 0 else 0.0
            global_balanced_acc = (global_recall + specificity) / 2.0

            # Global PR-AUC from concatenated row-level labels and scores
            if len(np.unique(agg_y)) > 1:
                global_pr_auc = float(average_precision_score(agg_y, agg_scores))
            else:
                global_pr_auc = 0.0

            if asr_applicable:
                global_asr = float(agg_successful / agg_attempted) if agg_attempted > 0 else 0.0
            else:
                global_asr = None

            def summarize_mag(arr_list):
                if not arr_list:
                    return {"mean": 0.0, "max": 0.0, "min": 0.0, "std": 0.0}
                arr = np.concatenate(arr_list)
                if len(arr) == 0:
                    return {"mean": 0.0, "max": 0.0, "min": 0.0, "std": 0.0}
                return {
                    "mean": float(np.mean(arr)),
                    "max": float(np.max(arr)),
                    "min": float(np.min(arr)),
                    "std": float(np.std(arr)),
                }
                
            cache_identity = getattr(self.attack_cache, "cache_identity", None) or dict(self.provenance_hashes)

            summary = RunSummary(
                run_id=self.run_id,
                seed=self.seed,
                attack_scenario=self.attack_scenario,
                defense=self.defense_name,
                config_id=self.config_id,
                total_batches=144,
                tp=agg_tp,
                fp=agg_fp,
                tn=agg_tn,
                fn=agg_fn,
                accuracy=float(global_accuracy),
                recall=float(global_recall),
                precision=float(global_precision),
                f1=float(global_f1),
                balanced_accuracy=float(global_balanced_acc),
                pr_auc_average_precision=float(global_pr_auc),
                total_eligible=agg_eligible,
                total_attempted=agg_attempted,
                total_successful=agg_successful,
                total_queries=total_queries,
                cache_identity=cache_identity,
                status_code_counts=status_counts,
                l0_summary=summarize_mag(all_l0),
                l1_summary=summarize_mag(all_l1),
                l2_summary=summarize_mag(all_l2),
                linf_summary=summarize_mag(all_linf),
                global_asr=global_asr,
                completed_successfully=True,
            )

            with open(run_tmp_dir / "run_summary.json", "w") as f:
                json.dump(dataclasses.asdict(summary), f, indent=2)

            # Reopen and validate serialized outputs before writing completion marker
            _validate_run_outputs(run_tmp_dir, summary)

            # 5. completion.json — written LAST
            marker = CompletionMarker(
                run_id=self.run_id,
                timestamp=_now_iso(),
                provenance_hashes=self.provenance_hashes,
            )
            with open(run_tmp_dir / "completion.json", "w") as f:
                json.dump(dataclasses.asdict(marker), f, indent=2)

            # Atomic rename to final dir
            run_tmp_dir.rename(final_dir)

        except Exception as e:
            ts = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
            quarantine_dir = self.output_dir / f"{self.run_id}_failed_quarantined_{ts}"
            if run_tmp_dir.exists():
                run_tmp_dir.rename(quarantine_dir)
                fail_record = FailureRecord(
                    run_id=self.run_id,
                    timestamp=_now_iso(),
                    error_type=type(e).__name__,
                    error_message=str(e),
                    failed_at_batch=locals().get("batch_id", None),
                )
                with open(quarantine_dir / "quarantined_failure.json", "w") as f:
                    json.dump(dataclasses.asdict(fail_record), f)
            raise

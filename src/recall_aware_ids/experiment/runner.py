import numpy as np
import pandas as pd
import json
import dataclasses
from pathlib import Path
from typing import Dict, Any, Optional
import shutil
import datetime

from recall_aware_ids.experiment.schemas import BatchConfigLog, BatchConfusionLog, CompletionMarker, FailureRecord
from recall_aware_ids.experiment.metrics import calculate_metrics

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
    def __init__(self,
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
                 config_id: str):
        
        # Validation
        if ".." in run_id or "/" in run_id:
            raise ValueError("run_id cannot escape directory")
        if not isinstance(seed, int):
            raise TypeError("seed must be int")
            
        # Batches validation
        batch_counts = resolved_batches['batch_id'].value_counts()
        if len(batch_counts) != 144:
            raise ValueError(f"Expected 144 batches, got {len(batch_counts)}")
        if not all(batch_counts == 500):
            raise ValueError("Not all batches have exactly 500 records")
        if set(batch_counts.index) != set(range(144)):
            raise ValueError("Batch IDs must be exactly 0..143")
        
        self.resolved_batches = resolved_batches
        self.label_provider = label_provider
        self.attack_cache = attack_cache
        self.defense_adapter = defense_adapter
        self.policy_controller = policy_controller
        
        self.output_dir = output_dir
        self.run_id = run_id
        self.seed = seed
        self.attack_scenario = attack_scenario
        self.defense_name = defense_name
        self.config_id = config_id
        
        # Policy interface validation
        if hasattr(self.policy_controller, 'requires_feedback') is False:
            raise ValueError("Policy must have requires_feedback property")
        
    def _get_batch_indices(self, batch_id: int):
        batch_df = self.resolved_batches[self.resolved_batches['batch_id'] == batch_id]
        batch_df = batch_df.sort_values('eval_position')
        return batch_df['measurement_idx'].values

    def execute_run(self):
        final_dir = self.output_dir / self.run_id
        if final_dir.exists():
            raise FileExistsError(f"Run {self.run_id} already exists. Will not overwrite.")
            
        run_tmp_dir = self.output_dir / f"{self.run_id}_tmp"
        if run_tmp_dir.exists():
            quarantine_dir = self.output_dir / f"{self.run_id}_stale_quarantined_{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            run_tmp_dir.rename(quarantine_dir)
            
        run_tmp_dir.mkdir(parents=True, exist_ok=False)
        
        config_log_path = run_tmp_dir / "config.json"
        confusion_log_path = run_tmp_dir / "confusion.json"
        
        config_logs = []
        confusion_logs = []
        
        self.policy_controller.reset()
        
        try:
            for batch_id in range(144):
                # 1. TIMING: Request intensity used [t] BEFORE fetching labels
                decision = self.policy_controller.get_intensity(batch_id)
                intensity = decision.intensity
                
                # 2. Obtain canonical attacked feature rows (no labels provided to attack)
                cache_data = self.attack_cache.get_batch_data(batch_id)
                X_attacked = cache_data['X_attacked']
                
                if len(X_attacked) != 500:
                    raise ValueError(f"X_attacked must have 500 rows, got {len(X_attacked)}")
                if X_attacked.shape[1] != 78:
                    raise ValueError(f"X_attacked must have 78 columns, got {X_attacked.shape[1]}")
                if not np.all(np.isfinite(X_attacked)):
                    raise ValueError("X_attacked must be finite")
                
                # 3. Apply defense and predict
                defended_preds, defended_result, defended_scores = self.defense_adapter.defend_batch(
                    X=X_attacked, 
                    intensity=intensity, 
                    seed=self.seed, 
                    attack_scenario=self.attack_scenario, 
                    batch_id=batch_id
                )
                
                if len(defended_preds) != 500 or len(defended_scores) != 500:
                    raise ValueError("Predictions and scores must have length 500")
                
                # 4. NOW retrieve labels in the evaluation-only metric layer
                batch_indices = self._get_batch_indices(batch_id)
                y_batch = self.label_provider.get_labels(batch_indices, batch_id)
                
                # 5. Calculate TP/FN and metrics
                batch_metrics = calculate_metrics(
                    y_true=y_batch,
                    y_pred=defended_preds,
                    positive_scores=defended_scores,
                    eligible_mask=cache_data['eligible'],
                    attempted_mask=cache_data['attempted'],
                    successful_mask=cache_data['successful'],
                    asr_applicable=(self.attack_scenario != "Silent Probing")
                )
                
                tp = batch_metrics["tp"]
                fn = batch_metrics["fn"]
                
                # 6. Update RA for batch t+1 (if requires feedback)
                if self.policy_controller.requires_feedback:
                    policy_update = self.policy_controller.submit_observations(batch_id, tp, fn)
                    c_log = BatchConfigLog(
                        run_id=self.run_id,
                        batch_id=batch_id,
                        config_id=self.config_id,
                        intensity=intensity,
                        state=policy_update.state,
                        multiplier=policy_update.multiplier,
                        rolling_recall=policy_update.rolling_recall,
                        hit_min_bound=policy_update.hit_min_bound,
                        hit_max_bound=policy_update.hit_max_bound,
                        zero_denominator=policy_update.zero_denominator,
                        tp=policy_update.tp,
                        fn=policy_update.fn,
                        window_start_batch_id=policy_update.window_start_batch_id,
                        window_end_batch_id=policy_update.window_end_batch_id,
                        configured_window_size=policy_update.configured_window_size,
                        window_batch_count=policy_update.window_batch_count,
                        window_tp_sum=policy_update.window_tp_sum,
                        window_fn_sum=policy_update.window_fn_sum,
                        unclipped_next_intensity=policy_update.unclipped_next_intensity,
                        clipped_next_intensity=policy_update.clipped_next_intensity,
                        fs_effective_d=None # Can be extracted from defended_result if needed
                    )
                else:
                    # Base policy
                    c_log = BatchConfigLog(
                        run_id=self.run_id,
                        batch_id=batch_id,
                        config_id=self.config_id,
                        intensity=intensity,
                        state="Base",
                        multiplier=1.0,
                        rolling_recall=0.0,
                        hit_min_bound=False,
                        hit_max_bound=False,
                        zero_denominator=False
                    )
                
                cf_log = BatchConfusionLog(
                    run_id=self.run_id,
                    batch_id=batch_id,
                    tp=tp,
                    fp=batch_metrics["fp"],
                    tn=batch_metrics["tn"],
                    fn=fn,
                    accuracy=batch_metrics["accuracy"],
                    recall=batch_metrics["recall"],
                    precision=batch_metrics["precision"],
                    f1=batch_metrics["f1"],
                    balanced_accuracy=batch_metrics["balanced_accuracy"],
                    pr_auc_average_precision=batch_metrics["pr_auc_average_precision"],
                    eligible_count=batch_metrics["eligible_count"],
                    attempted_count=batch_metrics["attempted_count"],
                    successful_count=batch_metrics["successful_count"],
                    asr_applicable=batch_metrics["asr_applicable"],
                    asr=batch_metrics["asr"]
                )
                
                config_logs.append(dataclasses.asdict(c_log))
                confusion_logs.append(dataclasses.asdict(cf_log))
                
            # Write outputs inside tmp dir
            with open(config_log_path, 'w') as f:
                json.dump(config_logs, f, indent=2)
                
            with open(confusion_log_path, 'w') as f:
                json.dump(confusion_logs, f, indent=2)
                
            # Write completion marker
            marker = CompletionMarker(run_id=self.run_id, timestamp=datetime.datetime.utcnow().isoformat(), provenance_hashes={"dummy": "hash"})
            with open(run_tmp_dir / "completion.json", 'w') as f:
                json.dump(dataclasses.asdict(marker), f)
                
            # Atomically rename
            run_tmp_dir.rename(final_dir)
            
        except Exception as e:
            # On failure, quarantine the temp directory and write failure marker
            quarantine_dir = self.output_dir / f"{self.run_id}_failed_quarantined_{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            run_tmp_dir.rename(quarantine_dir)
            
            fail_record = FailureRecord(
                run_id=self.run_id,
                timestamp=datetime.datetime.utcnow().isoformat(),
                error_type=type(e).__name__,
                error_message=str(e),
                failed_at_batch=locals().get('batch_id', None)
            )
            with open(quarantine_dir / "quarantined_failure.json", 'w') as f:
                json.dump(dataclasses.asdict(fail_record), f)
            raise e

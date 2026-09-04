import numpy as np
import json
import dataclasses
from pathlib import Path
from recall_aware_ids.experiment.schemas import BatchConfigLog, BatchConfusionLog
from recall_aware_ids.experiment.metrics import calculate_metrics

class ExperimentRunner:
    def __init__(self,
                 X_eval,
                 y_eval,
                 roles_manifest,
                 batches_manifest,
                 model_adapter,
                 attack_generator,
                 defense_adapter,
                 policy_controller,
                 output_dir: Path,
                 run_id: str,
                 seed: int,
                 attack_scenario: str,
                 defense_name: str,
                 config_id: str):
        
        self.X_eval = X_eval
        self.y_eval = y_eval
        self.roles_manifest = roles_manifest
        self.batches_manifest = batches_manifest
        self.model_adapter = model_adapter
        self.attack_generator = attack_generator
        self.defense_adapter = defense_adapter
        self.policy_controller = policy_controller
        
        self.output_dir = output_dir
        self.run_id = run_id
        self.seed = seed
        self.attack_scenario = attack_scenario
        self.defense_name = defense_name
        self.config_id = config_id
        
        # Determine 144 batch iterations by inspecting batches_manifest
        # But we do not load anything in __init__ just save references

    def _get_batch_data(self, batch_id: int):
        # We assume batches_manifest provides exactly 500 rows for this batch_id
        # and that those rows are guaranteed to be measurement_pool only.
        batch_indices = self.batches_manifest[self.batches_manifest['batch_id'] == batch_id].index
        
        # Explicitly enforce batch_size 500
        if len(batch_indices) != 500:
            raise ValueError(f"Batch {batch_id} must have exactly 500 records. Found {len(batch_indices)}")
            
        X_batch = self.X_eval.iloc[batch_indices].values.astype(np.float32)
        y_batch = self.y_eval.iloc[batch_indices].values.astype(int)
        
        # Get clean predictions
        clean_preds, clean_scores = self.model_adapter.predict(X_batch)
        
        return X_batch, y_batch, clean_preds, clean_scores, batch_indices

    def execute_run(self):
        self.policy_controller.reset()
        
        config_log_path = self.output_dir / f"{self.run_id}_config.json.tmp"
        confusion_log_path = self.output_dir / f"{self.run_id}_confusion.json.tmp"
        
        config_logs = []
        confusion_logs = []
        
        # Execute 144 batches exactly
        for batch_id in range(144):
            # 1. Fetch labels and clean data
            X_batch, y_batch, clean_preds, clean_scores, batch_indices = self._get_batch_data(batch_id)
            
            # 2. Timing Contract: Get intensity BEFORE attack or labels are processed by policy
            decision = self.policy_controller.get_intensity(batch_id)
            intensity = decision.intensity
            
            # 3. Eligibility Mask
            # "Eligible: true label is Attack and the clean frozen RF initially predicts Attack."
            eligible_mask = (y_batch == 1) & (clean_preds == 1)
            
            # 4. Attack generation (injected)
            # The attack generator applies its specific semantics:
            # Silent probing -> returns original X_batch.
            # Surrogate/Boundary -> modifies ONLY eligible records if attempted.
            X_attacked, attempted_mask, successful_mask = self.attack_generator.attack_batch(
                X_batch=X_batch,
                y_batch=y_batch, # Provided only for selection layer, not generator internals
                clean_preds=clean_preds,
                eligible_mask=eligible_mask,
                batch_id=batch_id,
                seed=self.seed
            )
            
            # 5. Defense Application
            defended_preds, defended_scores, _ = self.defense_adapter.defend_batch(
                X_attacked, 
                intensity, 
                self.seed, 
                self.attack_scenario, 
                batch_id
            )
            
            # 6. Metrics Calculation
            batch_metrics = calculate_metrics(
                y_true=y_batch,
                y_pred=defended_preds,
                positive_scores=defended_scores,
                eligible_mask=eligible_mask,
                attempted_mask=attempted_mask,
                successful_mask=successful_mask
            )
            
            tp = batch_metrics["tp"]
            fn = batch_metrics["fn"]
            
            # 7. Submit completed batch observations
            policy_update = self.policy_controller.submit_observations(batch_id, tp, fn)
            
            # 8. Record logs
            c_log = BatchConfigLog(
                run_id=self.run_id,
                batch_id=batch_id,
                config_id=self.config_id,
                intensity=intensity,
                state=policy_update.state,
                rolling_recall=policy_update.rolling_recall,
                hit_min_bound=policy_update.hit_min_bound,
                hit_max_bound=policy_update.hit_max_bound
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
                asr=batch_metrics["asr"]
            )
            
            config_logs.append(dataclasses.asdict(c_log))
            confusion_logs.append(dataclasses.asdict(cf_log))
            
        # Write outputs atomically
        with open(config_log_path, 'w') as f:
            json.dump(config_logs, f, indent=2)
            
        with open(confusion_log_path, 'w') as f:
            json.dump(confusion_logs, f, indent=2)
            
        # Rename to strip .tmp extension
        config_log_path.rename(self.output_dir / f"{self.run_id}_config.json")
        confusion_log_path.rename(self.output_dir / f"{self.run_id}_confusion.json")

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import json

@dataclass(frozen=True)
class AttackCacheManifest:
    attack_scenario: str
    effective_seed: int
    attacks_yaml_hash: str
    attack_script_hashes: Dict[str, str]
    frozen_rf_hash: str
    evaluation_roles_hash: str
    evaluation_batches_hash: str
    X_eval_hash: str
    metadata_eval_hash: str
    measurement_identity_hash: str
    crafting_identity_hash: str
    feature_names_hash: str
    feature_mask_hash: str
    scaler_hash: str
    training_bounds_hash: str
    query_budgets: Dict[str, Any]
    attack_parameters: Dict[str, Any]
    schema_version: str
    row_count: int
    output_sha256: str

@dataclass(frozen=True)
class AttackedSampleStatus:
    eval_position: int
    eligible: bool
    attempted: bool
    successful: bool
    status_code: str
    queries_used: int
    l0: float
    l1: float
    l2: float
    linf: float

@dataclass(frozen=True)
class BatchConfigLog:
    run_id: str
    batch_id: int
    config_id: str
    intensity: float
    state: str
    multiplier: float
    rolling_recall: float
    hit_min_bound: bool
    hit_max_bound: bool
    zero_denominator: bool
    tp: Optional[int] = None
    fn: Optional[int] = None
    window_start_batch_id: Optional[int] = None
    window_end_batch_id: Optional[int] = None
    configured_window_size: Optional[int] = None
    window_batch_count: Optional[int] = None
    window_tp_sum: Optional[int] = None
    window_fn_sum: Optional[int] = None
    unclipped_next_intensity: Optional[float] = None
    clipped_next_intensity: Optional[float] = None
    fs_effective_d: Optional[float] = None

@dataclass(frozen=True)
class BatchConfusionLog:
    run_id: str
    batch_id: int
    tp: int
    fp: int
    tn: int
    fn: int
    accuracy: float
    recall: float
    precision: float
    f1: float
    balanced_accuracy: float
    pr_auc_average_precision: float
    eligible_count: int
    attempted_count: int
    successful_count: int
    asr_applicable: bool
    asr: Optional[float]

@dataclass(frozen=True)
class RunSummary:
    run_id: str
    seed: int
    attack_scenario: str
    defense: str
    config_id: str
    total_batches: int
    tp: int
    fp: int
    tn: int
    fn: int
    accuracy: float
    recall: float
    precision: float
    f1: float
    balanced_accuracy: float
    pr_auc_average_precision: float
    total_eligible: int
    total_attempted: int
    total_successful: int
    global_asr: Optional[float]
    completed_successfully: bool

@dataclass(frozen=True)
class CompletionMarker:
    run_id: str
    timestamp: str
    provenance_hashes: Dict[str, str]
    
    def __post_init__(self):
        if not self.provenance_hashes:
            raise ValueError("provenance_hashes must be nonempty")

@dataclass(frozen=True)
class FailureRecord:
    run_id: str
    timestamp: str
    error_type: str
    error_message: str
    failed_at_batch: Optional[int]

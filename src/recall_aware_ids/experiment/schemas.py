from dataclasses import dataclass
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
    sample_id: int
    eligible: bool
    attempted: bool
    successful: bool
    queries_used: int

@dataclass(frozen=True)
class BatchConfigLog:
    run_id: str
    batch_id: int
    config_id: str
    intensity: float
    state: str
    rolling_recall: float
    hit_min_bound: bool
    hit_max_bound: bool

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
    asr: float

@dataclass(frozen=True)
class RunSummary:
    run_id: str
    seed: int
    attack_scenario: str
    defense: str
    config_id: str
    total_batches: int
    clean_tp: int
    clean_fn: int
    attacked_tp: int
    attacked_fn: int
    total_eligible: int
    total_attempted: int
    total_successful: int
    global_asr: float
    completed_successfully: bool

@dataclass(frozen=True)
class CompletionMarker:
    run_id: str
    timestamp: str
    provenance_hashes: Dict[str, str]

@dataclass(frozen=True)
class FailureRecord:
    run_id: str
    timestamp: str
    error_type: str
    error_message: str
    failed_at_batch: Optional[int]

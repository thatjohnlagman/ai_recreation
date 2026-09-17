"""
Frozen dataclasses for all Phase 10 experiment records.
Every dataclass carries __post_init__ validation — malformed objects are rejected at construction time.
"""
from __future__ import annotations
import math
import json
import re
import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

# ---------------------------------------------------------------------------
# Allowed sentinel values
# ---------------------------------------------------------------------------
_VALID_STATES = frozenset({"Base", "Green", "Yellow", "Red"})
_HEX64 = re.compile(r'^[0-9a-f]{64}$')

_REQUIRED_PROVENANCE_KEYS = frozenset({
    "frozen_rf_hash",
    "scaler_hash",
    "feature_names_hash",
    "feature_mask_hash",
    "training_bounds_hash",
    "evaluation_roles_hash",
    "evaluation_batches_hash",
    "attacks_yaml_hash",
    "controllers_yaml_hash",
    "defenses_yaml_hash",
    "experiment_yaml_hash",
})

_PLACEHOLDER_STRINGS = frozenset({"dummy", "hash", "placeholder", "todo", "none", "null", "xxx"})

_ALLOWED_STATUS_CODES = frozenset({
    "SUCCESS", "TARGET_REJECTION", "NO_FEASIBLE_CANDIDATE",
    "BUDGET_EXHAUSTION", "INSUFFICIENT_BUDGET_FOR_FULL_SEARCH",
    "PROJECTION_FAILED_BEFORE_TRANSFER", "PROJECTION_FAILED_DURING_SEARCH",
    "INELIGIBLE_FALSE_NEGATIVE", "INELIGIBLE_TRUE_BENIGN",
    "NOT_ATTEMPTED", "NOT_APPLICABLE",
})

def _is_hex64(s: str) -> bool:
    return bool(_HEX64.match(s))

def _check_finite_nonneg_float(val, name: str):
    if not isinstance(val, float):
        val = float(val)
    if not math.isfinite(val):
        raise ValueError(f"{name} must be finite, got {val!r}")
    if val < 0.0:
        raise ValueError(f"{name} must be >= 0, got {val}")
    return val

def _check_metric_range(val, name: str):
    """Validates a fraction/metric in [0,1]; None passes through."""
    if val is None:
        return
    if not isinstance(val, float):
        val = float(val)
    if not math.isfinite(val):
        raise ValueError(f"{name} must be finite")
    if not (0.0 <= val <= 1.0):
        raise ValueError(f"{name} must be in [0,1], got {val}")

def _validate_iso_timestamp(ts: Any, name: str = "timestamp"):
    if not isinstance(ts, str):
        raise TypeError(f"{name} must be a string, got {type(ts)}")
    if not ts:
        raise ValueError(f"{name} must be a non-empty string")
    try:
        iso_str = ts[:-1] + "+00:00" if ts.endswith("Z") else ts
        datetime.datetime.fromisoformat(iso_str)
    except Exception as exc:
        raise ValueError(f"{name} is not a valid ISO 8601 timestamp: {ts!r}") from exc


# ---------------------------------------------------------------------------
# AttackCacheManifest
# ---------------------------------------------------------------------------
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
    X_attacked_sha256: str
    status_sha256: str
    # Legacy combined field – keep for compatibility but may equal X_attacked_sha256
    output_sha256: str

    def __post_init__(self):
        if not self.attack_scenario:
            raise ValueError("attack_scenario must be nonempty")
        if isinstance(self.effective_seed, bool):
            raise TypeError("effective_seed cannot be bool")
        if not isinstance(self.effective_seed, int) or self.effective_seed < 0:
            raise ValueError("effective_seed must be a non-negative int")
        if self.row_count != 72000:
            raise ValueError(f"row_count must be 72000, got {self.row_count}")
        if self.schema_version != "1.0":
            raise ValueError(f"schema_version must be '1.0', got {self.schema_version!r}")
        # Hash fields must be 64-char hex
        for fname in ("frozen_rf_hash", "evaluation_roles_hash", "evaluation_batches_hash",
                      "X_eval_hash", "metadata_eval_hash", "measurement_identity_hash",
                      "crafting_identity_hash", "feature_names_hash", "feature_mask_hash",
                      "scaler_hash", "training_bounds_hash",
                      "X_attacked_sha256", "status_sha256", "output_sha256"):
            val = getattr(self, fname)
            if not _is_hex64(val):
                raise ValueError(f"{fname} must be a 64-char lowercase hex string, got {val!r}")
        if not isinstance(self.attack_script_hashes, dict):
            raise TypeError("attack_script_hashes must be a dict")
        if not isinstance(self.query_budgets, dict):
            raise TypeError("query_budgets must be a dict")
        if not isinstance(self.attack_parameters, dict):
            raise TypeError("attack_parameters must be a dict")


# ---------------------------------------------------------------------------
# AttackedSampleStatus
# ---------------------------------------------------------------------------
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

    def __post_init__(self):
        if isinstance(self.eval_position, bool):
            raise TypeError("eval_position cannot be bool")
        if not isinstance(self.eval_position, int) or self.eval_position < 0:
            raise ValueError("eval_position must be a non-negative int")
        if not isinstance(self.eligible, bool):
            raise TypeError("eligible must be bool")
        if not isinstance(self.attempted, bool):
            raise TypeError("attempted must be bool")
        if not isinstance(self.successful, bool):
            raise TypeError("successful must be bool")
        if self.status_code not in _ALLOWED_STATUS_CODES:
            raise ValueError(f"status_code {self.status_code!r} not in allowed set: {sorted(_ALLOWED_STATUS_CODES)}")
        if isinstance(self.queries_used, bool):
            raise TypeError("queries_used cannot be bool")
        if not isinstance(self.queries_used, int) or self.queries_used < 0:
            raise ValueError("queries_used must be a non-negative int")
        for name, val in (("l0", self.l0), ("l1", self.l1), ("l2", self.l2), ("linf", self.linf)):
            if not math.isfinite(val):
                raise ValueError(f"{name} must be finite")
            if val < 0.0:
                raise ValueError(f"{name} must be >= 0")
        # Logical consistency
        if self.successful and not self.attempted:
            raise ValueError("successful=True requires attempted=True")
        if self.attempted and not self.eligible:
            raise ValueError("attempted=True requires eligible=True")


# ---------------------------------------------------------------------------
# BatchConfigLog
# ---------------------------------------------------------------------------
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
    # RA-only fields (None for Base)
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
    fs_effective_d: Optional[int] = None

    def __post_init__(self):
        if not self.run_id:
            raise ValueError("run_id must be nonempty")
        if isinstance(self.batch_id, bool):
            raise TypeError("batch_id cannot be bool")
        if not isinstance(self.batch_id, int) or self.batch_id < 0:
            raise ValueError("batch_id must be a non-negative int")
        if not self.config_id:
            raise ValueError("config_id must be nonempty")
        if not math.isfinite(self.intensity) or self.intensity < 0.0:
            raise ValueError(f"intensity must be finite and >= 0, got {self.intensity}")
        if self.state not in _VALID_STATES:
            raise ValueError(f"state {self.state!r} not in {sorted(_VALID_STATES)}")
        if not math.isfinite(self.multiplier) or self.multiplier <= 0.0:
            raise ValueError(f"multiplier must be finite and > 0, got {self.multiplier}")
        if not isinstance(self.hit_min_bound, bool):
            raise TypeError("hit_min_bound must be bool")
        if not isinstance(self.hit_max_bound, bool):
            raise TypeError("hit_max_bound must be bool")
        if not isinstance(self.zero_denominator, bool):
            raise TypeError("zero_denominator must be bool")

        is_base = (self.state == "Base")
        ra_fields = (self.tp, self.fn, self.window_start_batch_id, self.window_end_batch_id,
                     self.configured_window_size, self.window_batch_count,
                     self.window_tp_sum, self.window_fn_sum,
                     self.unclipped_next_intensity, self.clipped_next_intensity)

        if is_base:
            # All RA fields must be None
            for name, val in zip(
                ("tp", "fn", "window_start_batch_id", "window_end_batch_id",
                 "configured_window_size", "window_batch_count",
                 "window_tp_sum", "window_fn_sum",
                 "unclipped_next_intensity", "clipped_next_intensity"),
                ra_fields
            ):
                if val is not None:
                    raise ValueError(f"Base log must have {name}=None, got {val!r}")
        else:
            # RA fields must be present and valid
            if self.tp is None or self.fn is None:
                raise ValueError("RA log must have tp and fn")
            if isinstance(self.tp, bool) or not isinstance(self.tp, int) or self.tp < 0:
                raise ValueError("tp must be a non-negative int")
            if isinstance(self.fn, bool) or not isinstance(self.fn, int) or self.fn < 0:
                raise ValueError("fn must be a non-negative int")
            if self.rolling_recall is not None and not (0.0 <= self.rolling_recall <= 1.0):
                raise ValueError(f"rolling_recall must be in [0,1], got {self.rolling_recall}")
            for name, val in (("unclipped_next_intensity", self.unclipped_next_intensity),
                               ("clipped_next_intensity", self.clipped_next_intensity)):
                if val is None:
                    raise ValueError(f"RA log must have {name}")
                if not math.isfinite(val) or val < 0.0:
                    raise ValueError(f"{name} must be finite and >= 0")


# ---------------------------------------------------------------------------
# BatchConfusionLog
# ---------------------------------------------------------------------------
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

    def __post_init__(self):
        if not self.run_id:
            raise ValueError("run_id must be nonempty")
        if isinstance(self.batch_id, bool) or not isinstance(self.batch_id, int) or self.batch_id < 0:
            raise ValueError("batch_id must be a non-negative int")
        for name in ("tp", "fp", "tn", "fn", "eligible_count", "attempted_count", "successful_count"):
            val = getattr(self, name)
            if isinstance(val, bool):
                raise TypeError(f"{name} cannot be bool")
            if not isinstance(val, int) or val < 0:
                raise ValueError(f"{name} must be a non-negative int")
        for name in ("accuracy", "recall", "precision", "f1", "balanced_accuracy", "pr_auc_average_precision"):
            val = getattr(self, name)
            if not math.isfinite(val):
                raise ValueError(f"{name} must be finite")
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [0,1], got {val}")
        if not isinstance(self.asr_applicable, bool):
            raise TypeError("asr_applicable must be bool")
        # asr nullability
        if self.asr_applicable and self.asr is None:
            raise ValueError("asr_applicable=True but asr is None")
        if not self.asr_applicable and self.asr is not None:
            raise ValueError("asr_applicable=False but asr is not None")
        if self.asr is not None:
            if not math.isfinite(self.asr) or not (0.0 <= self.asr <= 1.0):
                raise ValueError(f"asr must be in [0,1], got {self.asr}")
        # Count relationships
        if self.eligible_count > 500:
            raise ValueError(f"eligible_count ({self.eligible_count}) > 500")
        if self.attempted_count > self.eligible_count:
            raise ValueError(f"attempted_count ({self.attempted_count}) > eligible_count ({self.eligible_count})")
        if self.successful_count > self.attempted_count:
            raise ValueError(f"successful_count ({self.successful_count}) > attempted_count ({self.attempted_count})")
        total = self.tp + self.fp + self.tn + self.fn
        if total != 500:
            raise ValueError(f"confusion metrics must sum to exactly 500, got {total}")
        
        # zero-attempt ASR logic
        if self.asr_applicable and self.attempted_count == 0:
            if self.asr != 0.0:
                raise ValueError(f"asr must be 0.0 when attempted is 0, got {self.asr}")


# ---------------------------------------------------------------------------
# RunSummary
# ---------------------------------------------------------------------------
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
    total_queries: int
    cache_identity: Dict[str, Any]
    status_code_counts: Dict[str, int]
    l0_summary: Dict[str, float]
    l1_summary: Dict[str, float]
    l2_summary: Dict[str, float]
    linf_summary: Dict[str, float]
    global_asr: Optional[float]
    completed_successfully: bool

    def __post_init__(self):
        if not self.run_id:
            raise ValueError("run_id must be nonempty")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be int")
        if self.total_batches != 144:
            raise ValueError(f"total_batches must be 144, got {self.total_batches}")
        for name in ("tp", "fp", "tn", "fn", "total_eligible", "total_attempted", "total_successful"):
            val = getattr(self, name)
            if isinstance(val, bool):
                raise TypeError(f"{name} cannot be bool")
            if not isinstance(val, int) or val < 0:
                raise ValueError(f"{name} must be a non-negative int")
        for name in ("accuracy", "recall", "precision", "f1", "balanced_accuracy", "pr_auc_average_precision"):
            val = getattr(self, name)
            if not math.isfinite(val) or not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [0,1], got {val}")

        # ASR validation based on scenario
        canonical_scenario = self.attack_scenario.replace(" ", "")
        if canonical_scenario == "SilentProbing":
            if self.global_asr is not None:
                raise ValueError(f"Silent Probing requires global_asr is None, got {self.global_asr}")
        elif canonical_scenario in ("SurrogateTransfer", "DecisionBoundary"):
            if self.global_asr is None:
                raise ValueError(f"{self.attack_scenario} requires a finite global_asr in [0,1], got None")
            if not math.isfinite(self.global_asr) or not (0.0 <= self.global_asr <= 1.0):
                raise ValueError(f"global_asr must be in [0,1], got {self.global_asr}")
            if self.total_attempted == 0 and self.global_asr != 0.0:
                raise ValueError(f"Applicable attack with zero attempts requires global_asr == 0.0, got {self.global_asr}")
        else:
            raise ValueError(f"Unknown attack_scenario: {self.attack_scenario}")

        if not isinstance(self.completed_successfully, bool):
            raise TypeError("completed_successfully must be bool")
        if self.total_eligible > 72000:
            raise ValueError(f"total_eligible ({self.total_eligible}) > 72000")
        if self.total_attempted > self.total_eligible:
            raise ValueError("total_attempted > total_eligible")
        if self.total_successful > self.total_attempted:
            raise ValueError("total_successful > total_attempted")

        # total_queries: reject bool, non-negative int
        if isinstance(self.total_queries, bool):
            raise TypeError("total_queries cannot be bool")
        if not isinstance(self.total_queries, int) or self.total_queries < 0:
            raise ValueError("total_queries must be non-negative int")

        # status_code_counts: reject bool, non-negative int, sum to 72000
        if not isinstance(self.status_code_counts, dict):
            raise TypeError("status_code_counts must be dict")
        for k, v in self.status_code_counts.items():
            if k not in _ALLOWED_STATUS_CODES:
                raise ValueError(f"Unknown status code in status_code_counts: {k}")
            if isinstance(v, bool):
                raise TypeError(f"status count for {k} cannot be bool")
            if not isinstance(v, int) or v < 0:
                raise ValueError(f"status count for {k} must be a non-negative int")
        total_status = sum(self.status_code_counts.values())
        if total_status != 72000:
            raise ValueError(f"status_code_counts must sum to exactly 72000, got {total_status}")

        # cache_identity validation
        if not isinstance(self.cache_identity, dict):
            raise TypeError("cache_identity must be dict")
        missing_prov = _REQUIRED_PROVENANCE_KEYS - set(self.cache_identity.keys())
        if missing_prov:
            raise ValueError(f"cache_identity missing required provenance keys: {sorted(missing_prov)}")
        for pk in _REQUIRED_PROVENANCE_KEYS:
            pval = self.cache_identity[pk]
            if not isinstance(pval, str) or not _is_hex64(pval):
                raise ValueError(f"cache_identity[{pk!r}] must be a 64-char lowercase hex string, got {pval!r}")
            if pval in ("0" * 64, "a" * 64, "b" * 64) or any(p in pval.lower() for p in ("dummy", "placeholder", "todo", "xxx")):
                raise ValueError(f"cache_identity[{pk!r}] contains dummy/placeholder hash: {pval!r}")

        # Magnitude summaries: mean, min, max, std, finite, non-negative, min <= mean <= max, std >= 0
        for m_name in ("l0_summary", "l1_summary", "l2_summary", "linf_summary"):
            m_dict = getattr(self, m_name)
            if not isinstance(m_dict, dict):
                raise TypeError(f"{m_name} must be dict")
            expected_keys = {"mean", "min", "max", "std"}
            if set(m_dict.keys()) != expected_keys:
                raise ValueError(f"{m_name} must contain exactly {expected_keys}, got {set(m_dict.keys())}")
            for k in expected_keys:
                mv = m_dict[k]
                if isinstance(mv, bool):
                    raise TypeError(f"{m_name}[{k!r}] cannot be bool")
                if not isinstance(mv, (int, float)) or not math.isfinite(mv):
                    raise ValueError(f"{m_name}[{k!r}] must be finite numeric, got {mv!r}")
                if mv < 0.0:
                    raise ValueError(f"{m_name}[{k!r}] cannot be negative, got {mv}")
            if not (m_dict["min"] <= m_dict["mean"] + 1e-9 and m_dict["mean"] <= m_dict["max"] + 1e-9):
                raise ValueError(f"{m_name} violates min <= mean <= max: min={m_dict['min']}, mean={m_dict['mean']}, max={m_dict['max']}")
            if m_dict["std"] < 0.0:
                raise ValueError(f"{m_name} std cannot be negative, got {m_dict['std']}")

        total = self.tp + self.fp + self.tn + self.fn
        if total != 72000:
            raise ValueError(f"confusion metrics must sum to exactly 72000, got {total}")


# ---------------------------------------------------------------------------
# CompletionMarker
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CompletionMarker:
    run_id: str
    timestamp: str
    provenance_hashes: Dict[str, str]

    def __post_init__(self):
        if not self.run_id:
            raise ValueError("run_id must be nonempty")
        if not self.provenance_hashes:
            raise ValueError("provenance_hashes must be nonempty")
        # Check required keys
        missing = _REQUIRED_PROVENANCE_KEYS - set(self.provenance_hashes.keys())
        if missing:
            raise ValueError(f"provenance_hashes missing required keys: {sorted(missing)}")
        # Validate each value
        for key, val in self.provenance_hashes.items():
            if not isinstance(key, str) or not isinstance(val, str):
                raise TypeError(f"provenance key/value must be strings, got {key!r}: {val!r}")
            if key.lower() in _PLACEHOLDER_STRINGS or val.lower() in _PLACEHOLDER_STRINGS:
                raise ValueError(f"provenance_hashes contains placeholder: {key!r}: {val!r}")
            if not _is_hex64(val):
                raise ValueError(
                    f"provenance_hashes[{key!r}] must be a 64-char lowercase hex string, got {val!r}"
                )
        # JSON-serialisable (no exotic types)
        try:
            json.dumps(self.provenance_hashes)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"provenance_hashes must be JSON-serialisable: {exc}")
        _validate_iso_timestamp(self.timestamp, "timestamp")


# ---------------------------------------------------------------------------
# FailureRecord
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FailureRecord:
    run_id: str
    timestamp: str
    error_type: str
    error_message: str
    failed_at_batch: Optional[int]

    def __post_init__(self):
        if not self.run_id:
            raise ValueError("run_id must be nonempty")
        _validate_iso_timestamp(self.timestamp, "timestamp")
        if self.failed_at_batch is not None:
            if isinstance(self.failed_at_batch, bool):
                raise TypeError("failed_at_batch cannot be bool")
            if not isinstance(self.failed_at_batch, int) or self.failed_at_batch < 0:
                raise ValueError("failed_at_batch must be a non-negative int")

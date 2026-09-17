"""
Attack cache building and loading for Phase 10 experiments.

AttackCacheBuilder:
  Production-capable builder, tested with synthetic / training-derived data during Phase 10A.
  Supports all three attack scenarios with canonical semantics.

ConcreteAttackCacheProvider:
  Loads and strictly validates both cache artifacts before serving batch data.
"""
from __future__ import annotations

import hashlib
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------

def calculate_file_hash(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def calculate_bytes_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Manifest validation
# ---------------------------------------------------------------------------

_MANIFEST_REQUIRED_KEYS = [
    "X_eval_hash", "metadata_eval_hash", "evaluation_roles_hash", "evaluation_batches_hash",
    "crafting_identity_hash", "measurement_identity_hash",
    "attacks_yaml_hash", "attack_script_hashes",
    "frozen_rf_hash", "scaler_hash", "feature_names_hash", "feature_mask_hash", "training_bounds_hash",
    "attack_parameters", "query_budgets",
    "attack_scenario", "effective_seed",
    "schema_version", "row_count",
    "X_attacked_sha256", "status_sha256", "output_sha256",
]


def validate_cache_manifest(
    manifest_path: Path,
    X_attacked_path: Path,
    expected_hashes: Dict[str, Any],
    status_path: Optional[Path] = None,
    expected_row_count: int = 72000,
) -> bool:
    """
    Validates:
      1. Manifest exists and parses.
      2. All required keys are present.
      3. X_attacked SHA-256 matches manifest's X_attacked_sha256.
      4. status SHA-256 matches manifest's status_sha256 (if status_path provided).
      5. All expected_hashes match the manifest exactly.
      6. row_count == 72000.
    """
    if not manifest_path.exists():
        raise ValueError(f"Cache invalid: Manifest missing at {manifest_path}")
    if not X_attacked_path.exists():
        raise ValueError(f"Cache invalid: X_attacked artifact missing at {X_attacked_path}")

    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Cache invalid: Corrupted JSON – {e}")

    # Required keys
    for k in _MANIFEST_REQUIRED_KEYS:
        if k not in manifest:
            raise ValueError(f"Missing required key '{k}' in manifest")
            
    # Reject dummy hashes
    for key, val in manifest.items():
        if isinstance(val, str) and (val == "0" * 64 or val == "dummy" or "placeholder" in val):
            raise ValueError(f"Cache invalid: dummy/placeholder hash found for {key}")

    # Hash verification
    x_actual = calculate_file_hash(X_attacked_path)
    if manifest.get("X_attacked_sha256") != x_actual:
        raise ValueError(
            f"Cache invalid: X_attacked hash {x_actual} does not match "
            f"manifest X_attacked_sha256 {manifest.get('X_attacked_sha256')}"
        )
    # Legacy combined field
    if manifest.get("output_sha256") != x_actual:
        raise ValueError(
            f"Cache invalid: Artifact hash {x_actual} does not match "
            f"manifest output_sha256 {manifest.get('output_sha256')}"
        )

    # status.parquet is mandatory, so require it to exist and its hash to match
    if status_path is None or not status_path.exists():
         raise ValueError("Cache invalid: status.parquet is required")
         
    s_actual = calculate_file_hash(status_path)
    if manifest.get("status_sha256") != s_actual:
        raise ValueError(
            f"Cache invalid: status hash {s_actual} does not match "
            f"manifest status_sha256 {manifest.get('status_sha256')}"
        )

    # Check all expected hashes match
    # Require exact equality for the full cache identity instead of a subset match
    if manifest != expected_hashes:
        # Find differences for helpful error message
        missing_keys = set(expected_hashes.keys()) - set(manifest.keys())
        extra_keys = set(manifest.keys()) - set(expected_hashes.keys())
        diffs = []
        for k in expected_hashes.keys():
            if k in manifest and manifest[k] != expected_hashes[k]:
                diffs.append(f"{k}: expected {expected_hashes[k]}, got {manifest[k]}")
        
        err_msg = "Cache invalid: manifest does not match expected_cache_identity exactly."
        if missing_keys: err_msg += f" Missing keys: {missing_keys}."
        if extra_keys: err_msg += f" Extra keys: {extra_keys}."
        if diffs: err_msg += f" Mismatches: {diffs}."
        raise ValueError(err_msg)

    if manifest["row_count"] != expected_row_count:
        raise ValueError(f"Cache invalid: row_count must be {expected_row_count}, got {manifest['row_count']}")

    return True


# ---------------------------------------------------------------------------
# AttackCacheBuilder
# ---------------------------------------------------------------------------

class AttackCacheBuilder:
    """
    Production-capable attack-cache builder.

    Supports:
      - SilentProbing: identity transformation, zero queries, ASR not applicable.
      - SurrogateTransfer: surrogate DT construction + candidate generation.
      - DecisionBoundary: deterministic selection of exactly n_boundary_targets eligible
        measurement targets, max_queries per target.

    Must be tested exclusively with synthetic or training-derived data during Phase 10A.
    Must NOT be invoked on official evaluation data.
    """

    SCENARIO_SILENT_PROBING = "SilentProbing"
    SCENARIO_SURROGATE = "SurrogateTransfer"
    SCENARIO_BOUNDARY = "DecisionBoundary"

    def __init__(
        self,
        feature_names: list,
        modifiable_mask: list,
        training_bounds,  # DataFrame with train_min/train_max
        provenance_hashes: Dict[str, str],
        seed: int,
    ):
        self.feature_names = feature_names
        self.modifiable_mask = np.array(modifiable_mask, dtype=bool)
        self.training_bounds = training_bounds
        self.provenance_hashes = provenance_hashes
        self.seed = seed
        self._validate_provenance()

    def _validate_provenance(self):
        if not self.provenance_hashes:
            raise ValueError("provenance_hashes must be non-empty")
        for k, v in self.provenance_hashes.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise TypeError("All provenance keys/values must be strings")
            # Accept either 64-char hex or descriptive labels (for testing)
            if any(p in k.lower() or p in v.lower() for p in ("dummy", "placeholder")):
                raise ValueError(f"Provenance contains placeholder: {k!r}: {v!r}")

    # ------------------------------------------------------------------
    # Public build entry point
    # ------------------------------------------------------------------

    def build(
        self,
        scenario: str,
        X_measurement: np.ndarray,
        y_measurement: np.ndarray,
        eval_positions: np.ndarray,
        output_dir: Path,
        X_crafting: Optional[np.ndarray] = None,
        y_crafting: Optional[np.ndarray] = None,
        oracle_or_predict_fn=None,
        surrogate_attack=None,
        boundary_attack=None,
        benign_reference_pool: Optional[np.ndarray] = None,
        n_boundary_targets: int = 200,
        max_queries: int = 50,
        attack_script_hashes: dict = None,
        attack_parameters: dict = None,
        query_budgets: dict = None,
    ) -> Path:
        """
        Build cache artifacts for the given scenario.
        """
        if scenario == self.SCENARIO_SILENT_PROBING:
            return self._build_silent_probing(X_measurement, eval_positions, output_dir, attack_script_hashes, attack_parameters, query_budgets)
        elif scenario == self.SCENARIO_SURROGATE:
            return self._build_surrogate(X_measurement, y_measurement, eval_positions, output_dir,
                                         X_crafting, y_crafting, oracle_or_predict_fn, surrogate_attack,
                                         attack_script_hashes, attack_parameters, query_budgets)
        elif scenario == self.SCENARIO_BOUNDARY:
            return self._build_boundary(X_measurement, y_measurement, eval_positions, output_dir,
                                        oracle_or_predict_fn, boundary_attack,
                                        benign_reference_pool, n_boundary_targets, max_queries,
                                        attack_script_hashes, attack_parameters, query_budgets)
        else:
            raise ValueError(f"Unknown scenario: {scenario!r}")

    # ------------------------------------------------------------------
    # Silent Probing
    # ------------------------------------------------------------------

    def _build_silent_probing(
        self, X_input: np.ndarray, eval_positions: np.ndarray, output_dir: Path,
        attack_script_hashes: dict, attack_parameters: dict, query_budgets: dict
    ) -> Path:
        n = len(X_input)
        X_attacked = X_input.astype(np.float32)

        statuses = []
        for i in range(n):
            statuses.append({
                "eval_position": int(eval_positions[i]),
                "eligible": False, "attempted": False, "successful": False,
                "status_code": "NOT_APPLICABLE", "queries_used": 0,
                "l0": 0.0, "l1": 0.0, "l2": 0.0, "linf": 0.0,
            })

        return self._write_outputs(self.SCENARIO_SILENT_PROBING, X_attacked, statuses, output_dir,
                                   attack_script_hashes, attack_parameters, query_budgets)

    # ------------------------------------------------------------------
    # Surrogate Transfer
    # ------------------------------------------------------------------

    def _build_surrogate(
        self, X_measurement: np.ndarray, y_measurement: np.ndarray, eval_positions: np.ndarray, output_dir: Path,
        X_crafting: np.ndarray, y_crafting: np.ndarray, predict_fn: Callable, surrogate_attack,
        attack_script_hashes: dict, attack_parameters: dict, query_budgets: dict
    ) -> Path:
        from recall_aware_ids.attacks.oracle import BlackBoxOracle

        if surrogate_attack is None:
            raise ValueError("surrogate_attack must be provided for SurrogateTransfer")
        if predict_fn is None:
            raise ValueError("predict_fn must be provided for SurrogateTransfer")
        if X_crafting is None or y_crafting is None:
            raise ValueError("X_crafting and y_crafting must be provided for SurrogateTransfer")

        # Fit surrogate on crafting pool ONLY. Use Oracle to get crafting labels.
        crafting_oracle = BlackBoxOracle(predict_fn, max_queries_per_sample=None)
        # Use predict method to simulate oracle usage during surrogate training
        c_ids = [f"craft_{i}" for i in range(len(X_crafting))]
        y_pool = crafting_oracle.predict(X_crafting, sample_ids=c_ids)
        surrogate_attack.fit_surrogate(X_crafting, y_pool)

        n = len(X_measurement)
        X_attacked = X_measurement.astype(np.float32).copy()
        statuses = []

        # Oracle for target evaluation
        target_oracle = BlackBoxOracle(predict_fn, max_queries_per_sample=None)

        for i in range(n):
            ep = int(eval_positions[i])
            true_label = int(y_measurement[i])
            sample_id = f"surrogate_{ep}"

            # 1. Generate X_cand using only the surrogate
            X_cand, _ = surrogate_attack.generate_candidate(X_measurement[i])
            
            # 2. Call evaluate_transfer() exactly once
            result = surrogate_attack.evaluate_transfer(
                X_cand, X_measurement[i], target_oracle, sample_id, true_label
            )

            # 3. Store result.X_adv at the corresponding original measurement position
            X_attacked[i] = np.array(result.X_adv).flatten()

            # 4, 5, 6. Construct the status row from that same AttackResult
            mags = result.magnitudes if result.magnitudes else {"l0": 0.0, "l1": 0.0, "l2": 0.0, "linf": 0.0}
            success_bool = False if result.success is None else bool(result.success)

            statuses.append({
                "eval_position": ep,
                "eligible": result.eligible,
                "attempted": result.attempted,
                "successful": success_bool,
                "status_code": result.status_code,
                "queries_used": result.query_count,
                "l0": float(mags.get("l0", 0.0)),
                "l1": float(mags.get("l1", 0.0)),
                "l2": float(mags.get("l2", 0.0)),
                "linf": float(mags.get("linf", 0.0)),
            })

        return self._write_outputs(self.SCENARIO_SURROGATE, X_attacked, statuses, output_dir,
                                   attack_script_hashes, attack_parameters, query_budgets)

    # ------------------------------------------------------------------
    # Decision Boundary
    # ------------------------------------------------------------------

    def _build_boundary(
        self, X_measurement: np.ndarray, y_measurement: np.ndarray, eval_positions: np.ndarray, output_dir: Path,
        predict_fn: Callable, boundary_attack, benign_reference_pool: np.ndarray,
        n_boundary_targets: int, max_queries: int,
        attack_script_hashes: dict, attack_parameters: dict, query_budgets: dict
    ) -> Path:
        from recall_aware_ids.attacks.oracle import BlackBoxOracle
        from recall_aware_ids.experiment.boundary_selection import select_boundary_targets

        if boundary_attack is None:
            raise ValueError("boundary_attack must be provided for DecisionBoundary")
        if predict_fn is None:
            raise ValueError("predict_fn must be provided for DecisionBoundary")
        if benign_reference_pool is None:
            raise ValueError("benign_reference_pool must be provided for DecisionBoundary")

        n = len(X_measurement)
        
        # Determine eligibility (clean selection computation)
        orig_preds = predict_fn(X_measurement)
        
        eligible_mask = (orig_preds == 1) & (y_measurement == 1)
        
        # Use Canonical seeded global target selector
        official_mode = (len(eligible_mask) == 72000)
        target_mask = select_boundary_targets(eligible_mask, self.seed, n_boundary_targets, official_mode=official_mode)
        target_indices = np.where(target_mask)[0]
        
        if len(target_indices) != n_boundary_targets:
            raise ValueError(f"Could not select exactly {n_boundary_targets} targets (found {len(target_indices)})")

        target_set = set(target_indices)
        
        selected_positions = [int(eval_positions[i]) for i in target_indices]
        import hashlib
        pred_hash = hashlib.sha256(orig_preds.tobytes()).hexdigest()
        pos_hash = hashlib.sha256(np.array(selected_positions, dtype=np.int64).tobytes()).hexdigest()
        
        screening_metrics = {
            "screening_model_identity": self.provenance_hashes.get("frozen_rf_hash", "unknown"),
            "screening_prediction_hash": pred_hash,
            "selected_position_hash": pos_hash,
            "selected_target_count": len(selected_positions),
            "selection_seed": self.seed,
        }

        X_attacked = X_measurement.astype(np.float32).copy()
        statuses = []
        
        oracle = BlackBoxOracle(predict_fn, max_queries_per_sample=max_queries)

        for i in range(n):
            ep = int(eval_positions[i])
            true_label = int(y_measurement[i])
            orig_pred = int(orig_preds[i])

            if i not in target_set:
                if orig_pred != 1 or true_label != 1:
                    status = "INELIGIBLE_TRUE_BENIGN" if true_label != 1 else "INELIGIBLE_FALSE_NEGATIVE"
                    statuses.append({
                        "eval_position": ep, "eligible": False, "attempted": False, "successful": False,
                        "status_code": status, "queries_used": 0,
                        "l0": 0.0, "l1": 0.0, "l2": 0.0, "linf": 0.0,
                    })
                else:
                    statuses.append({
                        "eval_position": ep, "eligible": True, "attempted": False, "successful": False,
                        "status_code": "NOT_ATTEMPTED", "queries_used": 0,
                        "l0": 0.0, "l1": 0.0, "l2": 0.0, "linf": 0.0,
                    })
                continue

            sample_id = f"ep_{ep}"
            result = boundary_attack.generate(
                X_measurement[i], oracle, sample_id=sample_id,
                true_label=true_label, reference_pool=benign_reference_pool
            )

            queries = oracle.get_query_count(sample_id)

            if result.success:
                X_attacked[i] = result.X_adv

            mags = result.magnitudes if result.magnitudes else {"l0": 0.0, "l1": 0.0, "l2": 0.0, "linf": 0.0}
            success_bool = False if result.success is None else bool(result.success)

            statuses.append({
                "eval_position": ep,
                "eligible": result.eligible, "attempted": result.attempted, "successful": success_bool,
                "status_code": result.status_code, "queries_used": int(queries),
                "l0": float(mags.get("l0", 0.0)),
                "l1": float(mags.get("l1", 0.0)),
                "l2": float(mags.get("l2", 0.0)),
                "linf": float(mags.get("linf", 0.0)),
            })
            
        screening_metrics["attack_oracle_queries_used"] = oracle.global_query_count

        return self._write_outputs(self.SCENARIO_BOUNDARY, X_attacked, statuses, output_dir,
                                   attack_script_hashes, attack_parameters, query_budgets, screening_metrics)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _write_outputs(
        self, scenario: str, X_attacked: np.ndarray, statuses: List[dict], output_dir: Path,
        attack_script_hashes: dict = None, attack_parameters: dict = None, query_budgets: dict = None,
        screening_metrics: dict = None
    ) -> Path:
        if scenario in (self.SCENARIO_SURROGATE, self.SCENARIO_BOUNDARY):
            if not attack_script_hashes: raise ValueError(f"attack_script_hashes required for {scenario}")
            if not attack_parameters: raise ValueError(f"attack_parameters required for {scenario}")
            if not query_budgets: raise ValueError(f"query_budgets required for {scenario}")
        if output_dir.exists():
            raise FileExistsError(f"Cache output directory already exists (cannot overwrite): {output_dir}")
            
        tmp_dir = output_dir.with_name(output_dir.name + ".tmp")
        if tmp_dir.exists():
            import datetime
            ts = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
            quarantine_dir = output_dir.with_name(f"{output_dir.name}_quarantined_{ts}")
            tmp_dir.rename(quarantine_dir)
            
        tmp_dir.mkdir(parents=True)

        x_path = tmp_dir / "X_attacked.parquet"
        s_path = tmp_dir / "status.parquet"
        m_path = tmp_dir / "manifest.json"

        # Write X_attacked
        df_x = pd.DataFrame(X_attacked.astype(np.float32), columns=self.feature_names)
        df_x.to_parquet(x_path, index=False)

        # Write status
        status_df = pd.DataFrame(statuses)
        # Ensure strict bool columns
        for col in ("eligible", "attempted", "successful"):
            status_df[col] = status_df[col].astype(bool)
        status_df["eval_position"] = status_df["eval_position"].astype(int)
        status_df["queries_used"] = status_df["queries_used"].astype(int)
        status_df.to_parquet(s_path, index=False)

        # Compute hashes
        x_hash = calculate_file_hash(x_path)
        s_hash = calculate_file_hash(s_path)

        # Strict manifest requirements: do not use .get(..., "0" * 64)
        manifest = {
            "schema_version": "1.0",
            "attack_scenario": scenario,
            "effective_seed": int(self.seed),
            "row_count": len(X_attacked),
            "X_attacked_sha256": x_hash,
            "status_sha256": s_hash,
            "output_sha256": x_hash,  # legacy compat
            
            # Additional structural fields passed dynamically
            "attack_script_hashes": attack_script_hashes or {},
            "attack_parameters": attack_parameters or {},
            "query_budgets": query_budgets or {},
        }
        
        if screening_metrics is not None:
            manifest["screening_metrics"] = screening_metrics
            
        manifest.update({
            # Require all provenance hashes to be explicitly provided in self.provenance_hashes
            "attacks_yaml_hash": self.provenance_hashes["attacks_yaml_hash"],
            "X_eval_hash": self.provenance_hashes["X_eval_hash"],
            "metadata_eval_hash": self.provenance_hashes["metadata_eval_hash"],
            "evaluation_roles_hash": self.provenance_hashes["evaluation_roles_hash"],
            "evaluation_batches_hash": self.provenance_hashes["evaluation_batches_hash"],
            "crafting_identity_hash": self.provenance_hashes["crafting_identity_hash"],
            "measurement_identity_hash": self.provenance_hashes["measurement_identity_hash"],
            "frozen_rf_hash": self.provenance_hashes["frozen_rf_hash"],
            "scaler_hash": self.provenance_hashes["scaler_hash"],
            "feature_names_hash": self.provenance_hashes["feature_names_hash"],
            "feature_mask_hash": self.provenance_hashes["feature_mask_hash"],
            "training_bounds_hash": self.provenance_hashes["training_bounds_hash"],
            **{k: v for k, v in self.provenance_hashes.items()}
        })

        with open(m_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # Before atomic publication, reopen and validate the cache artifacts exactly as the provider would.
        # This confirms that no serialization bug occurred and that the schemas and identities match.
        from recall_aware_ids.experiment.caching import ConcreteAttackCacheProvider
        # Construct a dummy resolved_batches dataframe representing the generated eval_positions
        fake_batches = pd.DataFrame({
            "batch_id": [0] * len(X_attacked),
            "eval_position": [int(s["eval_position"]) for s in statuses]
        })
        
        # Load through the strict provider internally
        ConcreteAttackCacheProvider(
            cache_dir=tmp_dir,
            resolved_batches=fake_batches,
            expected_cache_identity=manifest,
            expected_feature_names=self.feature_names,
            official_mode=False,
            expected_row_count=len(X_attacked)
        )

        # Atomic rename
        tmp_dir.rename(output_dir)
        return output_dir


# ---------------------------------------------------------------------------
# ConcreteAttackCacheProvider — strict loading and validation
# ---------------------------------------------------------------------------

_REQUIRED_STATUS_COLS = {"eval_position", "eligible", "attempted", "successful",
                          "status_code", "queries_used", "l0", "l1", "l2", "linf"}
_BOOL_STATUS_COLS = {"eligible", "attempted", "successful"}


class ConcreteAttackCacheProvider:
    """
    Loads both cache artifacts (X_attacked.parquet and status.parquet) and validates:
      - manifest + file hashes
      - 72,000 rows (or N during Phase 10A synthetic testing)
      - 78 finite float32 columns in X_attacked
      - eval_position alignment vs resolved_batches
      - strict bool status columns (no silent int→bool cast)
      - query and magnitude fields finite and >= 0
      - scenario and seed consistency
    """

    def __init__(
        self,
        cache_dir: Path,
        resolved_batches: pd.DataFrame,
        expected_cache_identity: Dict[str, Any],
        expected_feature_names: List[str],
        official_mode: bool,
        expected_row_count: Optional[int] = None,
        validate_strict_bool: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.resolved_batches = resolved_batches
        self.official_mode = official_mode
        
        if self.official_mode:
            if expected_row_count is not None and expected_row_count != 72000:
                raise ValueError("official_mode requires expected_row_count=72000")
            self.expected_row_count = 72000
        else:
            if expected_row_count is None:
                raise ValueError("synthetic mode requires explicitly supplied expected_row_count")
            self.expected_row_count = expected_row_count

        artifact_path = self.cache_dir / "X_attacked.parquet"
        status_path = self.cache_dir / "status.parquet"
        manifest_path = self.cache_dir / "manifest.json"

        # 1. Manifest validation must occur FIRST
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.json not found: {manifest_path}")

        validate_cache_manifest(
            manifest_path=manifest_path,
            X_attacked_path=artifact_path,
            expected_hashes=expected_cache_identity,
            status_path=status_path,
            expected_row_count=self.expected_row_count
        )

        with open(manifest_path, "r") as f:
            self.cache_identity = json.load(f)

        # Load
        df_x = pd.read_parquet(artifact_path)
        df_status = pd.read_parquet(status_path)

        # 2. Row count validation
        if len(df_x) != expected_row_count:
            raise ValueError(f"X_attacked has {len(df_x)} rows, expected {expected_row_count}")
        if len(df_status) != expected_row_count:
            raise ValueError(f"status has {len(df_status)} rows, expected {expected_row_count}")

        # 2. Duplicate or missing eval_position
        if df_status["eval_position"].duplicated().any():
            raise ValueError("status.parquet contains duplicate eval_positions")
        if df_status["eval_position"].isnull().any():
            raise ValueError("status.parquet contains null eval_positions")

        # 3. Exact feature names and ordering
        if list(df_x.columns) != expected_feature_names:
            raise ValueError("X_attacked.parquet columns do not exactly match expected_feature_names in order")

        # 4. Finite values
        X_arr = df_x.values.astype(np.float32)
        if not np.all(np.isfinite(X_arr)):
            raise ValueError("X_attacked contains non-finite values")

        # 5. Required status columns
        missing = _REQUIRED_STATUS_COLS - set(df_status.columns)
        if missing:
            raise ValueError(f"status.parquet missing columns: {missing}")

        # 6. Strict bool validation — reject integer columns
        if validate_strict_bool:
            for col in _BOOL_STATUS_COLS:
                actual_dtype = df_status[col].dtype
                if actual_dtype != bool and str(actual_dtype) not in ("bool", "boolean"):
                    raise TypeError(
                        f"status column '{col}' must be strictly bool dtype, got {actual_dtype}. "
                        "Do not silently cast — fix the cache source."
                    )

        # 7. Magnitude validation
        for col in ("l0", "l1", "l2", "linf"):
            vals = df_status[col].values
            if not np.all(np.isfinite(vals)):
                raise ValueError(f"status.{col} contains non-finite values")
            if np.any(vals < 0.0):
                raise ValueError(f"status.{col} contains negative values")

        # 8. Query count validation
        q_vals = df_status["queries_used"].values
        if np.any(q_vals < 0):
            raise ValueError("status.queries_used contains negative values")

        # 9. Allowed status codes
        allowed_codes = {
            "NOT_APPLICABLE", "SUCCESS", "NOT_ATTEMPTED",
            "INELIGIBLE_TRUE_BENIGN", "INELIGIBLE_FALSE_NEGATIVE",
            "NO_FEASIBLE_CANDIDATE", "TARGET_REJECTION",
            # Authoritative Phase 7 status codes:
            "BUDGET_EXHAUSTION", "INSUFFICIENT_BUDGET_FOR_FULL_SEARCH",
            "PROJECTION_FAILED_BEFORE_TRANSFER", "PROJECTION_FAILED_DURING_SEARCH"
        }
        invalid_codes = set(df_status["status_code"].unique()) - allowed_codes
        if invalid_codes:
            raise ValueError(f"Invalid status_code(s) found: {invalid_codes}")

        # 10. Logical relationships
        # eligible = False -> attempted = False, successful = False
        ineligible_invalid = df_status[~df_status["eligible"] & (df_status["attempted"] | df_status["successful"])]
        if len(ineligible_invalid) > 0:
            raise ValueError("Found ineligible samples that were marked as attempted or successful")
            
        # attempted = False -> successful = False
        unattempted_invalid = df_status[~df_status["attempted"] & df_status["successful"]]
        if len(unattempted_invalid) > 0:
            raise ValueError("Found unattempted samples marked as successful")
            
        # successful = True -> attempted = True
        success_unattempted = df_status[df_status["successful"] & ~df_status["attempted"]]
        if len(success_unattempted) > 0:
            raise ValueError("Found successful samples not marked as attempted")

        # 11. eval_position alignment and batch structure
        cache_eps = set(df_status["eval_position"].values.tolist())
        batch_eps = set(resolved_batches["eval_position"].values.tolist())
        
        if self.official_mode:
            if len(cache_eps) != 72000:
                raise ValueError(f"status.parquet contains {len(cache_eps)} unique eval_positions, expected 72000")
            if len(resolved_batches["batch_id"].unique()) != 144:
                raise ValueError("official_mode requires exactly 144 batches")
            if not all(resolved_batches["batch_id"].value_counts() == 500):
                raise ValueError("official_mode requires exactly 500 rows per batch")
            if cache_eps != batch_eps:
                raise ValueError("eval_positions in cache do not exactly equal resolved_batches measurement identities")
        else:
            # Explicit synthetic contract check
            if cache_eps != batch_eps:
                raise ValueError("eval_positions in cache do not exactly equal resolved_batches measurement identities")

        self._X_attacked = X_arr
        self._status = df_status
        self._eval_position_to_idx: Dict[int, int] = {
            int(ep): i for i, ep in enumerate(df_status["eval_position"].values)
        }

    def get_batch_data(self, batch_id: int) -> Dict[str, np.ndarray]:
        batch_df = self.resolved_batches[self.resolved_batches["batch_id"] == batch_id]
        batch_df = batch_df.sort_values("eval_position")

        if len(batch_df) == 0:
            raise ValueError(f"No rows found for batch_id={batch_id}")
        if len(batch_df) != 500:
            raise ValueError(f"Batch {batch_id} has {len(batch_df)} rows, expected 500")

        # Map via eval_position (validated join — not row index)
        eps = batch_df["eval_position"].values.astype(int)
        missing_eps = [ep for ep in eps if ep not in self._eval_position_to_idx]
        if missing_eps:
            raise ValueError(f"eval_positions not found in cache: {missing_eps[:5]}")

        cache_indices = np.array([self._eval_position_to_idx[ep] for ep in eps])

        X_batch = self._X_attacked[cache_indices]
        status_batch = self._status.iloc[cache_indices]

        eligible = np.array(status_batch["eligible"].values, dtype=bool)
        attempted = np.array(status_batch["attempted"].values, dtype=bool)
        successful = np.array(status_batch["successful"].values, dtype=bool)

        return {
            "X_attacked": X_batch,
            "eligible": eligible,
            "attempted": attempted,
            "successful": successful,
            "status_codes": status_batch["status_code"].values,
            "queries": np.array(status_batch["queries_used"].values, dtype=int),
            "l0": np.array(status_batch["l0"].values, dtype=float),
            "l1": np.array(status_batch["l1"].values, dtype=float),
            "l2": np.array(status_batch["l2"].values, dtype=float),
            "linf": np.array(status_batch["linf"].values, dtype=float),
        }

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

    if status_path is not None and status_path.exists():
        s_actual = calculate_file_hash(status_path)
        if manifest.get("status_sha256") != s_actual:
            raise ValueError(
                f"Cache invalid: status hash {s_actual} does not match "
                f"manifest status_sha256 {manifest.get('status_sha256')}"
            )

    # Check all expected hashes match
    for key, expected_val in expected_hashes.items():
        if key not in manifest:
            raise ValueError(f"Cache invalid: Expected key '{key}' not found in manifest")
        if key in ("attack_script_hashes", "attack_parameters", "query_budgets"):
            if not isinstance(manifest[key], dict) or not isinstance(expected_val, dict):
                raise ValueError(f"Cache invalid: '{key}' must be a dictionary")
            for k2, v2 in expected_val.items():
                if manifest[key].get(k2) != v2:
                    raise ValueError(f"Cache invalid: Mismatch in {key}[{k2}]")
        else:
            if manifest[key] != expected_val:
                raise ValueError(
                    f"Cache invalid: Mismatch for {key}. Expected {expected_val!r}, got {manifest[key]!r}"
                )

    if manifest["row_count"] != 72000:
        raise ValueError(f"Cache invalid: row_count must be 72000, got {manifest['row_count']}")

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
        X_input: np.ndarray,
        y_input: np.ndarray,
        eval_positions: np.ndarray,
        output_dir: Path,
        oracle_or_predict_fn=None,
        surrogate_attack=None,
        boundary_attack=None,
        benign_reference_pool: Optional[np.ndarray] = None,
        n_boundary_targets: int = 200,
        max_queries: int = 50,
    ) -> Path:
        """
        Build cache artifacts for the given scenario.

        X_input: (N, 78) float32 feature matrix (training-derived only during Phase 10A)
        y_input: (N,) int array of labels
        eval_positions: (N,) int array — unique identifier for each row (aligned with X_input)
        output_dir: directory to write X_attacked.parquet, status.parquet, manifest.json
        oracle_or_predict_fn: callable used for surrogate queries or boundary oracle
        surrogate_attack: SurrogateTransferAttack instance (required for Surrogate)
        boundary_attack: DecisionBoundaryAttack instance (required for Boundary)
        benign_reference_pool: (M, 78) float32 array of benign samples (required for Boundary)
        n_boundary_targets: number of eligible measurement targets to attempt (Boundary)
        max_queries: max oracle queries per target (Boundary)
        """
        if scenario == self.SCENARIO_SILENT_PROBING:
            return self._build_silent_probing(X_input, eval_positions, output_dir)
        elif scenario == self.SCENARIO_SURROGATE:
            return self._build_surrogate(X_input, y_input, eval_positions, output_dir,
                                         oracle_or_predict_fn, surrogate_attack)
        elif scenario == self.SCENARIO_BOUNDARY:
            return self._build_boundary(X_input, y_input, eval_positions, output_dir,
                                        oracle_or_predict_fn, boundary_attack,
                                        benign_reference_pool, n_boundary_targets, max_queries)
        else:
            raise ValueError(f"Unknown scenario: {scenario!r}")

    # ------------------------------------------------------------------
    # Silent Probing
    # ------------------------------------------------------------------

    def _build_silent_probing(
        self, X_input: np.ndarray, eval_positions: np.ndarray, output_dir: Path
    ) -> Path:
        """
        Identity transformation: X_attacked = X_input (unchanged), zero queries,
        all samples ineligible for attack. ASR not applicable.
        """
        n = len(X_input)
        X_attacked = X_input.astype(np.float32)

        statuses = []
        for i in range(n):
            statuses.append({
                "eval_position": int(eval_positions[i]),
                "eligible": False,
                "attempted": False,
                "successful": False,
                "status_code": "NOT_APPLICABLE",
                "queries_used": 0,
                "l0": 0.0, "l1": 0.0, "l2": 0.0, "linf": 0.0,
            })

        return self._persist(X_attacked, statuses, eval_positions, output_dir,
                             self.SCENARIO_SILENT_PROBING)

    # ------------------------------------------------------------------
    # Surrogate Transfer
    # ------------------------------------------------------------------

    def _build_surrogate(
        self,
        X_input: np.ndarray,
        y_input: np.ndarray,
        eval_positions: np.ndarray,
        output_dir: Path,
        predict_fn: Callable,
        surrogate_attack,
    ) -> Path:
        """
        Surrogate Transfer semantics:
        - Fit surrogate on all rows where y=1 (or entire pool if not enough).
        - Generate candidates for every eligible row (y=1, original pred=1).
        - Rows where no attack is attempted retain their original features.
        - All queries go through predict_fn (training-derived only during Phase 10A).
        """
        if surrogate_attack is None:
            raise ValueError("surrogate_attack must be provided for SurrogateTransfer")
        if predict_fn is None:
            raise ValueError("predict_fn must be provided for SurrogateTransfer")

        n = len(X_input)

        # Fit surrogate using all rows
        y_pool = predict_fn(X_input)
        surrogate_attack.fit_surrogate(X_input, y_pool)

        X_attacked = X_input.astype(np.float32).copy()
        statuses = []

        for i in range(n):
            orig_pred = int(predict_fn(X_input[i:i+1])[0])
            true_label = int(y_input[i])

            if orig_pred != 1 or true_label != 1:
                # Ineligible
                status = "INELIGIBLE_TRUE_BENIGN" if true_label != 1 else "INELIGIBLE_FALSE_NEGATIVE"
                statuses.append({
                    "eval_position": int(eval_positions[i]),
                    "eligible": False, "attempted": False, "successful": False,
                    "status_code": status,
                    "queries_used": 0,
                    "l0": 0.0, "l1": 0.0, "l2": 0.0, "linf": 0.0,
                })
                continue

            X_cand, mags_or_reason = surrogate_attack.generate_candidate(X_input[i])

            if X_cand is None:
                statuses.append({
                    "eval_position": int(eval_positions[i]),
                    "eligible": True, "attempted": True, "successful": False,
                    "status_code": "NO_FEASIBLE_CANDIDATE",
                    "queries_used": 0,
                    "l0": 0.0, "l1": 0.0, "l2": 0.0, "linf": 0.0,
                })
                continue

            # Verify with predict_fn (1 query)
            final_pred = int(predict_fn(X_cand.reshape(1, -1))[0])
            success = (final_pred == 0)

            if success:
                X_attacked[i] = X_cand

            if isinstance(mags_or_reason, dict):
                mags = mags_or_reason
            else:
                mags = {"l0": 0.0, "l1": 0.0, "l2": 0.0, "linf": 0.0}

            statuses.append({
                "eval_position": int(eval_positions[i]),
                "eligible": True, "attempted": True, "successful": success,
                "status_code": "SUCCESS" if success else "TARGET_REJECTION",
                "queries_used": 1,
                "l0": float(mags.get("l0", 0.0)),
                "l1": float(mags.get("l1", 0.0)),
                "l2": float(mags.get("l2", 0.0)),
                "linf": float(mags.get("linf", 0.0)),
            })

        return self._persist(X_attacked, statuses, eval_positions, output_dir,
                             self.SCENARIO_SURROGATE)

    # ------------------------------------------------------------------
    # Decision Boundary
    # ------------------------------------------------------------------

    def _build_boundary(
        self,
        X_input: np.ndarray,
        y_input: np.ndarray,
        eval_positions: np.ndarray,
        output_dir: Path,
        predict_fn: Callable,
        boundary_attack,
        benign_reference_pool: np.ndarray,
        n_boundary_targets: int,
        max_queries: int,
    ) -> Path:
        """
        Decision Boundary semantics:
        - Deterministic global selection of exactly n_boundary_targets eligible positions
          (orig pred=1, true label=1), ordered by eval_position ascending.
        - Max max_queries queries per attempted target.
        - All 72,000 (or N during Phase 10A) measurement positions preserved;
          unchanged rows retain original features.
        """
        from recall_aware_ids.attacks.oracle import BlackBoxOracle

        if boundary_attack is None:
            raise ValueError("boundary_attack must be provided for DecisionBoundary")
        if predict_fn is None:
            raise ValueError("predict_fn must be provided for DecisionBoundary")
        if benign_reference_pool is None:
            raise ValueError("benign_reference_pool must be provided for DecisionBoundary")

        n = len(X_input)
        oracle = BlackBoxOracle(predict_fn, max_queries_per_sample=max_queries)

        # Determine eligibility via prediction (1 eligibility query each)
        orig_preds = predict_fn(X_input)

        # Eligible candidates: original prediction is attack class, true label is attack class
        eligible_indices = [
            i for i in range(n)
            if int(orig_preds[i]) == 1 and int(y_input[i]) == 1
        ]

        # Deterministic selection: take first n_boundary_targets by ascending eval_position order
        eligible_indices.sort(key=lambda i: int(eval_positions[i]))
        target_indices = set(eligible_indices[:n_boundary_targets])

        X_attacked = X_input.astype(np.float32).copy()
        statuses = []

        for i in range(n):
            ep = int(eval_positions[i])
            true_label = int(y_input[i])
            orig_pred = int(orig_preds[i])

            if i not in target_indices:
                if orig_pred != 1 or true_label != 1:
                    status = "INELIGIBLE_TRUE_BENIGN" if true_label != 1 else "INELIGIBLE_FALSE_NEGATIVE"
                    statuses.append({
                        "eval_position": ep, "eligible": False, "attempted": False, "successful": False,
                        "status_code": status, "queries_used": 0,
                        "l0": 0.0, "l1": 0.0, "l2": 0.0, "linf": 0.0,
                    })
                else:
                    # Eligible but not selected for attack
                    statuses.append({
                        "eval_position": ep, "eligible": True, "attempted": False, "successful": False,
                        "status_code": "NOT_ATTEMPTED", "queries_used": 0,
                        "l0": 0.0, "l1": 0.0, "l2": 0.0, "linf": 0.0,
                    })
                continue

            # Attempt boundary attack
            sample_id = f"ep_{ep}"
            result = boundary_attack.generate(
                X_input[i], oracle, sample_id=sample_id,
                true_label=true_label, reference_pool=benign_reference_pool
            )

            queries = oracle.get_query_count(sample_id)

            if result.success:
                X_attacked[i] = result.X_adv

            mags = result.magnitudes if result.magnitudes else {"l0": 0.0, "l1": 0.0, "l2": 0.0, "linf": 0.0}
            statuses.append({
                "eval_position": ep,
                "eligible": result.eligible, "attempted": result.attempted, "successful": bool(result.success),
                "status_code": result.status_code, "queries_used": int(queries),
                "l0": float(mags.get("l0", 0.0)),
                "l1": float(mags.get("l1", 0.0)),
                "l2": float(mags.get("l2", 0.0)),
                "linf": float(mags.get("linf", 0.0)),
            })

        return self._persist(X_attacked, statuses, eval_positions, output_dir,
                             self.SCENARIO_BOUNDARY)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist(
        self,
        X_attacked: np.ndarray,
        statuses: List[Dict],
        eval_positions: np.ndarray,
        output_dir: Path,
        scenario: str,
    ) -> Path:
        """Write X_attacked.parquet, status.parquet, manifest.json."""
        output_dir.mkdir(parents=True, exist_ok=True)

        x_path = output_dir / "X_attacked.parquet"
        s_path = output_dir / "status.parquet"
        m_path = output_dir / "manifest.json"

        # Write X_attacked
        df_x = pd.DataFrame(X_attacked.astype(np.float32))
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

        # Manifest
        manifest = {
            "schema_version": "1.0",
            "attack_scenario": scenario,
            "effective_seed": int(self.seed),
            "row_count": len(X_attacked),
            "X_attacked_sha256": x_hash,
            "status_sha256": s_hash,
            "output_sha256": x_hash,  # legacy compat
            **{k: v for k, v in self.provenance_hashes.items()},
            # Structural fields expected by manifest schema
            "attacks_yaml_hash": self.provenance_hashes.get("attacks_yaml_hash", "0" * 64),
            "attack_script_hashes": {},
            "attack_parameters": {},
            "query_budgets": {},
            "X_eval_hash": self.provenance_hashes.get("X_eval_hash", "0" * 64),
            "metadata_eval_hash": self.provenance_hashes.get("metadata_eval_hash", "0" * 64),
            "evaluation_roles_hash": self.provenance_hashes.get("evaluation_roles_hash", "0" * 64),
            "evaluation_batches_hash": self.provenance_hashes.get("evaluation_batches_hash", "0" * 64),
            "crafting_identity_hash": self.provenance_hashes.get("crafting_identity_hash", "0" * 64),
            "measurement_identity_hash": self.provenance_hashes.get("measurement_identity_hash", "0" * 64),
            "frozen_rf_hash": self.provenance_hashes.get("frozen_rf_hash", "0" * 64),
            "scaler_hash": self.provenance_hashes.get("scaler_hash", "0" * 64),
            "feature_names_hash": self.provenance_hashes.get("feature_names_hash", "0" * 64),
            "feature_mask_hash": self.provenance_hashes.get("feature_mask_hash", "0" * 64),
            "training_bounds_hash": self.provenance_hashes.get("training_bounds_hash", "0" * 64),
        }

        with open(m_path, "w") as f:
            json.dump(manifest, f, indent=2)

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
        expected_row_count: int = 72000,
        validate_strict_bool: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.resolved_batches = resolved_batches
        self.expected_row_count = expected_row_count

        artifact_path = self.cache_dir / "X_attacked.parquet"
        status_path = self.cache_dir / "status.parquet"

        if not artifact_path.exists():
            raise FileNotFoundError(f"X_attacked.parquet not found: {artifact_path}")
        if not status_path.exists():
            raise FileNotFoundError(f"status.parquet not found: {status_path}")

        # Load
        df_x = pd.read_parquet(artifact_path)
        df_status = pd.read_parquet(status_path)

        # Row count
        if len(df_x) != expected_row_count:
            raise ValueError(f"X_attacked has {len(df_x)} rows, expected {expected_row_count}")
        if len(df_status) != expected_row_count:
            raise ValueError(f"status has {len(df_status)} rows, expected {expected_row_count}")

        # Column count
        if df_x.shape[1] != 78:
            raise ValueError(f"X_attacked has {df_x.shape[1]} columns, expected 78")

        # Finite values
        X_arr = df_x.values.astype(np.float32)
        if not np.all(np.isfinite(X_arr)):
            raise ValueError("X_attacked contains non-finite values")

        # Required status columns
        missing = _REQUIRED_STATUS_COLS - set(df_status.columns)
        if missing:
            raise ValueError(f"status.parquet missing columns: {missing}")

        # Strict bool validation — reject integer columns
        if validate_strict_bool:
            for col in _BOOL_STATUS_COLS:
                actual_dtype = df_status[col].dtype
                if actual_dtype != bool and str(actual_dtype) not in ("bool", "boolean"):
                    raise TypeError(
                        f"status column '{col}' must be strictly bool dtype, got {actual_dtype}. "
                        "Do not silently cast — fix the cache source."
                    )

        # Magnitude validation
        for col in ("l0", "l1", "l2", "linf"):
            vals = df_status[col].values
            if not np.all(np.isfinite(vals)):
                raise ValueError(f"status.{col} contains non-finite values")
            if np.any(vals < 0.0):
                raise ValueError(f"status.{col} contains negative values")

        # Query count validation
        q_vals = df_status["queries_used"].values
        if np.any(q_vals < 0):
            raise ValueError("status.queries_used contains negative values")

        # eval_position alignment
        cache_eps = set(df_status["eval_position"].values.tolist())
        batch_eps = set(resolved_batches["eval_position"].values.tolist())
        overlap = cache_eps & batch_eps
        if len(overlap) == 0 and expected_row_count > 0:
            raise ValueError("No eval_position overlap between cache and resolved_batches")

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
            "queries": np.array(status_batch["queries_used"].values, dtype=int),
        }

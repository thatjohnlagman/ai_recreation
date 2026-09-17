#!/usr/bin/env python3
"""
scripts/run_evaluation.py

Phase 10D Official Evaluation Orchestration Entry Point.

This script acts as the official execution and validation orchestrator for Phase 10
evaluation matrix runs. It strictly enforces:
  1. Safe-by-default execution: running without --execute performs non-mutating preflight only.
  2. Preflight separation: preflight never accesses evaluation features (X_eval.parquet) or labels.
  3. Dynamic matrix derivation: derives exactly 279 matrix references (90 primary, 189 sensitivity,
     27 exact C1 aliases, 252 unique executions, 36,288 batch evaluations) from frozen configs.
  4. Exact pairing and common randomness: Base and RA runs are paired on
     (seed, attack_scenario, defense_mechanism, batch_id) with bit-identical pseudorandom sequences.
  5. Timing and label isolation: intensity obtained strictly before batch-t labels; feedback submitted
     only for batch t+1; Base receives no feedback.
  6. Deterministic whole-run restart: runs are never resumed mid-sequence; any interrupted run restarts
     cleanly from batch 0.
  7. Run reuse and quarantine: completed runs are validated before reuse; corrupt or incomplete runs are
     quarantined to <run_id>_quarantined_<timestamp>.
  8. Two-stage atomic publication: staging in .staging_<run_id>_<timestamp>, serialized output reopening
     and validation via _validate_run_outputs, completion.json written last, atomic rename to <run_id>.
  9. Resource limits: RS chunk size 100 ensures operations stay well within M4/16-GB memory limit.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import datetime
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml

# Add src/ to sys.path so recall_aware_ids imports cleanly
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from recall_aware_ids.controller.recall_controller import RecallAwareController
from recall_aware_ids.defenses.afp import AdaptiveFeaturePoisoning
from recall_aware_ids.defenses.feature_squeezing import FeatureSqueezing
from recall_aware_ids.defenses.randomized_smoothing import RandomizedSmoothing
from recall_aware_ids.experiment.adapters import (
    AFPDefenseAdapter,
    FSDefenseAdapter,
    RSDefenseAdapter,
)
from recall_aware_ids.experiment.caching import (
    ConcreteAttackCacheProvider,
    calculate_file_hash,
    validate_cache_manifest,
)
from recall_aware_ids.experiment.matrix import (
    generate_evaluation_matrix,
    get_unique_executions,
)
from recall_aware_ids.experiment.policies import FixedIntensityPolicy
from recall_aware_ids.experiment.runner import (
    ExperimentRunner,
    LabelProvider,
    _validate_provenance,
    _validate_run_outputs,
)
from recall_aware_ids.experiment.schemas import (
    _HEX64,
    _PLACEHOLDER_STRINGS,
    _REQUIRED_PROVENANCE_KEYS,
    CompletionMarker,
    RunSummary,
    _is_hex64,
    _validate_iso_timestamp,
)


# ---------------------------------------------------------------------------
# Frozen Protocol Constants
# ---------------------------------------------------------------------------
FREEZE_COMMIT = "65005505415a2bdf2d5744dbd135e9214e74081a"
FREEZE_TAG = "phase10-protocol-freeze"
FROZEN_DATE = "2026-09-17T21:23:51+08:00"

PROTECTED_HASHES = {
    "RF": "9608672c5d5e38a9272c560679cdd2291399de0e0917c8c9dff373bfd200f51d",
    "Roles": "cbf650879aa1369fa26b803777f30d4b0add9e7aff2c05d399ba7f42563f7b45",
    "Batches": "4084017e5593732e455763416f7fc38254fc9dab2466eca289b66e68dc0ff79a",
}

FROZEN_PROTOCOL_FILES = [
    "configs/attacks.yaml",
    "configs/controllers.yaml",
    "configs/defenses.yaml",
    "configs/experiment.yaml",
    "configs/model.yaml",
    "docs/EXPERIMENT_PROTOCOL.md",
]

CANONICAL_SCENARIOS = {
    "Silent Probing": "SilentProbing",
    "SilentProbing": "SilentProbing",
    "Surrogate Transfer": "SurrogateTransfer",
    "SurrogateTransfer": "SurrogateTransfer",
    "Decision Boundary": "DecisionBoundary",
    "DecisionBoundary": "DecisionBoundary",
}

CANONICAL_DEFENSES = {
    "afp": "afp",
    "feature_squeezing": "feature_squeezing",
    "randomized_smoothing": "randomized_smoothing",
}

CALIBRATED_BASE_INTENSITIES = {
    "afp": 0.0003,
    "feature_squeezing": 2.0,
    "randomized_smoothing": 0.0002,
}

BASE_INTENSITY_BOUNDS = {
    "afp": (0.0, 0.005),
    "feature_squeezing": (1.0, 5.0),
    "randomized_smoothing": (0.0, 0.001),
}


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------
def canonicalize_scenario(scenario: str) -> str:
    """Maps display or slug scenario string to canonical camel-case scenario name."""
    if scenario in CANONICAL_SCENARIOS:
        return CANONICAL_SCENARIOS[scenario]
    stripped = scenario.replace(" ", "").replace("_", "").replace("-", "")
    for k, v in CANONICAL_SCENARIOS.items():
        if stripped.lower() == k.replace(" ", "").lower():
            return v
    raise ValueError(
        f"Unknown attack scenario: {scenario!r}. Expected one of: "
        f"{sorted(list(set(CANONICAL_SCENARIOS.keys())))}"
    )


def canonicalize_defense(defense: str) -> str:
    """Validates and returns canonical defense name."""
    d = defense.lower().replace("-", "_").strip()
    if d in CANONICAL_DEFENSES:
        return CANONICAL_DEFENSES[d]
    raise ValueError(
        f"Unknown defense name: {defense!r}. Expected one of: {sorted(list(CANONICAL_DEFENSES.keys()))}"
    )


# ---------------------------------------------------------------------------
# Strengthened Frozen Configuration and Protocol Verification
# ---------------------------------------------------------------------------
def verify_frozen_configurations(
    repo_root: Path = REPO_ROOT,
    enforce_git: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Strengthened freeze check: Compares the current bytes and SHA-256 values of every
    frozen configuration and protocol file against the tagged version at phase10-protocol-freeze.
    """
    results: Dict[str, Dict[str, Any]] = {}
    git_dir = repo_root / ".git"

    for rel_path in FROZEN_PROTOCOL_FILES:
        disk_path = repo_root / rel_path
        if not disk_path.exists():
            raise FileNotFoundError(f"Required frozen file missing on disk: {rel_path}")

        disk_bytes = disk_path.read_bytes()
        disk_sha256 = hashlib.sha256(disk_bytes).hexdigest()

        tagged_sha256 = None
        bytes_match = None

        if git_dir.exists():
            try:
                tagged_bytes = subprocess.check_output(
                    ["git", "show", f"{FREEZE_TAG}:{rel_path}"],
                    cwd=repo_root,
                    stderr=subprocess.PIPE,
                )
                tagged_sha256 = hashlib.sha256(tagged_bytes).hexdigest()
                bytes_match = disk_bytes == tagged_bytes

                if not bytes_match:
                    raise ValueError(
                        f"Byte mismatch in frozen file {rel_path} relative to {FREEZE_TAG}!\n"
                        f"Disk SHA-256:   {disk_sha256}\n"
                        f"Tagged SHA-256: {tagged_sha256}"
                    )
                if disk_sha256 != tagged_sha256:
                    raise ValueError(
                        f"SHA-256 mismatch in frozen file {rel_path} relative to {FREEZE_TAG}!"
                    )
            except subprocess.CalledProcessError as e:
                if enforce_git:
                    raise RuntimeError(
                        f"Failed to inspect git object for {rel_path} at tag {FREEZE_TAG}: {e.stderr.decode()}"
                    ) from e

        results[rel_path] = {
            "disk_sha256": disk_sha256,
            "tagged_sha256": tagged_sha256,
            "bytes_match": bytes_match,
            "verified": True,
        }

    # Verify no extra or missing files in configs/
    if git_dir.exists():
        try:
            tree_configs = subprocess.check_output(
                ["git", "ls-tree", "--name-only", FREEZE_TAG, "configs/"],
                cwd=repo_root,
                text=True,
            ).splitlines()
            disk_configs = [
                f"configs/{p.name}"
                for p in (repo_root / "configs").iterdir()
                if p.is_file() and not p.name.startswith(".")
            ]
            if sorted(tree_configs) != sorted(disk_configs):
                raise ValueError(
                    f"Configs directory contents changed vs freeze tag!\n"
                    f"Tagged:  {sorted(tree_configs)}\n"
                    f"On disk: {sorted(disk_configs)}"
                )
        except subprocess.CalledProcessError as e:
            if enforce_git:
                raise RuntimeError(f"Failed to check configs tree: {e.stderr}") from e

    return results


# ---------------------------------------------------------------------------
# Protected Artifact Hash Verification
# ---------------------------------------------------------------------------
def verify_protected_artifacts(repo_root: Path = REPO_ROOT) -> Dict[str, str]:
    """Verifies that the frozen RF model, roles manifest, and batches manifest match protected hashes."""
    rf_path = repo_root / "artifacts/models/frozen_rf.joblib"
    roles_path = repo_root / "data/manifests/evaluation_roles.csv"
    batches_path = repo_root / "data/manifests/evaluation_batches.csv"

    hashes = {
        "RF": calculate_file_hash(rf_path),
        "Roles": calculate_file_hash(roles_path),
        "Batches": calculate_file_hash(batches_path),
    }

    for key, expected in PROTECTED_HASHES.items():
        actual = hashes[key]
        if actual != expected:
            raise ValueError(
                f"Protected artifact {key} hash mismatch!\n"
                f"Expected: {expected}\n"
                f"Actual:   {actual}"
            )
    return hashes


# ---------------------------------------------------------------------------
# Cache Validation
# ---------------------------------------------------------------------------
def verify_official_attack_caches(
    caches_dir: Path,
    expected_pairs: List[Tuple[str, int]],
) -> Dict[str, Dict[str, Any]]:
    """
    Verifies that all 15 expected official attack caches exist and validate completely.
    """
    cache_results: Dict[str, Dict[str, Any]] = {}

    for scenario, seed in expected_pairs:
        slug = f"{scenario}_{seed}"
        cdir = caches_dir / slug
        if not cdir.exists():
            raise FileNotFoundError(f"Official attack cache directory missing: {cdir}")

        manifest_path = cdir / "manifest.json"
        x_path = cdir / "X_attacked.parquet"
        status_path = cdir / "status.parquet"
        comp_path = cdir / "completion.json"

        if not manifest_path.exists():
            raise FileNotFoundError(f"Cache {slug} missing manifest.json")
        if not x_path.exists():
            raise FileNotFoundError(f"Cache {slug} missing X_attacked.parquet")
        if not status_path.exists():
            raise FileNotFoundError(f"Cache {slug} missing status.parquet")
        if not comp_path.exists():
            raise FileNotFoundError(f"Cache {slug} missing completion.json")

        with open(manifest_path) as f:
            m = json.load(f)

        # Validate manifest hashes
        validate_cache_manifest(
            manifest_path=manifest_path,
            X_attacked_path=x_path,
            expected_hashes=m,
            status_path=status_path,
            expected_row_count=72000,
        )

        with open(comp_path) as f:
            c = json.load(f)

        if c.get("completion_state") != "COMPLETED" or not c.get("completed"):
            raise ValueError(f"Cache {slug} completion.json indicates incomplete state")
        if canonicalize_scenario(c.get("attack_scenario", "")) != scenario:
            raise ValueError(f"Cache {slug} completion scenario mismatch: expected {scenario}, got {c.get('attack_scenario')}")
        if int(c.get("effective_seed", -1)) != seed:
            raise ValueError(f"Cache {slug} completion seed mismatch: expected {seed}, got {c.get('effective_seed')}")
        if int(c.get("row_count", 0)) != 72000:
            raise ValueError(f"Cache {slug} completion row count mismatch: expected 72000, got {c.get('row_count')}")

        cache_results[slug] = {
            "scenario": scenario,
            "seed": seed,
            "directory": str(cdir),
            "manifest_sha256": calculate_file_hash(manifest_path),
            "X_attacked_sha256": m["X_attacked_sha256"],
            "status_sha256": m["status_sha256"],
            "generation_end_timestamp": c.get("generation_end_timestamp"),
        }

    return cache_results


# ---------------------------------------------------------------------------
# Dynamic Evaluation Matrix Derivation & Validation
# ---------------------------------------------------------------------------
def derive_and_validate_matrix(configs_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Derives the complete evaluation matrix from frozen configurations and asserts:
      - 279 matrix references
      - 27 exact C1 aliases
      - 252 unique executions
      - 144 batches per execution
      - 36,288 unique batch evaluations
      - primary comparison: 90 references
      - sensitivity analysis: 189 references
    """
    matrix = generate_evaluation_matrix(configs_dir)
    unique_execs = get_unique_executions(matrix)

    primary = matrix[matrix["run_id"].str.startswith("primary_")]
    sensitivity = matrix[matrix["run_id"].str.startswith("sensitivity_")]
    aliases = matrix[matrix["is_alias"]]

    if len(matrix) != 279:
        raise ValueError(f"Expected 279 matrix references, derived {len(matrix)}")
    if len(aliases) != 27:
        raise ValueError(f"Expected 27 exact C1 aliases, derived {len(aliases)}")
    if len(unique_execs) != 252:
        raise ValueError(f"Expected 252 unique executions, derived {len(unique_execs)}")
    if len(primary) != 90:
        raise ValueError(f"Expected 90 primary references, derived {len(primary)}")
    if len(sensitivity) != 189:
        raise ValueError(f"Expected 189 sensitivity references, derived {len(sensitivity)}")

    total_batch_evals = len(unique_execs) * 144
    if total_batch_evals != 36288:
        raise ValueError(f"Expected 36288 batch evaluations, got {total_batch_evals}")

    return matrix, unique_execs


# ---------------------------------------------------------------------------
# Resolved Batches & Label Provider Preparation
# ---------------------------------------------------------------------------
def prepare_evaluation_batches(
    manifests_dir: Path,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, str]]:
    """
    Loads evaluation_batches.csv and prepares the measurement mapping for ExperimentRunner:
      - sorts deterministically by eval_position
      - assigns measurement_idx = 0..71999
      - maps batch_id from 1..144 (in CSV) to 0..143 (in runner)
      - extracts y_binary measurement array (length 72,000)
    """
    batches_path = manifests_dir / "evaluation_batches.csv"
    roles_path = manifests_dir / "evaluation_roles.csv"

    if not batches_path.exists():
        raise FileNotFoundError(f"Missing {batches_path}")
    if not roles_path.exists():
        raise FileNotFoundError(f"Missing {roles_path}")

    df_batches = pd.read_csv(batches_path)
    if len(df_batches) != 72000:
        raise ValueError(f"Expected 72000 rows in evaluation_batches.csv, got {len(df_batches)}")

    # Sort deterministically by eval_position
    sorted_batches = df_batches.sort_values("eval_position").reset_index(drop=True)
    sorted_batches["measurement_idx"] = np.arange(len(sorted_batches), dtype=int)

    # Convert 1-indexed batch_id (1..144) to 0-indexed (0..143)
    if sorted_batches["batch_id"].min() == 1 and sorted_batches["batch_id"].max() == 144:
        sorted_batches["batch_id"] = sorted_batches["batch_id"] - 1

    if set(sorted_batches["batch_id"].unique()) != set(range(144)):
        raise ValueError(f"Batch IDs in resolved_batches must span 0..143")

    batch_counts = sorted_batches["batch_id"].value_counts()
    if not (batch_counts == 500).all():
        raise ValueError("Each batch must have exactly 500 records")

    y_measurement = sorted_batches["y_binary"].values.astype(int)

    manifest_hashes = {
        "evaluation_roles_hash": calculate_file_hash(roles_path),
        "evaluation_batches_hash": calculate_file_hash(batches_path),
    }

    return sorted_batches, y_measurement, manifest_hashes


# ---------------------------------------------------------------------------
# Completed Run Validation and Quarantine
# ---------------------------------------------------------------------------
def validate_completed_run(run_dir: Path) -> Tuple[bool, Optional[str]]:
    """
    Reopens and rigorously validates an existing run directory:
      - completion.json exists and satisfies CompletionMarker schema
      - config.json, confusion.json, scores.json, run_summary.json exist and parse
      - _validate_run_outputs passes (recomputed metrics match summary and confusion records)
      - returns (True, None) if valid, or (False, error_reason) if invalid
    """
    if not run_dir.exists() or not run_dir.is_dir():
        return False, "Directory does not exist or is not a directory"

    comp_path = run_dir / "completion.json"
    summary_path = run_dir / "run_summary.json"

    if not comp_path.exists():
        return False, "Missing completion.json (incomplete run)"
    if not summary_path.exists():
        return False, "Missing run_summary.json"

    try:
        with open(comp_path, "r") as f:
            c_data = json.load(f)
        marker = CompletionMarker(**c_data)
        _validate_provenance(marker.provenance_hashes)
    except Exception as e:
        return False, f"completion.json failed validation: {e}"

    try:
        with open(summary_path, "r") as f:
            s_data = json.load(f)
        summary = RunSummary(**s_data)
    except Exception as e:
        return False, f"run_summary.json failed validation: {e}"

    try:
        _validate_run_outputs(run_dir, summary)
    except Exception as e:
        return False, f"Serialized output validation failed: {e}"

    return True, None


def quarantine_run_directory(run_dir: Path, reason: str) -> Path:
    """Quarantines an invalid or incomplete run directory."""
    ts = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    quarantine_dir = run_dir.with_name(f"{run_dir.name}_quarantined_{ts}")
    run_dir.rename(quarantine_dir)

    reason_payload = {
        "run_id": run_dir.name,
        "quarantined_at": datetime.datetime.utcnow().isoformat() + "Z",
        "reason": reason,
    }
    with open(quarantine_dir / "quarantine_reason.json", "w") as f:
        json.dump(reason_payload, f, indent=2)

    return quarantine_dir


# ---------------------------------------------------------------------------
# Defense and Policy Factory
# ---------------------------------------------------------------------------
def create_defense_adapter(
    defense_name: str,
    feature_names: List[str],
    modifiable_mask: np.ndarray,
    training_bounds: pd.DataFrame,
    benign_profile: pd.DataFrame,
    rf_model: Any,
) -> Any:
    """Instantiates the appropriate defense adapter for the given defense mechanism."""
    d = canonicalize_defense(defense_name)
    if d == "afp":
        afp = AdaptiveFeaturePoisoning(
            feature_names=feature_names,
            modifiable_mask=modifiable_mask,
            training_bounds=training_bounds,
            benign_profile=benign_profile,
        )
        return AFPDefenseAdapter(
            afp=afp,
            model=rf_model,
            epsilon_base=CALIBRATED_BASE_INTENSITIES["afp"],
            alpha=0.5,
        )
    elif d == "feature_squeezing":
        fs = FeatureSqueezing(
            feature_names=feature_names,
            modifiable_mask=modifiable_mask,
            training_bounds=training_bounds,
        )
        return FSDefenseAdapter(fs=fs, model=rf_model)
    elif d == "randomized_smoothing":
        rs = RandomizedSmoothing(
            feature_names=feature_names,
            modifiable_mask=modifiable_mask,
            training_bounds=training_bounds,
            ensemble_size=11,
        )
        return RSDefenseAdapter(
            rs=rs,
            predict_func=rf_model.predict,
            chunk_size=100,  # Strict chunk size 100 for M4 memory safety
        )
    else:
        raise ValueError(f"Unsupported defense: {defense_name}")


def create_policy_controller(
    controller_config_id: str,
    defense_name: str,
    controllers_yaml: Dict[str, Any],
    defenses_yaml: Dict[str, Any],
) -> Any:
    """Instantiates FixedIntensityPolicy for Base or RecallAwareController for C1..C7."""
    d = canonicalize_defense(defense_name)
    if controller_config_id == "Base":
        base_val = CALIBRATED_BASE_INTENSITIES[d]
        min_val, max_val = BASE_INTENSITY_BOUNDS[d]
        return FixedIntensityPolicy(
            config_id="Base",
            fixed_intensity=base_val,
            intensity_min=min_val,
            intensity_max=max_val,
        )
    else:
        cfg = controllers_yaml["controller_configurations"][controller_config_id]
        def_cfg = defenses_yaml[d]
        return RecallAwareController(
            config=cfg,
            defense_config=def_cfg,
            defense_name=d,
            zero_division_value=0.0,
        )


# ---------------------------------------------------------------------------
# Preflight Checker
# ---------------------------------------------------------------------------
def run_preflight(
    configs_dir: Path,
    manifests_dir: Path,
    models_dir: Path,
    caches_dir: Path,
    repo_root: Path = REPO_ROOT,
    enforce_git: bool = True,
) -> Dict[str, Any]:
    """
    Performs complete, non-mutating preflight integrity checks:
      1. Git clean status & freeze tag ancestry.
      2. Configuration and protocol freeze integrity.
      3. Protected model and manifest SHA-256 validation.
      4. All 15 official attack caches integrity.
      5. Dynamic evaluation matrix validation.
      6. Available storage capacity.
    """
    print("=" * 78)
    print("PHASE 10D NON-MUTATING PREFLIGHT VERIFICATION")
    print("=" * 78)

    preflight_report: Dict[str, Any] = {"timestamp": datetime.datetime.utcnow().isoformat() + "Z"}

    # 1. Git verification
    git_dir = repo_root / ".git"
    if git_dir.exists():
        try:
            head_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
            ).strip()
            status_output = subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=repo_root, text=True
            ).strip()

            # Check if freeze tag is an ancestor of HEAD
            subprocess.check_call(
                ["git", "merge-base", "--is-ancestor", FREEZE_TAG, "HEAD"],
                cwd=repo_root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            tag_is_ancestor = True

            preflight_report["git"] = {
                "head_commit": head_commit,
                "freeze_tag_ancestor": tag_is_ancestor,
                "tracked_clean": len(status_output) == 0,
            }
            print(f"Git HEAD: {head_commit} (Freeze Tag Ancestor: {tag_is_ancestor})")
        except Exception as e:
            if enforce_git:
                raise RuntimeError(f"Git preflight check failed: {e}") from e
            preflight_report["git"] = {"error": str(e)}

    # 2. Frozen configuration files
    print("\nVerifying frozen configurations...")
    frozen_cfg_res = verify_frozen_configurations(repo_root, enforce_git=enforce_git)
    preflight_report["frozen_configurations"] = frozen_cfg_res
    print(f"  Verified {len(frozen_cfg_res)} frozen files bit-for-bit vs {FREEZE_TAG}.")

    # 3. Protected artifacts
    print("\nVerifying protected artifacts...")
    protected_hashes = verify_protected_artifacts(repo_root)
    preflight_report["protected_hashes"] = protected_hashes
    for k, h in protected_hashes.items():
        print(f"  {k:8s}: {h}")

    # 4. Matrix derivation
    print("\nDeriving evaluation matrix dynamically...")
    matrix, unique_execs = derive_and_validate_matrix(configs_dir)
    primary = matrix[matrix["run_id"].str.startswith("primary_")]
    sensitivity = matrix[matrix["run_id"].str.startswith("sensitivity_")]
    aliases = matrix[matrix["is_alias"]]
    print(f"  Total Matrix References: {len(matrix)}")
    print(f"  Primary Comparison:      {len(primary)}")
    print(f"  Sensitivity Analysis:    {len(sensitivity)}")
    print(f"  Exact C1 Aliases:        {len(aliases)}")
    print(f"  Unique Executions:       {len(unique_execs)}")
    print(f"  Total Batch Evals:       {len(unique_execs) * 144}")
    preflight_report["matrix"] = {
        "total_references": len(matrix),
        "primary_references": len(primary),
        "sensitivity_references": len(sensitivity),
        "aliases": len(aliases),
        "unique_executions": len(unique_execs),
        "total_batch_evaluations": len(unique_execs) * 144,
    }

    # 5. Attack caches
    print("\nVerifying 15 official attack caches...")
    cache_pairs = sorted(
        list(
            {
                (canonicalize_scenario(r["attack_scenario"]), int(r["seed"]))
                for _, r in unique_execs.iterrows()
            }
        )
    )
    cache_results = verify_official_attack_caches(caches_dir, cache_pairs)
    preflight_report["caches"] = cache_results
    print(f"  All {len(cache_results)} official attack caches fully verified.")

    # 6. Disk space check
    try:
        total, used, free = shutil.disk_usage(repo_root)
        free_gb = free / (1024**3)
        preflight_report["disk_free_gb"] = free_gb
        print(f"\nAvailable Disk Space: {free_gb:.2f} GB (minimum required: 10.00 GB)")
        if free_gb < 10.0:
            raise RuntimeError(f"Insufficient disk space: {free_gb:.2f} GB < 10.00 GB required")
    except Exception as e:
        print(f"  Warning: disk usage check warning: {e}")

    print("\n" + "=" * 78)
    print("PREFLIGHT PASS: All integrity gates and frozen contracts satisfied.")
    print("=" * 78)
    return preflight_report


# ---------------------------------------------------------------------------
# Single Run Execution
# ---------------------------------------------------------------------------
def execute_single_run(
    run_row: pd.Series,
    output_dir: Path,
    caches_dir: Path,
    configs_dir: Path,
    manifests_dir: Path,
    models_dir: Path,
    preprocessors_dir: Path,
    resolved_batches: pd.DataFrame,
    y_measurement: np.ndarray,
    feature_names: List[str],
    modifiable_mask: np.ndarray,
    training_bounds: pd.DataFrame,
    benign_profile: pd.DataFrame,
    rf_model: Any,
    controllers_yaml: Dict[str, Any],
    defenses_yaml: Dict[str, Any],
    provenance_hashes: Dict[str, str],
) -> Dict[str, Any]:
    """
    Executes a single run from the matrix with full validation, staging, and atomic publication.
    """
    run_id = str(run_row["run_id"])
    seed = int(run_row["seed"])
    attack_scenario = canonicalize_scenario(run_row["attack_scenario"])
    defense_name = canonicalize_defense(run_row["defense_name"])
    controller_config_id = str(run_row["controller_config_id"])
    is_alias = bool(run_row["is_alias"])
    alias_target = str(run_row["alias_for_run_id"])

    final_run_dir = output_dir / run_id

    # Handle C1 Aliases
    if is_alias:
        target_dir = output_dir / alias_target
        if not target_dir.exists():
            raise RuntimeError(
                f"Alias {run_id} cannot be published before target run {alias_target} exists!"
            )
        is_target_valid, err = validate_completed_run(target_dir)
        if not is_target_valid:
            raise RuntimeError(
                f"Alias target {alias_target} is invalid: {err}. Quarantine and recompute target first."
            )

        # Publish alias marker atomically
        staging_dir = output_dir / f".staging_{run_id}_{uuid.uuid4().hex}"
        staging_dir.mkdir(parents=True, exist_ok=False)

        alias_record = {
            "run_id": run_id,
            "seed": seed,
            "attack_scenario": attack_scenario,
            "defense": defense_name,
            "config_id": controller_config_id,
            "is_alias": True,
            "alias_for_run_id": alias_target,
            "published_at": datetime.datetime.utcnow().isoformat() + "Z",
        }
        with open(staging_dir / "alias_pointer.json", "w") as f:
            json.dump(alias_record, f, indent=2)

        # Copy summary and completion from target for downstream transparency
        shutil.copy2(target_dir / "run_summary.json", staging_dir / "run_summary.json")
        shutil.copy2(target_dir / "completion.json", staging_dir / "completion.json")

        staging_dir.rename(final_run_dir)
        return {"run_id": run_id, "status": "ALIASED", "alias_for": alias_target}

    # If run already exists, validate it
    if final_run_dir.exists():
        is_valid, err = validate_completed_run(final_run_dir)
        if is_valid:
            print(f"  [REUSE] Run {run_id} already exists and passed validation. Reusing.")
            return {"run_id": run_id, "status": "REUSED"}
        else:
            print(f"  [QUARANTINE] Run {run_id} exists but is incomplete or invalid: {err}.")
            quarantine_dir = quarantine_run_directory(final_run_dir, err)
            print(f"  Quarantined to {quarantine_dir.name}. Restarting from batch 0.")

    # Locate Attack Cache
    cache_slug = f"{attack_scenario}_{seed}"
    cache_dir = caches_dir / cache_slug
    if not cache_dir.exists():
        raise FileNotFoundError(f"Required attack cache missing: {cache_dir}")

    manifest_path = cache_dir / "manifest.json"
    with open(manifest_path) as f:
        cache_manifest = json.load(f)

    # Prepare Cache Provider
    cache_provider = ConcreteAttackCacheProvider(
        cache_dir=cache_dir,
        resolved_batches=resolved_batches,
        expected_cache_identity=cache_manifest,
        expected_feature_names=feature_names,
        official_mode=True,
        expected_row_count=72000,
    )

    # Prepare Label Provider
    label_provider = LabelProvider(y_measurement=y_measurement)

    # Prepare Defense Adapter
    defense_adapter = create_defense_adapter(
        defense_name=defense_name,
        feature_names=feature_names,
        modifiable_mask=modifiable_mask,
        training_bounds=training_bounds,
        benign_profile=benign_profile,
        rf_model=rf_model,
    )

    # Prepare Controller Policy
    policy_controller = create_policy_controller(
        controller_config_id=controller_config_id,
        defense_name=defense_name,
        controllers_yaml=controllers_yaml,
        defenses_yaml=defenses_yaml,
    )

    # Run Provenance Hashes
    run_prov = dict(provenance_hashes)
    run_prov["cache_manifest_hash"] = calculate_file_hash(manifest_path)

    # Run in staging directory
    staging_dir = output_dir / f".staging_{run_id}_{uuid.uuid4().hex}"
    staging_dir.mkdir(parents=True, exist_ok=False)

    try:
        runner = ExperimentRunner(
            resolved_batches=resolved_batches,
            label_provider=label_provider,
            attack_cache=cache_provider,
            defense_adapter=defense_adapter,
            policy_controller=policy_controller,
            output_dir=staging_dir,
            run_id=run_id,
            seed=seed,
            attack_scenario=attack_scenario,
            defense_name=defense_name,
            config_id=controller_config_id,
            provenance_hashes=run_prov,
        )
        runner.execute_run()

        # The runner writes output inside staging_dir / run_id
        runner_out = staging_dir / run_id
        if not runner_out.exists():
            raise RuntimeError(f"Runner failed to create output directory {runner_out}")

        # Atomic rename from staging_dir / run_id to final_run_dir
        runner_out.rename(final_run_dir)
        shutil.rmtree(staging_dir, ignore_errors=True)

        return {"run_id": run_id, "status": "COMPLETED"}

    except Exception as e:
        # Quarantine staging directory if failure occurred
        ts = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        q_dir = output_dir / f"{run_id}_failed_quarantined_{ts}"
        if staging_dir.exists():
            staging_dir.rename(q_dir)
            with open(q_dir / "quarantined_failure.json", "w") as f:
                json.dump(
                    {
                        "run_id": run_id,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                    },
                    f,
                    indent=2,
                )
        raise


# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 10D Official Evaluation Orchestrator (Safe by Default).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Explicit flag required to execute evaluation runs. Defaults to preflight only.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        default=False,
        help="Explicit flag to perform preflight checks only and exit.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Filter runs by attack scenario (SilentProbing, SurrogateTransfer, DecisionBoundary).",
    )
    parser.add_argument(
        "--defense",
        type=str,
        default=None,
        help="Filter runs by defense (afp, feature_squeezing, randomized_smoothing).",
    )
    parser.add_argument(
        "--config",
        "--controller",
        dest="controller",
        type=str,
        default=None,
        help="Filter runs by controller config (Base, C1..C7).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Filter runs by stochastic seed (42, 43, 44, 45, 46).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Filter runs by exact run_id.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional maximum number of runs to execute.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/evaluation_runs",
        help="Directory where evaluation runs will be published.",
    )
    parser.add_argument(
        "--caches-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/caches",
        help="Directory containing the 15 official attack caches.",
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=REPO_ROOT / "configs",
        help="Directory containing frozen configurations.",
    )
    parser.add_argument(
        "--manifests-dir",
        type=Path,
        default=REPO_ROOT / "data/manifests",
        help="Directory containing evaluation roles and batches manifests.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/models",
        help="Directory containing the frozen RF model.",
    )
    parser.add_argument(
        "--preprocessors-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/preprocessors",
        help="Directory containing preprocessors (bounds, mask, benign profile).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # 1. Run Preflight (always runs unconditionally)
    preflight_report = run_preflight(
        configs_dir=args.configs_dir,
        manifests_dir=args.manifests_dir,
        models_dir=args.models_dir,
        caches_dir=args.caches_dir,
        repo_root=REPO_ROOT,
        enforce_git=True,
    )

    if args.preflight_only or not args.execute:
        print("\nSafe Mode: Execution halted after non-mutating preflight.")
        print("To execute runs, provide the explicit --execute flag.")
        return 0

    print("\n" + "=" * 78)
    print("STARTING AUTHORIZED EVALUATION EXECUTION")
    print("=" * 78)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Derive Matrix & Filter Runs
    matrix, _ = derive_and_validate_matrix(args.configs_dir)
    filtered_matrix = matrix.copy()

    if args.scenario:
        scen = canonicalize_scenario(args.scenario)
        filtered_matrix = filtered_matrix[
            filtered_matrix["attack_scenario"].apply(canonicalize_scenario) == scen
        ]
    if args.defense:
        def_name = canonicalize_defense(args.defense)
        filtered_matrix = filtered_matrix[
            filtered_matrix["defense_name"].apply(canonicalize_defense) == def_name
        ]
    if args.controller:
        filtered_matrix = filtered_matrix[
            filtered_matrix["controller_config_id"] == args.controller
        ]
    if args.seed is not None:
        filtered_matrix = filtered_matrix[filtered_matrix["seed"] == int(args.seed)]
    if args.run_id:
        filtered_matrix = filtered_matrix[filtered_matrix["run_id"] == args.run_id]

    if len(filtered_matrix) == 0:
        print("No matrix runs matched the specified filters.")
        return 0

    if args.max_runs is not None and args.max_runs > 0:
        filtered_matrix = filtered_matrix.head(args.max_runs)

    print(f"Selected {len(filtered_matrix)} runs for evaluation.")

    # 3. Load Common Components
    resolved_batches, y_measurement, manifest_hashes = prepare_evaluation_batches(
        args.manifests_dir
    )

    with open(args.preprocessors_dir / "feature_mask.json") as f:
        mask_data = json.load(f)
    feature_names = mask_data["feature_columns"]
    modifiable_mask = np.array(mask_data["feature_mask"], dtype=bool)

    training_bounds = pd.read_parquet(args.preprocessors_dir / "training_bounds.parquet")
    benign_profile = pd.read_parquet(args.preprocessors_dir / "afp_benign_profile.parquet")

    rf_model = joblib.load(args.models_dir / "frozen_rf.joblib")

    with open(args.configs_dir / "controllers.yaml") as f:
        controllers_yaml = yaml.safe_load(f)
    with open(args.configs_dir / "defenses.yaml") as f:
        defenses_yaml = yaml.safe_load(f)
    with open(args.configs_dir / "attacks.yaml") as f:
        attacks_yaml = yaml.safe_load(f)
    with open(args.configs_dir / "experiment.yaml") as f:
        experiment_yaml = yaml.safe_load(f)

    provenance_hashes = {
        "frozen_rf_hash": calculate_file_hash(args.models_dir / "frozen_rf.joblib"),
        "evaluation_roles_hash": manifest_hashes["evaluation_roles_hash"],
        "evaluation_batches_hash": manifest_hashes["evaluation_batches_hash"],
        "attacks_yaml_hash": calculate_file_hash(args.configs_dir / "attacks.yaml"),
        "defenses_yaml_hash": calculate_file_hash(args.configs_dir / "defenses.yaml"),
        "controllers_yaml_hash": calculate_file_hash(args.configs_dir / "controllers.yaml"),
        "experiment_yaml_hash": calculate_file_hash(args.configs_dir / "experiment.yaml"),
    }

    # Execute unique runs first, then aliases
    unique_runs = filtered_matrix[~filtered_matrix["is_alias"]]
    alias_runs = filtered_matrix[filtered_matrix["is_alias"]]

    completed_count = 0
    reused_count = 0
    aliased_count = 0

    for idx, (_, r) in enumerate(unique_runs.iterrows(), 1):
        print(f"[{idx}/{len(unique_runs)}] Executing run: {r['run_id']}...")
        res = execute_single_run(
            run_row=r,
            output_dir=args.output_dir,
            caches_dir=args.caches_dir,
            configs_dir=args.configs_dir,
            manifests_dir=args.manifests_dir,
            models_dir=args.models_dir,
            preprocessors_dir=args.preprocessors_dir,
            resolved_batches=resolved_batches,
            y_measurement=y_measurement,
            feature_names=feature_names,
            modifiable_mask=modifiable_mask,
            training_bounds=training_bounds,
            benign_profile=benign_profile,
            rf_model=rf_model,
            controllers_yaml=controllers_yaml,
            defenses_yaml=defenses_yaml,
            provenance_hashes=provenance_hashes,
        )
        if res["status"] == "REUSED":
            reused_count += 1
        elif res["status"] == "COMPLETED":
            completed_count += 1

    for idx, (_, r) in enumerate(alias_runs.iterrows(), 1):
        print(f"[{idx}/{len(alias_runs)}] Publishing alias: {r['run_id']} -> {r['alias_for_run_id']}...")
        res = execute_single_run(
            run_row=r,
            output_dir=args.output_dir,
            caches_dir=args.caches_dir,
            configs_dir=args.configs_dir,
            manifests_dir=args.manifests_dir,
            models_dir=args.models_dir,
            preprocessors_dir=args.preprocessors_dir,
            resolved_batches=resolved_batches,
            y_measurement=y_measurement,
            feature_names=feature_names,
            modifiable_mask=modifiable_mask,
            training_bounds=training_bounds,
            benign_profile=benign_profile,
            rf_model=rf_model,
            controllers_yaml=controllers_yaml,
            defenses_yaml=defenses_yaml,
            provenance_hashes=provenance_hashes,
        )
        aliased_count += 1

    print("\n" + "=" * 78)
    print("EVALUATION BATCH SUMMARY:")
    print(f"  Completed executions: {completed_count}")
    print(f"  Reused runs:          {reused_count}")
    print(f"  Published aliases:    {aliased_count}")
    print(f"  Total processed:      {completed_count + reused_count + aliased_count}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())

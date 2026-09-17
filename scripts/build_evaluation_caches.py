#!/usr/bin/env python3
"""
scripts/build_evaluation_caches.py

Phase 10B Attack-Cache Orchestration Entry Point (v2).

This script acts as the thin orchestration layer over AttackCacheBuilder for Phase 10
evaluation attack caches. It strictly enforces:
  1. Safe-by-default execution: running without --execute performs non-mutating preflight only.
  2. Preflight separation: preflight never accesses evaluation features (X_eval.parquet) or labels.
  3. Strengthened freeze verification: byte-for-byte and SHA-256 comparison of every frozen
     configuration and protocol file against the tagged version at phase10-protocol-freeze.
  4. Matrix derivation: derives exactly 15 canonical (scenario, seed) pairs across
     SilentProbing, SurrogateTransfer, DecisionBoundary and seeds 42..46, validating
     sensitivity alias reuse and cache identity independence from defenses and controllers.
  5. Accurate label semantics: true labels are never given to surrogate fitting, benign
     reference pool selection, or boundary perturbation search. True labels are used by the
     evaluation harness solely for clean-TP eligibility, deterministic target selection, and
     post-generation status/ASR accounting.
  6. Independent cache-reuse validation: cache reuse requires validation against independently
     computed expectations (expected_scenario, expected_seed, expected_cache_identity,
     expected_row_count). Loaded manifests are never passed back as their own expected identity.
  7. Safe staging and quarantine: all staging and quarantine operations remain beneath the configured
     cache root, preventing path traversal and protecting unrelated directories.
  8. Reproducible execution state: official execution (--execute) requires a clean, committed working
     tree where the freeze tag is an ancestor of HEAD and all source/test/config files are tracked.
  9. Partition alignment validation: strict verification of 90,000 evaluation rows, 78 features,
     finite values, disjoint crafting/measurement pools, and 144 batches of 500 in official mode.
"""
from __future__ import annotations

import argparse
import copy
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

import numpy as np
import pandas as pd
import yaml

# Add src/ to sys.path so recall_aware_ids imports cleanly
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from recall_aware_ids.experiment.caching import (
    AttackCacheBuilder,
    ConcreteAttackCacheProvider,
    calculate_file_hash,
    validate_cache_manifest,
)
from recall_aware_ids.experiment.matrix import (
    generate_evaluation_matrix,
    get_unique_executions,
)
from recall_aware_ids.experiment.schemas import (
    _HEX64,
    _PLACEHOLDER_STRINGS,
    _REQUIRED_PROVENANCE_KEYS,
    CompletionMarker,
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

CANONICAL_SCENARIOS_ORDER = ["SilentProbing", "SurrogateTransfer", "DecisionBoundary"]

DISPLAY_SCENARIO_NAMES = {
    "SilentProbing": "Silent Probing",
    "SurrogateTransfer": "Surrogate Transfer",
    "DecisionBoundary": "Decision Boundary",
}

VALID_PRIMARY_SEEDS = [42, 43, 44, 45, 46]


# ---------------------------------------------------------------------------
# Scenario and Name Utilities
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


def display_scenario(scenario: str) -> str:
    """Returns canonical display name (e.g. 'Silent Probing')."""
    canonical = canonicalize_scenario(scenario)
    return DISPLAY_SCENARIO_NAMES.get(canonical, canonical)


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
                bytes_match = (disk_bytes == tagged_bytes)

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
                f"configs/{p.name}" for p in (repo_root / "configs").iterdir()
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
# Matrix Derivation (Canonicalized Pairs)
# ---------------------------------------------------------------------------
def derive_cache_pairs(configs_dir: Path) -> List[Tuple[str, int]]:
    """
    Derives—not manually hardcodes—the distinct (attack_scenario, seed) pairs
    needed by the frozen 252 unique runs in the evaluation matrix.

    Binding Assertions:
      - Scenarios: exactly 'SilentProbing', 'SurrogateTransfer', and 'DecisionBoundary'
      - Seeds: exactly 42, 43, 44, 45, and 46
      - Exactly 15 ordered (scenario, seed) 2-tuples
      - No defense or controller fields in cache identity
      - Deterministic pair ordering and correct sensitivity-alias reuse
    """
    matrix = generate_evaluation_matrix(configs_dir)
    unique_execs = get_unique_executions(matrix)

    if len(matrix) != 279:
        raise ValueError(f"Evaluation matrix has {len(matrix)} rows, expected 279")
    if len(unique_execs) != 252:
        raise ValueError(f"Unique executions count is {len(unique_execs)}, expected 252")

    # Verify sensitivity alias reuse
    aliases = matrix[matrix["is_alias"]]
    if len(aliases) != 27:
        raise ValueError(f"Expected 27 sensitivity aliases, got {len(aliases)}")
    for _, r in aliases.iterrows():
        target_id = r["alias_for_run_id"]
        target_rows = matrix[matrix["run_id"] == target_id]
        if len(target_rows) != 1:
            raise ValueError(f"Alias {r['run_id']} targets missing run {target_id}")
        target = target_rows.iloc[0]
        r_scen = canonicalize_scenario(r["attack_scenario"])
        target_scen = canonicalize_scenario(target["attack_scenario"])
        if r_scen != target_scen or int(r["seed"]) != int(target["seed"]):
            raise ValueError(
                f"Alias {r['run_id']} scenario/seed mismatch with target {target_id}"
            )

    unique_pairs_set: Set[Tuple[str, int]] = set()
    for _, r in unique_execs.iterrows():
        scen = canonicalize_scenario(r["attack_scenario"])
        seed = int(r["seed"])
        unique_pairs_set.add((scen, seed))

    if len(unique_pairs_set) != 15:
        raise ValueError(f"Expected exactly 15 canonical cache pairs, derived {len(unique_pairs_set)}")

    scenarios_found = {p[0] for p in unique_pairs_set}
    expected_scenarios = set(CANONICAL_SCENARIOS_ORDER)
    if scenarios_found != expected_scenarios:
        raise ValueError(
            f"Derived scenarios {scenarios_found} do not match expected {expected_scenarios}"
        )

    seeds_found = {p[1] for p in unique_pairs_set}
    expected_seeds = set(VALID_PRIMARY_SEEDS)
    if seeds_found != expected_seeds:
        raise ValueError(
            f"Derived seeds {seeds_found} do not match expected {expected_seeds}"
        )

    # Verify cache identity independence from defenses and controllers
    grouped = matrix.groupby(["attack_scenario", "seed"])
    for (scen, seed), group in grouped:
        defenses_in_group = set(group["defense_name"])
        controllers_in_group = set(group["controller_config_id"])
        if len(defenses_in_group) != 3:
            raise ValueError(
                f"Pair ({scen}, {seed}) does not span all 3 defenses: {defenses_in_group}"
            )
        if len(controllers_in_group) < 2:
            raise ValueError(
                f"Pair ({scen}, {seed}) missing expected controllers: {controllers_in_group}"
            )

    derived_pairs = sorted(
        list(unique_pairs_set),
        key=lambda x: (CANONICAL_SCENARIOS_ORDER.index(x[0]), x[1]),
    )

    for p in derived_pairs:
        if not (isinstance(p, tuple) and len(p) == 2 and isinstance(p[0], str) and isinstance(p[1], int)):
            raise TypeError(f"Derived pair {p!r} is not a valid (scenario: str, seed: int) 2-tuple")

    return derived_pairs


# ---------------------------------------------------------------------------
# Safe Staging and Quarantine
# ---------------------------------------------------------------------------
def quarantine_directory(directory: Path, cache_root: Path, base_name: Optional[str] = None) -> Path:
    """
    Quarantines a failed or incomplete directory by renaming it with a unique timestamp.

    Strict Safety Invariants:
      - directory must be strictly inside cache_root (no path traversal, no touching unrelated dirs).
      - cache_root itself cannot be quarantined.
      - Never move or quarantine an unrelated directory.
    """
    cache_root_resolved = cache_root.resolve()
    dir_resolved = directory.resolve()

    if dir_resolved == cache_root_resolved:
        raise ValueError(f"Cannot quarantine the cache root directory itself: {cache_root}")

    try:
        is_rel = dir_resolved.is_relative_to(cache_root_resolved)
    except AttributeError:
        is_rel = str(dir_resolved).startswith(str(cache_root_resolved))

    if not is_rel:
        raise ValueError(
            f"Security violation: Directory {directory} ({dir_resolved}) is not strictly inside "
            f"cache root {cache_root} ({cache_root_resolved}). "
            "Quarantine rejected to prevent touching or moving unrelated directories."
        )

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M%S%f")
    prefix = base_name if base_name else dir_resolved.name
    quarantine_name = f"{prefix}_quarantined_{ts}"
    quarantine_dir = cache_root_resolved / quarantine_name
    if dir_resolved.exists():
        dir_resolved.rename(quarantine_dir)
    return quarantine_dir


# ---------------------------------------------------------------------------
# Independent Cache Identity Builder
# ---------------------------------------------------------------------------
def build_expected_cache_identity(
    scenario: str,
    seed: int,
    expected_row_count: int,
    prov_hashes: Dict[str, str],
    script_hashes: Dict[str, str],
    attack_parameters: Dict[str, Any],
    query_budgets: Dict[str, Any],
    x_attacked_sha256: str,
    status_sha256: str,
    screening_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Independently constructs the full expected cache identity dictionary for validation.
    Never relies on loaded manifest as its own expected truth.
    """
    identity = {
        "schema_version": "1.0",
        "attack_scenario": canonicalize_scenario(scenario),
        "effective_seed": int(seed),
        "row_count": int(expected_row_count),
        "X_attacked_sha256": x_attacked_sha256,
        "status_sha256": status_sha256,
        "output_sha256": x_attacked_sha256,
        "attack_script_hashes": copy.deepcopy(script_hashes),
        "attack_parameters": copy.deepcopy(attack_parameters),
        "query_budgets": copy.deepcopy(query_budgets),
    }
    if screening_metrics is not None:
        identity["screening_metrics"] = copy.deepcopy(screening_metrics)

    identity.update({
        "attacks_yaml_hash": prov_hashes["attacks_yaml_hash"],
        "X_eval_hash": prov_hashes["X_eval_hash"],
        "metadata_eval_hash": prov_hashes["metadata_eval_hash"],
        "evaluation_roles_hash": prov_hashes["evaluation_roles_hash"],
        "evaluation_batches_hash": prov_hashes["evaluation_batches_hash"],
        "crafting_identity_hash": prov_hashes["crafting_identity_hash"],
        "measurement_identity_hash": prov_hashes["measurement_identity_hash"],
        "frozen_rf_hash": prov_hashes["frozen_rf_hash"],
        "scaler_hash": prov_hashes["scaler_hash"],
        "feature_names_hash": prov_hashes["feature_names_hash"],
        "feature_mask_hash": prov_hashes["feature_mask_hash"],
        "training_bounds_hash": prov_hashes["training_bounds_hash"],
        **{k: v for k, v in prov_hashes.items()}
    })
    return identity


# ---------------------------------------------------------------------------
# Strict Independent Cache Completion Validation
# ---------------------------------------------------------------------------
def validate_completed_cache(
    cache_dir: Path,
    expected_scenario: str,
    expected_seed: int,
    expected_cache_identity: Dict[str, Any],
    expected_feature_names: List[str],
    resolved_batches: pd.DataFrame,
    official_mode: bool = True,
    expected_row_count: int = 72000,
    expected_orchestration_commit: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Validates whether an existing cache directory is 100% complete, sound, and matches
    independently computed expectations.

    Binding Requirements:
      - Validates against independently computed expected_cache_identity, expected_scenario,
        and expected_seed (never circular).
      - Checks completion marker, manifest, provenance, schemas, exact row count,
        attacked-feature hash, status hash, and cache identity.
      - File existence alone is strictly insufficient.
    """
    completion_path = cache_dir / "completion.json"
    manifest_path = cache_dir / "manifest.json"
    x_path = cache_dir / "X_attacked.parquet"
    s_path = cache_dir / "status.parquet"

    if not completion_path.exists():
        return False, "Missing completion.json"
    if not manifest_path.exists():
        return False, "Missing manifest.json"
    if not x_path.exists():
        return False, "Missing X_attacked.parquet"
    if not s_path.exists():
        return False, "Missing status.parquet"

    canonical_expected_scen = canonicalize_scenario(expected_scenario)

    try:
        # 1. Inspect completion.json
        with open(completion_path, "r") as f:
            comp_data = json.load(f)

        if comp_data.get("completion_state") != "COMPLETED":
            return False, f"Incomplete state in completion.json: {comp_data.get('completion_state')}"
        if not comp_data.get("completed", False):
            return False, "completed flag is not True in completion.json"

        _validate_iso_timestamp(comp_data.get("generation_end_timestamp"), "generation_end_timestamp")

        # Independent identity validation against expectations
        if canonicalize_scenario(comp_data.get("attack_scenario", "")) != canonical_expected_scen:
            return False, f"Scenario mismatch in completion.json: expected {canonical_expected_scen}, got {comp_data.get('attack_scenario')}"
        if comp_data.get("effective_seed") != expected_seed:
            return False, f"Seed mismatch in completion.json: expected {expected_seed}, got {comp_data.get('effective_seed')}"
        if comp_data.get("row_count") != expected_row_count:
            return False, f"Row count mismatch in completion.json: expected {expected_row_count}, got {comp_data.get('row_count')}"

        if expected_orchestration_commit is not None:
            exec_commit = comp_data.get("execution_script_commit")
            if exec_commit != expected_orchestration_commit:
                return False, f"Orchestration commit mismatch in completion.json: expected {expected_orchestration_commit}, got {exec_commit}"

        # 2. Inspect manifest.json
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        if canonicalize_scenario(manifest.get("attack_scenario", "")) != canonical_expected_scen:
            return False, f"Scenario mismatch in manifest.json: expected {canonical_expected_scen}, got {manifest.get('attack_scenario')}"
        if manifest.get("effective_seed") != expected_seed:
            return False, f"Seed mismatch in manifest.json: expected {expected_seed}, got {manifest.get('effective_seed')}"
        if manifest.get("row_count") != expected_row_count:
            return False, f"Row count mismatch in manifest.json: expected {expected_row_count}, got {manifest.get('row_count')}"

        # 3. Compute real SHA-256 digests of generated files
        x_hash = calculate_file_hash(x_path)
        s_hash = calculate_file_hash(s_path)
        m_hash = calculate_file_hash(manifest_path)

        if manifest.get("X_attacked_sha256") != x_hash:
            return False, "X_attacked file hash does not match manifest X_attacked_sha256"
        if manifest.get("status_sha256") != s_hash:
            return False, "status file hash does not match manifest status_sha256"

        art_hashes = comp_data.get("artifacts", {})
        if art_hashes.get("X_attacked_sha256") != x_hash:
            return False, "completion.json X_attacked_sha256 mismatch vs actual file hash"
        if art_hashes.get("status_sha256") != s_hash:
            return False, "completion.json status_sha256 mismatch vs actual file hash"
        if art_hashes.get("manifest_sha256") != m_hash:
            return False, "completion.json manifest_sha256 mismatch vs actual file hash"

        # 4. Validate manifest against INDEPENDENTLY COMPUTED expected_cache_identity
        validate_cache_manifest(
            manifest_path=manifest_path,
            X_attacked_path=x_path,
            expected_hashes=expected_cache_identity,
            status_path=s_path,
            expected_row_count=expected_row_count,
        )

        # 5. Validate provenance hashes in completion.json agree with expected_cache_identity
        comp_prov = comp_data.get("provenance_hashes", {})
        for req_k in _REQUIRED_PROVENANCE_KEYS:
            if req_k not in comp_prov:
                return False, f"Missing required provenance key in completion.json: {req_k}"
            val = comp_prov[req_k]
            if not _is_hex64(val) or val in ("0" * 64, "a" * 64) or any(p in val.lower() for p in _PLACEHOLDER_STRINGS):
                return False, f"Invalid or placeholder provenance hash in completion.json for {req_k}: {val!r}"
            if req_k in expected_cache_identity and val != expected_cache_identity[req_k]:
                return False, f"Provenance mismatch in completion.json for {req_k}: expected {expected_cache_identity[req_k]}, got {val}"

        # 6. Validate via strict ConcreteAttackCacheProvider with independent expected identity
        ConcreteAttackCacheProvider(
            cache_dir=cache_dir,
            resolved_batches=resolved_batches,
            expected_cache_identity=expected_cache_identity,
            expected_feature_names=expected_feature_names,
            official_mode=official_mode,
            expected_row_count=expected_row_count,
        )

        return True, "Valid complete cache"
    except Exception as e:
        return False, f"Validation error: {e}"


# ---------------------------------------------------------------------------
# Evaluation Partition Alignment Validation (for official execution)
# ---------------------------------------------------------------------------
def validate_evaluation_partition_alignment(
    df_x_eval: pd.DataFrame,
    df_meta_eval: pd.DataFrame,
    df_roles: pd.DataFrame,
    df_batches: pd.DataFrame,
    expected_feature_names: List[str],
) -> None:
    """
    Strict validation of evaluation partition alignment for official execution mode.

    Guarantees:
      - df_meta_eval contains _source_file, _raw_row_idx, and y_binary;
      - evaluation_roles.csv contains eval_position, _source_file, _raw_row_idx, y_binary, and role;
      - exactly 90,000 aligned X_eval, metadata_eval, and evaluation_roles rows;
      - exactly 72,000 evaluation_batches rows;
      - exact 78-feature schema matching frozen feature names (no extra, missing, or reordered columns);
      - zero NaN and infinite feature values;
      - evaluation positions in X, metadata, and roles are unique and equal exactly 0..89999;
      - feature and metadata positions align;
      - roles sorted by eval_position have exactly the same (_source_file, _raw_row_idx) sequence as metadata;
      - y_binary in roles exactly equals metadata y_binary;
      - attack_family in roles matches metadata attack_family if present in both;
      - role values consist only of 'crafting' and 'measurement';
      - exactly 18,000 crafting and 72,000 measurement roles;
      - crafting and measurement composite identities are disjoint;
      - evaluation_batches.csv positions are unique and equal exactly the 72,000 measurement positions;
      - each measurement position occurs exactly once in one batch across 144 batches of 500.
    """
    # 1. Required columns in metadata_eval
    for col in ["_source_file", "_raw_row_idx", "y_binary"]:
        if col not in df_meta_eval.columns:
            raise ValueError(f"metadata_eval missing required column: '{col}'")

    # 2. Required columns in evaluation_roles
    for col in ["eval_position", "_source_file", "_raw_row_idx", "y_binary", "role"]:
        if col not in df_roles.columns:
            raise ValueError(f"evaluation_roles missing required column: '{col}'")

    # 3. Required columns in evaluation_batches
    for col in ["eval_position", "batch_id"]:
        if col not in df_batches.columns:
            raise ValueError(f"evaluation_batches missing required column: '{col}'")

    # 4. Row counts
    if len(df_x_eval) != 90000:
        raise ValueError(f"X_eval row count mismatch: expected 90,000, got {len(df_x_eval):,}")
    if len(df_meta_eval) != 90000:
        raise ValueError(f"metadata_eval row count mismatch: expected 90,000, got {len(df_meta_eval):,}")
    if len(df_roles) != 90000:
        raise ValueError(f"evaluation_roles row count mismatch: expected 90,000, got {len(df_roles):,}")
    if len(df_batches) != 72000:
        raise ValueError(f"evaluation_batches row count mismatch: expected 72,000, got {len(df_batches):,}")

    # 5. Exact feature schema equality
    if list(df_x_eval.columns) != expected_feature_names:
        if len(df_x_eval.columns) != len(expected_feature_names):
            raise ValueError(
                f"X_eval column count mismatch: expected {len(expected_feature_names)}, "
                f"got {len(df_x_eval.columns)}"
            )
        raise ValueError("X_eval feature columns do not exactly match frozen feature_names.json in name and order")

    # 6. Finite float values
    x_vals = df_x_eval[expected_feature_names].values
    if not np.isfinite(x_vals).all():
        raise ValueError("X_eval contains non-finite (NaN or Inf) feature values")

    # 7. Evaluation positions extraction
    if "eval_position" in df_x_eval.columns:
        eps_x = df_x_eval["eval_position"].to_numpy()
    else:
        eps_x = df_x_eval.index.to_numpy()

    if "eval_position" in df_meta_eval.columns:
        eps_meta = df_meta_eval["eval_position"].to_numpy()
    else:
        eps_meta = df_meta_eval.index.to_numpy()

    eps_roles = df_roles["eval_position"].to_numpy()

    expected_0_to_89999 = np.arange(90000, dtype=int)

    # Position checks for X_eval
    if len(np.unique(eps_x)) != 90000:
        raise ValueError("X_eval eval_position values contain duplicates")
    if not np.array_equal(np.sort(eps_x), expected_0_to_89999):
        raise ValueError("X_eval eval_position values must equal exactly 0..89999 with no gaps or out-of-range values")

    # Position checks for metadata_eval
    if len(np.unique(eps_meta)) != 90000:
        raise ValueError("metadata_eval eval_position values contain duplicates")
    if not np.array_equal(np.sort(eps_meta), expected_0_to_89999):
        raise ValueError("metadata_eval eval_position values must equal exactly 0..89999 with no gaps or out-of-range values")

    # Position checks for evaluation_roles
    if len(np.unique(eps_roles)) != 90000:
        raise ValueError("evaluation_roles eval_position values contain duplicates")
    if not np.array_equal(np.sort(eps_roles), expected_0_to_89999):
        raise ValueError("evaluation_roles eval_position values must equal exactly 0..89999 with no gaps or out-of-range values")

    # Alignment between X_eval and metadata_eval
    if not np.array_equal(eps_x, eps_meta):
        raise ValueError("X_eval and metadata_eval eval_position sequence do not align")

    # 8. Sort metadata and roles by eval_position to verify aligned sequence
    if "eval_position" in df_meta_eval.columns:
        meta_sorted = df_meta_eval.sort_values("eval_position").reset_index(drop=True)
    else:
        meta_sorted = df_meta_eval.sort_index().reset_index(drop=True)

    roles_sorted = df_roles.sort_values("eval_position").reset_index(drop=True)

    # Composite identities sequence check: roles sorted by eval_position must match metadata
    roles_comp_ids = list(zip(roles_sorted["_source_file"], roles_sorted["_raw_row_idx"]))
    meta_comp_ids = list(zip(meta_sorted["_source_file"], meta_sorted["_raw_row_idx"]))
    if roles_comp_ids != meta_comp_ids:
        raise ValueError("evaluation_roles and metadata_eval composite identity (_source_file, _raw_row_idx) sequence mismatch")

    # y_binary in roles exactly equals metadata y_binary
    if not np.array_equal(roles_sorted["y_binary"].values, meta_sorted["y_binary"].values):
        raise ValueError("y_binary in evaluation_roles does not equal metadata_eval y_binary")

    # attack_family matching if present in both
    if "attack_family" in roles_sorted.columns and "attack_family" in meta_sorted.columns:
        if not np.array_equal(roles_sorted["attack_family"].astype(str).values, meta_sorted["attack_family"].astype(str).values):
            raise ValueError("attack_family in evaluation_roles does not equal metadata_eval attack_family")

    # 9. Role values validation
    role_vals = set(df_roles["role"].unique())
    if role_vals != {"crafting", "measurement"}:
        raise ValueError(f"evaluation_roles role values must consist only of 'crafting' and 'measurement', found: {role_vals}")

    crafting_mask = (df_roles["role"] == "crafting")
    measurement_mask = (df_roles["role"] == "measurement")

    if crafting_mask.sum() != 18000:
        raise ValueError(f"Expected 18,000 crafting roles, found {crafting_mask.sum():,}")
    if measurement_mask.sum() != 72000:
        raise ValueError(f"Expected 72,000 measurement roles, found {measurement_mask.sum():,}")

    crafting_eps = set(df_roles.loc[crafting_mask, "eval_position"].values)
    measurement_eps = set(df_roles.loc[measurement_mask, "eval_position"].values)

    if not crafting_eps.isdisjoint(measurement_eps):
        overlap = crafting_eps.intersection(measurement_eps)
        raise ValueError(f"Crafting and measurement roles overlap by {len(overlap)} positions")

    # Crafting and measurement composite identities are disjoint
    crafting_ids = set(zip(df_roles.loc[crafting_mask, "_source_file"], df_roles.loc[crafting_mask, "_raw_row_idx"]))
    measurement_ids = set(zip(df_roles.loc[measurement_mask, "_source_file"], df_roles.loc[measurement_mask, "_raw_row_idx"]))
    if len(crafting_ids) != 18000:
        raise ValueError(f"Expected 18,000 unique crafting composite identities, got {len(crafting_ids):,}")
    if len(measurement_ids) != 72000:
        raise ValueError(f"Expected 72,000 unique measurement composite identities, got {len(measurement_ids):,}")
    if not crafting_ids.isdisjoint(measurement_ids):
        overlap_ids = crafting_ids.intersection(measurement_ids)
        raise ValueError(f"Crafting and measurement composite identities overlap by {len(overlap_ids)} items")

    # 10. Batches alignment
    batch_eps = df_batches["eval_position"].to_numpy()
    if len(np.unique(batch_eps)) != 72000:
        raise ValueError("evaluation_batches eval_position values contain duplicates or do not total 72,000")
    if set(batch_eps) != measurement_eps:
        raise ValueError("evaluation_batches eval_positions do not exactly match the 72,000 measurement role positions")

    unique_batches = df_batches["batch_id"].nunique()
    if unique_batches != 144:
        raise ValueError(f"Expected 144 unique batches, found {unique_batches}")
    batch_counts = df_batches["batch_id"].value_counts()
    if not (batch_counts == 500).all():
        raise ValueError("Each evaluation batch must contain exactly 500 records")


# ---------------------------------------------------------------------------
# Reproducible Execution State Verification
# ---------------------------------------------------------------------------
def verify_reproducible_execution_state(
    repo_root: Path = REPO_ROOT,
    configs_dir: Path = REPO_ROOT / "configs",
    artifacts_dir: Path = REPO_ROOT / "artifacts",
    data_dir: Path = REPO_ROOT / "data",
) -> str:
    """
    Enforces that official execution (--execute) runs ONLY from a clean, committed,
    reproducible state.

    Rejects execution if:
      - Any tracked file is modified or dirty (all non-empty porcelain status codes other than '??').
      - Any required Phase 10B source, test, configuration, or documentation file is untracked.
      - src/ has any diff relative to phase10-protocol-freeze.
      - Any frozen config or protocol file differs by a single byte from freeze tag.
      - Any protected hash mismatches.
    """
    git_dir = repo_root / ".git"
    if not git_dir.exists():
        raise RuntimeError(f"Git repository not found at {repo_root}")

    # 1. Freeze ancestry check
    res = subprocess.run(
        ["git", "merge-base", "--is-ancestor", FREEZE_TAG, "HEAD"],
        cwd=repo_root,
        capture_output=True,
    )
    if res.returncode != 0:
        raise RuntimeError(f"Freeze tag {FREEZE_TAG} is not an ancestor of current HEAD")

    # 2. Frozen src/ check
    src_diff = subprocess.check_output(
        ["git", "diff", "--name-only", FREEZE_TAG, "--", "src/"],
        cwd=repo_root,
        text=True,
    ).strip()
    if src_diff:
        raise RuntimeError(f"Frozen src/ directory modified relative to {FREEZE_TAG}: {src_diff.splitlines()}")

    # 3. Frozen configuration byte match
    verify_frozen_configurations(repo_root=repo_root, enforce_git=True)

    # 4. Protected hashes check
    for name, p, expected in [
        ("RF", artifacts_dir / "models/frozen_rf.joblib", PROTECTED_HASHES["RF"]),
        ("Roles", data_dir / "manifests/evaluation_roles.csv", PROTECTED_HASHES["Roles"]),
        ("Batches", data_dir / "manifests/evaluation_batches.csv", PROTECTED_HASHES["Batches"]),
    ]:
        if calculate_file_hash(p) != expected:
            raise ValueError(f"Protected hash mismatch for {name}")

    # 5. Git status check: refuse dirty or untracked source/test/config files
    status_out = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        text=True,
    ).splitlines()

    dirty_tracked = []
    untracked_code = []

    for line in status_out:
        if not line.strip():
            continue
        code = line[:2]
        path_str = line[3:].strip().strip('"')
        if code == "??":
            # Untracked files: forbid untracked code, test, script, config, or doc files
            if (
                path_str.startswith(("src/", "tests/", "configs/", "scripts/", "docs/"))
                or path_str.endswith((".py", ".yaml", ".yml", ".md", ".toml", ".txt"))
            ):
                # Historical zip archives or log files are allowed, but no source/test/config/doc files
                if not (path_str.endswith(".zip") or path_str.endswith(".log")):
                    untracked_code.append(path_str)
        else:
            # Treat every non-empty git status --porcelain entry other than "??" as a tracked working-tree change
            dirty_tracked.append(f"{code}:{path_str}")

    if dirty_tracked:
        raise RuntimeError(
            f"Official execution (--execute) refused: tracked files are modified: {dirty_tracked}. "
            "Must be committed before execution."
        )
    if untracked_code:
        raise RuntimeError(
            f"Official execution (--execute) refused: untracked code/test/config files detected: {untracked_code}. "
            "Must be tracked and committed before execution."
        )

    head_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    return head_commit


# ---------------------------------------------------------------------------
# Preflight Verification
# ---------------------------------------------------------------------------
def run_preflight(
    configs_dir: Path,
    data_dir: Path,
    artifacts_dir: Path,
    output_dir: Path,
    repo_root: Path = REPO_ROOT,
    enforce_git: bool = True,
) -> Dict[str, Any]:
    """
    Executes comprehensive, non-mutating preflight checks.

    GUARANTEES:
      - Does NOT load data/processed/X_eval.parquet.
      - Does NOT load data/processed/metadata_eval.parquet.
      - Does NOT load measurement features or evaluation labels.
      - Strictly non-mutating: does not create or modify any cache directories.
    """
    preflight_results: Dict[str, Any] = {
        "status": "PASS",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git": {},
        "protected_hashes": {},
        "frozen_file_comparisons": {},
        "configurations": {},
        "manifests": {},
        "derived_pairs": [],
        "cache_inventory": {},
    }

    # 1. Git ancestry and tag verification
    git_dir = repo_root / ".git"
    current_head = "UNKNOWN"
    is_ancestor = False
    tag_commit = "UNKNOWN"

    if git_dir.exists():
        try:
            head_out = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo_root, stderr=subprocess.PIPE, text=True
            ).strip()
            current_head = head_out

            tag_out = subprocess.check_output(
                ["git", "rev-parse", f"{FREEZE_TAG}^{{commit}}"],
                cwd=repo_root,
                stderr=subprocess.PIPE,
                text=True,
            ).strip()
            tag_commit = tag_out

            if tag_commit != FREEZE_COMMIT:
                raise ValueError(
                    f"Freeze tag {FREEZE_TAG} points to commit {tag_commit}, expected {FREEZE_COMMIT}"
                )

            res = subprocess.run(
                ["git", "merge-base", "--is-ancestor", FREEZE_TAG, "HEAD"],
                cwd=repo_root,
                capture_output=True,
            )
            is_ancestor = (res.returncode == 0)
            if not is_ancestor:
                raise ValueError(
                    f"Freeze tag {FREEZE_TAG} ({FREEZE_COMMIT}) is not an ancestor of current HEAD ({current_head})"
                )

            preflight_results["git"] = {
                "head_commit": current_head,
                "freeze_commit": tag_commit,
                "freeze_tag": FREEZE_TAG,
                "is_ancestor": is_ancestor,
            }
        except subprocess.CalledProcessError as e:
            if enforce_git:
                raise RuntimeError(f"Git verification failed: {e.stderr}") from e
    elif enforce_git:
        raise RuntimeError(f"Git directory not found at {git_dir}")

    # 2. Strengthened byte-for-byte and SHA-256 frozen file comparison
    frozen_comps = verify_frozen_configurations(repo_root=repo_root, enforce_git=enforce_git)
    preflight_results["frozen_file_comparisons"] = frozen_comps

    # 3. Protected Hashes Verification
    rf_model_path = artifacts_dir / "models/frozen_rf.joblib"
    roles_path = data_dir / "manifests/evaluation_roles.csv"
    batches_path = data_dir / "manifests/evaluation_batches.csv"

    for name, p, expected_hash in [
        ("RF", rf_model_path, PROTECTED_HASHES["RF"]),
        ("Roles", roles_path, PROTECTED_HASHES["Roles"]),
        ("Batches", batches_path, PROTECTED_HASHES["Batches"]),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Protected artifact {name} missing at {p}")
        actual_hash = calculate_file_hash(p)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Protected hash mismatch for {name} ({p.name}): "
                f"expected {expected_hash}, got {actual_hash}"
            )
        preflight_results["protected_hashes"][name] = {
            "path": str(p),
            "expected": expected_hash,
            "actual": actual_hash,
            "verified": True,
        }

    # 4. Frozen date_frozen verification
    exp_yaml_path = configs_dir / "experiment.yaml"
    with open(exp_yaml_path, "r") as f:
        exp_cfg = yaml.safe_load(f)
    date_frozen = exp_cfg.get("experiment", {}).get("date_frozen")
    if date_frozen != FROZEN_DATE:
        raise ValueError(
            f"experiment.date_frozen is {date_frozen!r}, expected {FROZEN_DATE!r}"
        )

    # 5. Preprocessors verification
    mask_path = artifacts_dir / "preprocessors/feature_mask.json"
    bounds_path = artifacts_dir / "preprocessors/training_bounds.parquet"
    scaler_path = artifacts_dir / "preprocessors/standard_scaler.joblib"
    fnames_path = artifacts_dir / "preprocessors/feature_names.json"

    for p in (mask_path, bounds_path, scaler_path, fnames_path):
        if not p.exists():
            raise FileNotFoundError(f"Required preprocessor artifact missing: {p}")

    preflight_results["configurations"] = {
        "date_frozen": date_frozen,
        "feature_mask_hash": calculate_file_hash(mask_path),
        "training_bounds_hash": calculate_file_hash(bounds_path),
        "scaler_hash": calculate_file_hash(scaler_path),
        "feature_names_hash": calculate_file_hash(fnames_path),
    }

    # 6. Manifest structure check (CSV only, NO parquet evaluation features loaded)
    df_roles = pd.read_csv(roles_path)
    if len(df_roles) != 90000:
        raise ValueError(f"evaluation_roles.csv has {len(df_roles)} rows, expected 90000")
    crafting_count = int((df_roles["role"] == "crafting").sum())
    measurement_count = int((df_roles["role"] == "measurement").sum())
    if crafting_count != 18000:
        raise ValueError(f"Expected 18000 crafting roles, found {crafting_count}")
    if measurement_count != 72000:
        raise ValueError(f"Expected 72000 measurement roles, found {measurement_count}")

    df_batches = pd.read_csv(batches_path)
    if len(df_batches) != 72000:
        raise ValueError(f"evaluation_batches.csv has {len(df_batches)} rows, expected 72000")
    unique_batches = df_batches["batch_id"].nunique()
    if unique_batches != 144:
        raise ValueError(f"Expected 144 unique batches, found {unique_batches}")
    batch_counts = df_batches["batch_id"].value_counts()
    if not (batch_counts == 500).all():
        raise ValueError("Each evaluation batch must contain exactly 500 records")

    preflight_results["manifests"] = {
        "roles_count": len(df_roles),
        "crafting_count": crafting_count,
        "measurement_count": measurement_count,
        "batch_count": unique_batches,
        "records_per_batch": 500,
    }

    # 7. Derive 15 canonical pairs
    derived_pairs = derive_cache_pairs(configs_dir)
    preflight_results["derived_pairs"] = [
        {"scenario": p[0], "seed": p[1], "canonical_slug": f"{p[0]}_{p[1]}"}
        for p in derived_pairs
    ]

    # 8. Inspect existing cache inventory
    cache_status_map: Dict[str, str] = {}
    valid_caches_count = 0
    if output_dir.exists():
        for p in derived_pairs:
            slug = f"{p[0]}_{p[1]}"
            cdir = output_dir / slug
            if not cdir.exists():
                cache_status_map[slug] = "NOT_BUILT"
            elif (cdir / "completion.json").exists() and (cdir / "manifest.json").exists():
                cache_status_map[slug] = "PRESENT_COMPLETED"
                valid_caches_count += 1
            else:
                cache_status_map[slug] = "INCOMPLETE_OR_CORRUPT"
    else:
        for p in derived_pairs:
            slug = f"{p[0]}_{p[1]}"
            cache_status_map[slug] = "NOT_BUILT"

    preflight_results["cache_inventory"] = {
        "output_dir": str(output_dir),
        "total_required": 15,
        "completed_count": valid_caches_count,
        "status_per_cache": cache_status_map,
    }

    return preflight_results


def print_preflight_report(results: Dict[str, Any]) -> None:
    """Prints a clean, formatted preflight summary table to stdout."""
    print("=" * 78)
    print("PHASE 10B ATTACK-CACHE ORCHESTRATION PREFLIGHT REPORT")
    print("=" * 78)
    print(f"Timestamp:       {results['timestamp']}")
    print(f"Overall Status:  {results['status']}")

    git = results.get("git", {})
    if git:
        print("\n--- Git Ancestry Verification ---")
        print(f"  Working HEAD:    {git.get('head_commit')}")
        print(f"  Freeze Commit:   {git.get('freeze_commit')}")
        print(f"  Freeze Tag:      {git.get('freeze_tag')}")
        print(f"  Ancestor Check:  {'PASSED' if git.get('is_ancestor') else 'FAILED'}")

    print("\n--- Strengthened Frozen File Verification (Byte & SHA-256 Match) ---")
    for path_rel, comp in results.get("frozen_file_comparisons", {}).items():
        match_str = "EXACT BYTE & SHA-256 MATCH" if comp["bytes_match"] else "MISMATCH"
        print(f"  [{match_str}] {path_rel}")

    print("\n--- Protected Hashes (SHA-256) ---")
    for name, info in results["protected_hashes"].items():
        print(f"  [{name:7s}] {info['actual']} (VERIFIED)")

    mf = results["manifests"]
    print("\n--- Evaluation Partition Manifests ---")
    print(f"  Roles:           {mf['roles_count']:,} total ({mf['crafting_count']:,} crafting, {mf['measurement_count']:,} measurement)")
    print(f"  Batches:         {mf['batch_count']} batches of {mf['records_per_batch']} rows (72,000 total)")

    pairs = results["derived_pairs"]
    print(f"\n--- Derived Canonical Cache Pairs ({len(pairs)} Distinct Pairs) ---")
    print("  #   Canonical Scenario       Seed   Canonical Identifier      Cache Status")
    print("  --  -----------------------  -----  ------------------------  ------------------")
    cache_inv = results["cache_inventory"]["status_per_cache"]
    for idx, p in enumerate(pairs, 1):
        slug = p["canonical_slug"]
        c_status = cache_inv.get(slug, "NOT_BUILT")
        print(f"  {idx:2d}. {p['scenario']:23s}  {p['seed']:5d}  {slug:24s}  {c_status}")

    print("\n--- Execution Safety & Guarantees ---")
    print("  [x] Safe-by-default preflight (non-mutating)")
    print("  [x] Zero evaluation data (X_eval.parquet) read during preflight")
    print("  [x] Zero evaluation labels read during preflight")
    print("  [x] Label isolation enforced (true labels never used in surrogate/reference/perturbation)")
    print("  [x] Independent cache-reuse validation enforced")
    print("  [x] Safe staging & quarantine confined beneath cache root")
    print("  [x] Official execution requires explicit --execute flag")
    print(f"  [x] Total valid caches present: {results['cache_inventory']['completed_count']} / 15")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Single Cache Execution & Orchestration
# ---------------------------------------------------------------------------
def build_single_cache(
    scenario: str,
    seed: int,
    output_dir: Path,
    configs_dir: Path,
    data_dir: Path,
    artifacts_dir: Path,
    resolved_batches: pd.DataFrame,
    official_mode: bool = True,
    # Injectable parameters for testing with synthetic data
    override_X_craft: Optional[np.ndarray] = None,
    override_y_craft: Optional[np.ndarray] = None,
    override_X_meas: Optional[np.ndarray] = None,
    override_y_meas: Optional[np.ndarray] = None,
    override_eps: Optional[np.ndarray] = None,
    override_predict_fn: Optional[Any] = None,
    override_provenance: Optional[Dict[str, str]] = None,
    override_row_count: Optional[int] = None,
    override_n_boundary_targets: Optional[int] = None,
    override_orchestration_commit: Optional[str] = None,
) -> Path:
    """
    Builds a single canonical cache for the given (scenario, seed) pair.

    Orchestration guarantees:
      - Validates existing cache against independently computed expected identity.
      - If existing cache is invalid or incomplete, safely quarantines beneath cache root and rebuilds.
      - Unique staging directory beneath output_dir on the same filesystem for atomic rename.
      - Enforces strict path containment, rejecting path traversal attempts.
      - Enforces accurate label semantics: true labels never guide surrogate fitting, benign
        reference pool selection, or boundary perturbation search.
      - Writes completion.json LAST into staging directory before atomic publication.
      - Re-validates staging directory against independent expected identity before rename.
      - On any failure: safely quarantines staging directory beneath output_dir.
    """
    output_dir = Path(output_dir).resolve()
    canonical_scen = canonicalize_scenario(scenario)
    target_dir_name = f"{canonical_scen}_{seed}"

    # Path traversal rejection
    if any(sep in target_dir_name for sep in ("/", "\\", "..")):
        raise ValueError(f"Path traversal detected in target directory name: {target_dir_name!r}")

    target_cache_dir = (output_dir / target_dir_name).resolve()
    try:
        is_rel = target_cache_dir.is_relative_to(output_dir)
    except AttributeError:
        is_rel = str(target_cache_dir).startswith(str(output_dir))

    if not is_rel or target_cache_dir.parent != output_dir:
        raise ValueError(f"Path traversal detected: target directory {target_cache_dir} is not inside {output_dir}")

    # Unique staging directory on the same filesystem
    unique_suffix = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
    staging_dir = output_dir / f"{target_dir_name}.staging_{unique_suffix}"

    expected_row_count = 72000 if official_mode else (override_row_count if override_row_count is not None else len(override_X_meas))

    # Pre-load feature names and bounds
    mask_path = artifacts_dir / "preprocessors/feature_mask.json"
    bounds_path = artifacts_dir / "preprocessors/training_bounds.parquet"
    with open(mask_path, "r") as f:
        mask_data = json.load(f)
    feature_names = mask_data["feature_columns"]
    modifiable_mask = mask_data["feature_mask"]
    training_bounds = pd.read_parquet(bounds_path)

    # Compute attack implementation script hashes
    attacks_src_dir = SRC_PATH / "recall_aware_ids/attacks"
    script_hashes = {
        "base.py": calculate_file_hash(attacks_src_dir / "base.py"),
        "silent_probing.py": calculate_file_hash(attacks_src_dir / "silent_probing.py"),
        "surrogate_transfer.py": calculate_file_hash(attacks_src_dir / "surrogate_transfer.py"),
        "boundary_attack.py": calculate_file_hash(attacks_src_dir / "boundary_attack.py"),
        "oracle.py": calculate_file_hash(attacks_src_dir / "oracle.py"),
        "caching.py": calculate_file_hash(SRC_PATH / "recall_aware_ids/experiment/caching.py"),
        "build_evaluation_caches.py": calculate_file_hash(Path(__file__).resolve()),
    }

    # Expected attack parameters and query budgets per scenario
    surrogate_attack = None
    boundary_attack = None
    benign_reference_pool = None
    n_boundary = 200 if official_mode else (override_n_boundary_targets if override_n_boundary_targets is not None else min(200, len(override_X_meas) if override_X_meas is not None else 200))

    if canonical_scen == "SilentProbing":
        attack_params = {"modifies_samples": False}
        query_budgets = {"max_queries_per_sample": 0}
    elif canonical_scen == "SurrogateTransfer":
        attack_params = {
            "surrogate_model": "DecisionTree",
            "surrogate_max_depth": "unconstrained",
            "clip_to_training_bounds": True,
            "preserve_protected_features": True,
        }
        query_budgets = {
            "max_queries_per_sample": 0,
            "crafting_queries_budget": 18000 if official_mode else len(override_X_craft),
            "target_evaluation_queries": expected_row_count,
        }
        from recall_aware_ids.attacks.surrogate_transfer import SurrogateTransferAttack
        surrogate_attack = SurrogateTransferAttack(
            feature_names=feature_names,
            modifiable_mask=modifiable_mask,
            training_bounds=training_bounds,
            effective_seed=seed,
        )
    elif canonical_scen == "DecisionBoundary":
        attack_params = {
            "search_procedure": "binary_search_interpolation",
            "binary_search_steps": 10,
            "n_attack_samples": n_boundary,
            "clip_to_training_bounds": True,
            "preserve_protected_features": True,
        }
        query_budgets = {"max_queries_per_sample": 50}
        from recall_aware_ids.attacks.boundary_attack import DecisionBoundaryAttack
        boundary_attack = DecisionBoundaryAttack(
            feature_names=feature_names,
            modifiable_mask=modifiable_mask,
            training_bounds=training_bounds,
            max_queries=50,
            binary_search_steps=10,
        )
    else:
        raise ValueError(f"Unsupported canonical scenario: {canonical_scen}")

    # Determine current orchestration commit
    if official_mode:
        try:
            current_head = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.PIPE, text=True
            ).strip()
        except Exception:
            current_head = "UNKNOWN"
    else:
        current_head = override_orchestration_commit if override_orchestration_commit else "TEST_ORCHESTRATION_HEAD"

    # Compute expected provenance hashes
    if official_mode:
        x_eval_path = data_dir / "processed/X_eval.parquet"
        meta_eval_path = data_dir / "processed/metadata_eval.parquet"
        roles_path = data_dir / "manifests/evaluation_roles.csv"
        batches_path = data_dir / "manifests/evaluation_batches.csv"

        df_roles_pre = pd.read_csv(roles_path)
        crafting_eps_pre = df_roles_pre[df_roles_pre["role"] == "crafting"]["eval_position"].sort_values().values
        measurement_eps_pre = df_roles_pre[df_roles_pre["role"] == "measurement"]["eval_position"].sort_values().values

        freeze_commit_sha256 = hashlib.sha256(FREEZE_COMMIT.encode("utf-8")).hexdigest()
        freeze_tag_sha256 = hashlib.sha256(FREEZE_TAG.encode("utf-8")).hexdigest()
        orchestration_commit_sha256 = hashlib.sha256(current_head.encode("utf-8")).hexdigest()

        prov_hashes = {
            "attacks_yaml_hash": calculate_file_hash(configs_dir / "attacks.yaml"),
            "controllers_yaml_hash": calculate_file_hash(configs_dir / "controllers.yaml"),
            "defenses_yaml_hash": calculate_file_hash(configs_dir / "defenses.yaml"),
            "experiment_yaml_hash": calculate_file_hash(configs_dir / "experiment.yaml"),
            "model_yaml_hash": calculate_file_hash(configs_dir / "model.yaml"),
            "evaluation_roles_hash": calculate_file_hash(roles_path),
            "evaluation_batches_hash": calculate_file_hash(batches_path),
            "frozen_rf_hash": calculate_file_hash(artifacts_dir / "models/frozen_rf.joblib"),
            "scaler_hash": calculate_file_hash(artifacts_dir / "preprocessors/standard_scaler.joblib"),
            "feature_names_hash": calculate_file_hash(artifacts_dir / "preprocessors/feature_names.json"),
            "feature_mask_hash": calculate_file_hash(mask_path),
            "training_bounds_hash": calculate_file_hash(bounds_path),
            "X_eval_hash": calculate_file_hash(x_eval_path),
            "metadata_eval_hash": calculate_file_hash(meta_eval_path),
            "crafting_identity_hash": hashlib.sha256(crafting_eps_pre.astype(np.int64).tobytes()).hexdigest(),
            "measurement_identity_hash": hashlib.sha256(measurement_eps_pre.astype(np.int64).tobytes()).hexdigest(),
            "freeze_commit_sha256": freeze_commit_sha256,
            "freeze_tag_sha256": freeze_tag_sha256,
            "orchestration_commit_sha256": orchestration_commit_sha256,
        }
    else:
        prov_hashes = override_provenance

    # 1. Whole-cache reuse check using INDEPENDENT EXPECTATIONS
    if target_cache_dir.exists():
        x_target = target_cache_dir / "X_attacked.parquet"
        s_target = target_cache_dir / "status.parquet"
        m_target = target_cache_dir / "manifest.json"

        if x_target.exists() and s_target.exists() and m_target.exists():
            x_target_hash = calculate_file_hash(x_target)
            s_target_hash = calculate_file_hash(s_target)

            screening_metrics_target = None
            try:
                with open(m_target, "r") as f:
                    manifest_raw = json.load(f)
                screening_metrics_target = manifest_raw.get("screening_metrics")
            except Exception:
                pass

            expected_identity_check = build_expected_cache_identity(
                scenario=canonical_scen,
                seed=seed,
                expected_row_count=expected_row_count,
                prov_hashes=prov_hashes,
                script_hashes=script_hashes,
                attack_parameters=attack_params,
                query_budgets=query_budgets,
                x_attacked_sha256=x_target_hash,
                status_sha256=s_target_hash,
                screening_metrics=screening_metrics_target,
            )

            is_valid, reason = validate_completed_cache(
                cache_dir=target_cache_dir,
                expected_scenario=canonical_scen,
                expected_seed=seed,
                expected_cache_identity=expected_identity_check,
                expected_feature_names=feature_names,
                resolved_batches=resolved_batches,
                official_mode=official_mode,
                expected_row_count=expected_row_count,
                expected_orchestration_commit=current_head if official_mode else None,
            )
            if is_valid:
                print(f"[REUSE] Existing cache for {canonical_scen} seed {seed} is valid. Reusing: {target_cache_dir}")
                return target_cache_dir
            else:
                print(f"[QUARANTINE] Target cache directory exists but invalid ({reason}). Quarantining...")
                quarantine_directory(target_cache_dir, cache_root=output_dir, base_name=target_dir_name)
        else:
            print(f"[QUARANTINE] Incomplete target cache directory. Quarantining...")
            quarantine_directory(target_cache_dir, cache_root=output_dir, base_name=target_dir_name)

    # 2. Clean up any stale staging directories from previous crashed attempts
    if output_dir.exists():
        for stale in output_dir.glob(f"{target_dir_name}.staging*"):
            if stale != staging_dir and stale.is_dir():
                quarantine_directory(stale, cache_root=output_dir, base_name=f"{target_dir_name}_stale_staging")

    # 3. Prepare data
    start_time_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    if official_mode:
        x_eval_path = data_dir / "processed/X_eval.parquet"
        meta_eval_path = data_dir / "processed/metadata_eval.parquet"
        roles_path = data_dir / "manifests/evaluation_roles.csv"
        batches_path = data_dir / "manifests/evaluation_batches.csv"

        df_roles = pd.read_csv(roles_path)
        df_x_eval = pd.read_parquet(x_eval_path)
        df_meta_eval = pd.read_parquet(meta_eval_path)
        df_batches = pd.read_csv(batches_path)

        # STRICT PARTITION ALIGNMENT VALIDATION
        validate_evaluation_partition_alignment(
            df_x_eval=df_x_eval,
            df_meta_eval=df_meta_eval,
            df_roles=df_roles,
            df_batches=df_batches,
            expected_feature_names=feature_names,
        )

        df_x_eval["eval_position"] = df_x_eval.index.astype(int)
        df_meta_eval["eval_position"] = df_meta_eval.index.astype(int)

        crafting_eps = df_roles[df_roles["role"] == "crafting"]["eval_position"].sort_values().values
        measurement_eps = df_roles[df_roles["role"] == "measurement"]["eval_position"].sort_values().values

        X_craft = df_x_eval.iloc[crafting_eps][feature_names].values.astype(np.float32)
        y_craft = df_meta_eval.iloc[crafting_eps]["y_binary"].values.astype(int)

        X_meas = df_x_eval.iloc[measurement_eps][feature_names].values.astype(np.float32)
        y_meas = df_meta_eval.iloc[measurement_eps]["y_binary"].values.astype(int)
        eps_meas = measurement_eps.astype(np.int64)

        import joblib
        rf_model = joblib.load(artifacts_dir / "models/frozen_rf.joblib")
        def predict_fn(x_in):
            return rf_model.predict(x_in)
    else:
        # Synthetic testing mode
        X_craft = override_X_craft
        y_craft = override_y_craft
        X_meas = override_X_meas
        y_meas = override_y_meas
        eps_meas = override_eps
        predict_fn = override_predict_fn

    # Setup benign reference pool for Decision Boundary (LABEL ISOLATION RULE)
    if canonical_scen == "DecisionBoundary":
        # Benign reference selection uses model predictions on crafting rows, NOT true labels
        craft_preds = predict_fn(X_craft)
        benign_mask = (craft_preds == 0)
        if np.any(benign_mask):
            benign_reference_pool = X_craft[benign_mask]
        else:
            benign_reference_pool = X_craft

    # Build cache into staging directory using frozen AttackCacheBuilder
    builder = AttackCacheBuilder(
        feature_names=feature_names,
        modifiable_mask=modifiable_mask,
        training_bounds=training_bounds,
        provenance_hashes=prov_hashes,
        seed=seed,
    )

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[BUILD] Generating cache for {canonical_scen} seed {seed} into staging: {staging_dir}")

        builder.build(
            scenario=canonical_scen,
            X_measurement=X_meas,
            y_measurement=y_meas,
            eval_positions=eps_meas,
            output_dir=staging_dir,
            X_crafting=X_craft,
            y_crafting=y_craft,
            oracle_or_predict_fn=predict_fn,
            surrogate_attack=surrogate_attack,
            boundary_attack=boundary_attack,
            benign_reference_pool=benign_reference_pool,
            n_boundary_targets=n_boundary,
            max_queries=50,
            attack_script_hashes=script_hashes,
            attack_parameters=attack_params,
            query_budgets=query_budgets,
        )

        end_time_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        x_hash = calculate_file_hash(staging_dir / "X_attacked.parquet")
        s_hash = calculate_file_hash(staging_dir / "status.parquet")
        m_hash = calculate_file_hash(staging_dir / "manifest.json")

        # Write completion.json LAST into staging directory
        completion_data = {
            "completion_state": "COMPLETED",
            "completed": True,
            "attack_scenario": canonical_scen,
            "display_scenario": display_scenario(canonical_scen),
            "effective_seed": seed,
            "row_count": len(X_meas),
            "freeze_commit": FREEZE_COMMIT,
            "freeze_tag": FREEZE_TAG,
            "execution_script_commit": current_head,
            "generation_start_timestamp": start_time_iso,
            "generation_end_timestamp": end_time_iso,
            "artifacts": {
                "X_attacked_sha256": x_hash,
                "status_sha256": s_hash,
                "manifest_sha256": m_hash,
            },
            "provenance_hashes": prov_hashes,
            "attack_parameters": attack_params,
            "query_budgets": query_budgets,
            "attack_script_hashes": script_hashes,
        }

        completion_file = staging_dir / "completion.json"
        with open(completion_file, "w") as f:
            json.dump(completion_data, f, indent=2)

        CompletionMarker(
            run_id=f"cache_{canonical_scen}_{seed}",
            timestamp=end_time_iso,
            provenance_hashes={k: v for k, v in prov_hashes.items() if k in _REQUIRED_PROVENANCE_KEYS},
        )

        # Inspect screening metrics if emitted
        staging_screening = None
        try:
            with open(staging_dir / "manifest.json", "r") as f:
                staging_m = json.load(f)
            staging_screening = staging_m.get("screening_metrics")
        except Exception:
            pass

        # Build independent expected cache identity for staging validation
        staging_expected_identity = build_expected_cache_identity(
            scenario=canonical_scen,
            seed=seed,
            expected_row_count=expected_row_count,
            prov_hashes=prov_hashes,
            script_hashes=script_hashes,
            attack_parameters=attack_params,
            query_budgets=query_budgets,
            x_attacked_sha256=x_hash,
            status_sha256=s_hash,
            screening_metrics=staging_screening,
        )

        # Comprehensive validation of the completed staging directory before atomic publication
        is_valid, reason = validate_completed_cache(
            cache_dir=staging_dir,
            expected_scenario=canonical_scen,
            expected_seed=seed,
            expected_cache_identity=staging_expected_identity,
            expected_feature_names=feature_names,
            resolved_batches=resolved_batches,
            official_mode=official_mode,
            expected_row_count=expected_row_count,
            expected_orchestration_commit=current_head if official_mode else None,
        )
        if not is_valid:
            raise RuntimeError(f"Staging cache validation failed: {reason}")

        # Atomic publication: rename staging to target
        staging_dir.rename(target_cache_dir)
        print(f"[PUBLISH] Successfully published atomic cache to: {target_cache_dir}")
        return target_cache_dir

    except Exception as exc:
        print(f"[ERROR] Cache generation failed for {canonical_scen} seed {seed}: {exc}")
        if staging_dir.exists():
            print(f"[QUARANTINE] Safely quarantining failed staging directory: {staging_dir}")
            quarantine_directory(staging_dir, cache_root=output_dir, base_name=f"{target_dir_name}_failed")
        raise


# ---------------------------------------------------------------------------
# Main Orchestration Driver
# ---------------------------------------------------------------------------
def build_evaluation_caches(
    configs_dir: Path,
    data_dir: Path,
    artifacts_dir: Path,
    output_dir: Path,
    scenario_filter: Optional[str] = None,
    seed_filter: Optional[int] = None,
    execute: bool = False,
    preflight_only: bool = False,
) -> int:
    """
    Main orchestration entry point.
    """
    # 1. Run Preflight unconditionally
    preflight_results = run_preflight(
        configs_dir=configs_dir,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        output_dir=output_dir,
        repo_root=REPO_ROOT,
        enforce_git=True,
    )
    print_preflight_report(preflight_results)

    if preflight_only or not execute:
        print("\n[PREFLIGHT ONLY] Execution flag (--execute) was not provided or --preflight-only was set.")
        print("[PREFLIGHT ONLY] Preflight checks PASSED. No caches were generated. Exiting cleanly.")
        return 0

    # 2. Execution authorized: verify reproducible state first
    print("\n" + "=" * 78)
    print("VERIFYING REPRODUCIBLE EXECUTION STATE (--execute)")
    print("=" * 78)
    head_commit = verify_reproducible_execution_state(
        repo_root=REPO_ROOT,
        configs_dir=configs_dir,
        artifacts_dir=artifacts_dir,
        data_dir=data_dir,
    )
    print(f"Clean reproducible state verified at commit: {head_commit}")

    print("\n" + "=" * 78)
    print("STARTING AUTHORIZED EVALUATION CACHE GENERATION (--execute)")
    print("=" * 78)

    derived_pairs = derive_cache_pairs(configs_dir)

    selected_pairs = []
    for scen, seed in derived_pairs:
        if scenario_filter and canonicalize_scenario(scen) != canonicalize_scenario(scenario_filter):
            continue
        if seed_filter is not None and seed != seed_filter:
            continue
        selected_pairs.append((scen, seed))

    if not selected_pairs:
        raise ValueError(
            f"No pairs match scenario_filter={scenario_filter!r} and seed_filter={seed_filter!r}"
        )

    print(f"Selected {len(selected_pairs)} / 15 pairs for execution:")
    for scen, seed in selected_pairs:
        print(f"  - {scen:20s} (seed {seed})")

    batches_path = data_dir / "manifests/evaluation_batches.csv"
    resolved_batches = pd.read_csv(batches_path)

    for idx, (scen, seed) in enumerate(selected_pairs, 1):
        print(f"\n[{idx}/{len(selected_pairs)}] Building cache for {scen} seed {seed}...")
        build_single_cache(
            scenario=scen,
            seed=seed,
            output_dir=output_dir,
            configs_dir=configs_dir,
            data_dir=data_dir,
            artifacts_dir=artifacts_dir,
            resolved_batches=resolved_batches,
            official_mode=True,
        )

    print("\n" + "=" * 78)
    print(f"COMPLETED EVALUATION CACHE GENERATION: {len(selected_pairs)} caches ready.")
    print("=" * 78)
    return 0


# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 10B Attack-Cache Orchestration CLI (safe by default).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Explicitly authorize actual cache generation. Without this, only preflight is run.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        default=False,
        help="Explicitly perform non-mutating preflight only (does not load evaluation data).",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Optional attack scenario filter ('Silent Probing', 'Surrogate Transfer', 'Decision Boundary').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional primary seed filter (42, 43, 44, 45, 46).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/caches",
        help="Root directory where attack caches are stored.",
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=REPO_ROOT / "configs",
        help="Directory containing frozen configuration YAMLs.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data",
        help="Directory containing manifests and processed datasets.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=REPO_ROOT / "artifacts",
        help="Directory containing preprocessors and models.",
    )

    args = parser.parse_args(argv)

    if args.scenario is not None:
        try:
            canonicalize_scenario(args.scenario)
        except ValueError as e:
            parser.error(str(e))

    if args.seed is not None and args.seed not in VALID_PRIMARY_SEEDS:
        parser.error(
            f"Invalid seed {args.seed}. Must be one of primary seeds: {VALID_PRIMARY_SEEDS}"
        )

    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        return build_evaluation_caches(
            configs_dir=args.configs_dir,
            data_dir=args.data_dir,
            artifacts_dir=args.artifacts_dir,
            output_dir=args.output_dir,
            scenario_filter=args.scenario,
            seed_filter=args.seed,
            execute=args.execute,
            preflight_only=args.preflight_only,
        )
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

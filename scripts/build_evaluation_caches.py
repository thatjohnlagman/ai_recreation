#!/usr/bin/env python3
"""
scripts/build_evaluation_caches.py

Phase 10B Attack-Cache Orchestration Entry Point.

This script acts as the thin orchestration layer over AttackCacheBuilder for Phase 10
evaluation attack caches. It enforces:
  1. Safe-by-default execution: running without --execute performs non-mutating preflight only.
  2. Strict separation: preflight never accesses evaluation features (X_eval.parquet) or labels.
  3. Matrix derivation: dynamically derives the 15 canonical (scenario, seed) pairs from
     frozen configuration files and validates sensitivity alias reuse.
  4. Complete provenance tracking: records freeze commit/tag, script commit, model hash,
     roles/batches hashes, config hashes, preprocessor hashes, and artifact checksums.
  5. Atomic publication & quarantine: writes to staging directories, validates via
     ConcreteAttackCacheProvider, writes completion.json last, atomically publishes,
     and quarantines any incomplete or corrupted directories.
  6. Whole-cache reuse: valid completed caches are verified and reused without recomputation.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
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

CANONICAL_SCENARIOS = {
    "Silent Probing": "SilentProbing",
    "SilentProbing": "SilentProbing",
    "Surrogate Transfer": "SurrogateTransfer",
    "SurrogateTransfer": "SurrogateTransfer",
    "Decision Boundary": "DecisionBoundary",
    "DecisionBoundary": "DecisionBoundary",
}

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
    # Try normalized strip
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
# Matrix Derivation
# ---------------------------------------------------------------------------
def derive_cache_pairs(configs_dir: Path) -> List[Tuple[str, int]]:
    """
    Derives—not manually hardcodes—the distinct (attack_scenario, seed) pairs
    needed by the frozen 252 unique runs in the evaluation matrix.

    Validates:
      - exactly three attack scenarios
      - five primary seeds (42..46)
      - exactly 15 distinct canonical cache identities
      - sensitivity aliases reuse the corresponding primary cache
      - cache identity is independent of defense and controller configuration
    """
    matrix = generate_evaluation_matrix(configs_dir)
    unique_execs = get_unique_executions(matrix)

    # Validate overall matrix counts
    if len(matrix) != 279:
        raise ValueError(f"Evaluation matrix has {len(matrix)} rows, expected 279")
    if len(unique_execs) != 252:
        raise ValueError(f"Unique executions count is {len(unique_execs)}, expected 252")

    # Verify alias reuse
    aliases = matrix[matrix["is_alias"]]
    if len(aliases) != 27:
        raise ValueError(f"Expected 27 sensitivity aliases, got {len(aliases)}")
    for _, r in aliases.iterrows():
        target_id = r["alias_for_run_id"]
        target_rows = matrix[matrix["run_id"] == target_id]
        if len(target_rows) != 1:
            raise ValueError(f"Alias {r['run_id']} targets missing run {target_id}")
        target = target_rows.iloc[0]
        if r["attack_scenario"] != target["attack_scenario"] or r["seed"] != target["seed"]:
            raise ValueError(
                f"Alias {r['run_id']} scenario/seed mismatch with target {target_id}"
            )

    # Derive unique (scenario, seed) pairs across all unique runs
    unique_pairs_set: Set[Tuple[str, int]] = set()
    for _, r in unique_execs.iterrows():
        scen = r["attack_scenario"]
        seed = int(r["seed"])
        unique_pairs_set.add((scen, seed))

    derived_pairs = sorted(list(unique_pairs_set), key=lambda x: (x[0], x[1]))

    # Validations on derived pairs
    scenarios_found = {p[0] for p in derived_pairs}
    expected_scenarios = {"Silent Probing", "Surrogate Transfer", "Decision Boundary"}
    if scenarios_found != expected_scenarios:
        raise ValueError(
            f"Derived scenarios {scenarios_found} do not match expected {expected_scenarios}"
        )

    seeds_found = {p[1] for p in derived_pairs}
    if seeds_found != set(VALID_PRIMARY_SEEDS):
        raise ValueError(
            f"Derived seeds {seeds_found} do not match expected {VALID_PRIMARY_SEEDS}"
        )

    if len(derived_pairs) != 15:
        raise ValueError(f"Expected exactly 15 canonical cache pairs, derived {len(derived_pairs)}")

    # Verify cache identity independence from defenses and controllers
    grouped = matrix.groupby(["attack_scenario", "seed"])
    for (scen, seed), group in grouped:
        defenses_in_group = set(group["defense_name"])
        controllers_in_group = set(group["controller_config_id"])
        # All 3 defenses and multiple controllers must map to this exact pair
        if len(defenses_in_group) != 3:
            raise ValueError(
                f"Pair ({scen}, {seed}) does not span all 3 defenses: {defenses_in_group}"
            )
        if len(controllers_in_group) < 2:
            raise ValueError(
                f"Pair ({scen}, {seed}) missing expected controllers: {controllers_in_group}"
            )

    return derived_pairs


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
      - Safe by default.
    """
    preflight_results: Dict[str, Any] = {
        "status": "PASS",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git": {},
        "protected_hashes": {},
        "configurations": {},
        "manifests": {},
        "derived_pairs": [],
        "cache_inventory": {},
    }

    # 1. Git verification
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

            # Ancestry check
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

            # Check for changes in configs/ relative to freeze tag
            diff_configs = subprocess.check_output(
                ["git", "diff", "--name-only", FREEZE_TAG, "--", "configs/"],
                cwd=repo_root,
                stderr=subprocess.PIPE,
                text=True,
            ).strip()
            if diff_configs:
                raise ValueError(
                    f"Configs directory differs from freeze tag: {diff_configs.splitlines()}"
                )

            preflight_results["git"] = {
                "head_commit": current_head,
                "freeze_commit": tag_commit,
                "freeze_tag": FREEZE_TAG,
                "is_ancestor": is_ancestor,
                "configs_unchanged_vs_freeze": True,
            }
        except subprocess.CalledProcessError as e:
            if enforce_git:
                raise RuntimeError(f"Git verification failed: {e.stderr}") from e
    elif enforce_git:
        raise RuntimeError(f"Git directory not found at {git_dir}")

    # 2. Protected Hashes Verification
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

    # 3. Frozen configuration file checks
    exp_yaml_path = configs_dir / "experiment.yaml"
    attacks_yaml_path = configs_dir / "attacks.yaml"
    for c_path in (exp_yaml_path, attacks_yaml_path, configs_dir / "defenses.yaml", configs_dir / "controllers.yaml"):
        if not c_path.exists():
            raise FileNotFoundError(f"Required configuration file missing: {c_path}")

    with open(exp_yaml_path, "r") as f:
        exp_cfg = yaml.safe_load(f)
    date_frozen = exp_cfg.get("experiment", {}).get("date_frozen")
    if date_frozen != FROZEN_DATE:
        raise ValueError(
            f"experiment.date_frozen is {date_frozen!r}, expected {FROZEN_DATE!r}"
        )

    # 4. Preprocessors verification
    mask_path = artifacts_dir / "preprocessors/feature_mask.json"
    bounds_path = artifacts_dir / "preprocessors/training_bounds.parquet"
    scaler_path = artifacts_dir / "preprocessors/standard_scaler.joblib"
    fnames_path = artifacts_dir / "preprocessors/feature_names.json"

    for p in (mask_path, bounds_path, scaler_path, fnames_path):
        if not p.exists():
            raise FileNotFoundError(f"Required preprocessor artifact missing: {p}")

    preflight_results["configurations"] = {
        "date_frozen": date_frozen,
        "experiment_yaml_hash": calculate_file_hash(exp_yaml_path),
        "attacks_yaml_hash": calculate_file_hash(attacks_yaml_path),
        "feature_mask_hash": calculate_file_hash(mask_path),
        "training_bounds_hash": calculate_file_hash(bounds_path),
        "scaler_hash": calculate_file_hash(scaler_path),
    }

    # 5. Manifest structure check (CSV only, NO parquet evaluation features loaded)
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

    # 6. Derive 15 canonical pairs
    derived_pairs = derive_cache_pairs(configs_dir)
    preflight_results["derived_pairs"] = [
        {"scenario": p[0], "seed": p[1], "canonical_slug": f"{canonicalize_scenario(p[0])}_{p[1]}"}
        for p in derived_pairs
    ]

    # 7. Inspect existing cache inventory
    cache_status_map: Dict[str, str] = {}
    valid_caches_count = 0
    if output_dir.exists():
        for p in derived_pairs:
            slug = f"{canonicalize_scenario(p[0])}_{p[1]}"
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
            slug = f"{canonicalize_scenario(p[0])}_{p[1]}"
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
        print(f"  Configs Status:  {'MATCHES FREEZE TAG' if git.get('configs_unchanged_vs_freeze') else 'MODIFIED'}")

    print("\n--- Protected Hashes (SHA-256) ---")
    for name, info in results["protected_hashes"].items():
        print(f"  [{name:7s}] {info['actual']} (VERIFIED)")

    mf = results["manifests"]
    print("\n--- Evaluation Partition Manifests ---")
    print(f"  Roles:           {mf['roles_count']:,} total ({mf['crafting_count']:,} crafting, {mf['measurement_count']:,} measurement)")
    print(f"  Batches:         {mf['batch_count']} batches of {mf['records_per_batch']} rows (72,000 total)")

    pairs = results["derived_pairs"]
    print(f"\n--- Derived Canonical Cache Pairs ({len(pairs)} Distinct Pairs) ---")
    print("  #   Scenario                 Seed   Canonical Identifier      Cache Status")
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
    print(f"  [x] Official execution requires explicit --execute flag")
    print(f"  [x] Total valid caches present: {results['cache_inventory']['completed_count']} / 15")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Cache Validation & Resumption
# ---------------------------------------------------------------------------
def validate_completed_cache(
    cache_dir: Path,
    expected_feature_names: List[str],
    resolved_batches: pd.DataFrame,
    official_mode: bool = True,
    expected_row_count: int = 72000,
) -> Tuple[bool, str]:
    """
    Validates whether an existing cache directory is 100% complete and sound.

    Checks:
      - completion.json exists, parses, and has status == 'COMPLETED'
      - manifest.json exists and validates via validate_cache_manifest
      - X_attacked.parquet and status.parquet exist and match manifest checksums
      - ConcreteAttackCacheProvider accepts the cache artifacts without error
      - completion.json artifact hashes match actual file hashes
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

    try:
        with open(completion_path, "r") as f:
            comp_data = json.load(f)
        if comp_data.get("completion_state") != "COMPLETED":
            return False, f"Incomplete state in completion.json: {comp_data.get('completion_state')}"
        if not comp_data.get("completed", False):
            return False, "completed flag is not True in completion.json"
        _validate_iso_timestamp(comp_data.get("generation_end_timestamp"), "generation_end_timestamp")

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        # Validate through validate_cache_manifest
        validate_cache_manifest(
            manifest_path=manifest_path,
            X_attacked_path=x_path,
            expected_hashes=manifest,
            status_path=s_path,
            expected_row_count=expected_row_count,
        )

        # Validate through strict provider
        ConcreteAttackCacheProvider(
            cache_dir=cache_dir,
            resolved_batches=resolved_batches,
            expected_cache_identity=manifest,
            expected_feature_names=expected_feature_names,
            official_mode=official_mode,
            expected_row_count=expected_row_count,
        )

        # Cross-validate completion.json checksums
        x_hash = calculate_file_hash(x_path)
        s_hash = calculate_file_hash(s_path)
        m_hash = calculate_file_hash(manifest_path)

        art_hashes = comp_data.get("artifacts", {})
        if art_hashes.get("X_attacked_sha256") != x_hash:
            return False, "completion.json X_attacked_sha256 mismatch"
        if art_hashes.get("status_sha256") != s_hash:
            return False, "completion.json status_sha256 mismatch"
        if art_hashes.get("manifest_sha256") != m_hash:
            return False, "completion.json manifest_sha256 mismatch"

        return True, "Valid complete cache"
    except Exception as e:
        return False, f"Validation error: {e}"


def quarantine_directory(directory: Path, base_name: Optional[str] = None) -> Path:
    """Quarantines a failed or incomplete directory by renaming it with a unique timestamp."""
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M%S%f")
    prefix = base_name if base_name else directory.name
    quarantine_name = f"{prefix}_quarantined_{ts}"
    quarantine_dir = directory.with_name(quarantine_name)
    if directory.exists():
        directory.rename(quarantine_dir)
    return quarantine_dir


# ---------------------------------------------------------------------------
# Official Cache Execution
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
) -> Path:
    """
    Builds a single canonical cache for the given (scenario, seed) pair.

    Orchestration guarantees:
      - Validates existing cache: if already valid and complete, reuses without recomputation.
      - If existing cache is invalid or incomplete, quarantines and rebuilds from scratch.
      - Writes to staging directory ({cache_dir}.staging).
      - Writes completion.json LAST into staging directory before atomic publication.
      - Re-validates staging directory before atomically renaming to target.
      - On any failure: cleans up / quarantines staging and leaves no half-written cache.
    """
    canonical_scen = canonicalize_scenario(scenario)
    target_dir_name = f"{canonical_scen}_{seed}"
    target_cache_dir = output_dir / target_dir_name
    staging_dir = output_dir / f"{target_dir_name}.staging"

    expected_row_count = 72000 if official_mode else (override_row_count if override_row_count else len(override_X_meas))

    # Pre-load feature names and bounds
    mask_path = artifacts_dir / "preprocessors/feature_mask.json"
    bounds_path = artifacts_dir / "preprocessors/training_bounds.parquet"
    with open(mask_path, "r") as f:
        mask_data = json.load(f)
    feature_names = mask_data["feature_columns"]
    modifiable_mask = mask_data["feature_mask"]
    training_bounds = pd.read_parquet(bounds_path)

    # 1. Whole-cache reuse check
    if target_cache_dir.exists():
        is_valid, reason = validate_completed_cache(
            cache_dir=target_cache_dir,
            expected_feature_names=feature_names,
            resolved_batches=resolved_batches,
            official_mode=official_mode,
            expected_row_count=expected_row_count,
        )
        if is_valid:
            print(f"[REUSE] Existing cache for {canonical_scen} seed {seed} is valid. Reusing: {target_cache_dir}")
            return target_cache_dir
        else:
            print(f"[QUARANTINE] Target cache directory exists but invalid ({reason}). Quarantining...")
            quarantine_directory(target_cache_dir, base_name=target_dir_name)

    # 2. Clean up any stale staging or tmp directory from previous crashes
    if staging_dir.exists():
        quarantine_directory(staging_dir, base_name=f"{target_dir_name}_stale_staging")
    stale_tmp = output_dir / f"{target_dir_name}.staging.tmp"
    if stale_tmp.exists():
        quarantine_directory(stale_tmp, base_name=f"{target_dir_name}_stale_staging_tmp")

    # 3. Prepare data and provenance
    start_time_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    if official_mode:
        # Load official evaluation data through exact eval_position joins
        x_eval_path = data_dir / "processed/X_eval.parquet"
        meta_eval_path = data_dir / "processed/metadata_eval.parquet"
        roles_path = data_dir / "manifests/evaluation_roles.csv"

        df_roles = pd.read_csv(roles_path)
        df_x_eval = pd.read_parquet(x_eval_path)
        df_meta_eval = pd.read_parquet(meta_eval_path)

        # Merge on eval_position to partition strictly
        df_x_eval["eval_position"] = df_x_eval.index.astype(int)
        df_meta_eval["eval_position"] = df_meta_eval.index.astype(int)

        crafting_eps = df_roles[df_roles["role"] == "crafting"]["eval_position"].sort_values().values
        measurement_eps = df_roles[df_roles["role"] == "measurement"]["eval_position"].sort_values().values

        X_craft = df_x_eval.iloc[crafting_eps][feature_names].values.astype(np.float32)
        y_craft = df_meta_eval.iloc[crafting_eps]["y_binary"].values.astype(int)

        X_meas = df_x_eval.iloc[measurement_eps][feature_names].values.astype(np.float32)
        y_meas = df_meta_eval.iloc[measurement_eps]["y_binary"].values.astype(int)
        eps_meas = measurement_eps.astype(np.int64)

        # Load RF model
        import joblib
        rf_model = joblib.load(artifacts_dir / "models/frozen_rf.joblib")
        def predict_fn(x_in):
            return rf_model.predict(x_in)

        # Compute provenance hashes
        prov_hashes = {
            "attacks_yaml_hash": calculate_file_hash(configs_dir / "attacks.yaml"),
            "controllers_yaml_hash": calculate_file_hash(configs_dir / "controllers.yaml"),
            "defenses_yaml_hash": calculate_file_hash(configs_dir / "defenses.yaml"),
            "experiment_yaml_hash": calculate_file_hash(configs_dir / "experiment.yaml"),
            "evaluation_roles_hash": calculate_file_hash(roles_path),
            "evaluation_batches_hash": calculate_file_hash(data_dir / "manifests/evaluation_batches.csv"),
            "frozen_rf_hash": calculate_file_hash(artifacts_dir / "models/frozen_rf.joblib"),
            "scaler_hash": calculate_file_hash(artifacts_dir / "preprocessors/standard_scaler.joblib"),
            "feature_names_hash": calculate_file_hash(artifacts_dir / "preprocessors/feature_names.json"),
            "feature_mask_hash": calculate_file_hash(mask_path),
            "training_bounds_hash": calculate_file_hash(bounds_path),
            "X_eval_hash": calculate_file_hash(x_eval_path),
            "metadata_eval_hash": calculate_file_hash(meta_eval_path),
            "crafting_identity_hash": hashlib.sha256(crafting_eps.astype(np.int64).tobytes()).hexdigest(),
            "measurement_identity_hash": hashlib.sha256(measurement_eps.astype(np.int64).tobytes()).hexdigest(),
        }
    else:
        # Synthetic testing mode
        X_craft = override_X_craft
        y_craft = override_y_craft
        X_meas = override_X_meas
        y_meas = override_y_meas
        eps_meas = override_eps
        predict_fn = override_predict_fn
        prov_hashes = override_provenance

    # Compute attack script hashes
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

    # Attack parameters and query budgets per scenario
    surrogate_attack = None
    boundary_attack = None
    benign_reference_pool = None
    n_boundary = 200 if official_mode else (override_n_boundary_targets if override_n_boundary_targets is not None else min(200, len(X_meas)))

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
            "crafting_queries_budget": len(X_craft),
            "target_evaluation_queries": len(X_meas),
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
        # Benign reference pool drawn strictly from crafting pool
        benign_mask = (y_craft == 0)
        benign_reference_pool = X_craft[benign_mask] if np.any(benign_mask) else X_craft
    else:
        raise ValueError(f"Unsupported canonical scenario: {canonical_scen}")

    # Build cache into staging directory
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

        # Compute file checksums of generated artifacts
        x_hash = calculate_file_hash(staging_dir / "X_attacked.parquet")
        s_hash = calculate_file_hash(staging_dir / "status.parquet")
        m_hash = calculate_file_hash(staging_dir / "manifest.json")

        # Get git details if available
        current_head = "UNKNOWN"
        try:
            current_head = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.PIPE, text=True
            ).strip()
        except Exception:
            pass

        # Write completion.json LAST
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

        # Validate completion marker schema using CompletionMarker
        CompletionMarker(
            run_id=f"cache_{canonical_scen}_{seed}",
            timestamp=end_time_iso,
            provenance_hashes={k: v for k, v in prov_hashes.items() if k in _REQUIRED_PROVENANCE_KEYS},
        )

        # Comprehensive validation of the completed staging directory
        is_valid, reason = validate_completed_cache(
            cache_dir=staging_dir,
            expected_feature_names=feature_names,
            resolved_batches=resolved_batches,
            official_mode=official_mode,
            expected_row_count=expected_row_count,
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
            print(f"[QUARANTINE] Quarantining failed staging directory: {staging_dir}")
            quarantine_directory(staging_dir, base_name=f"{target_dir_name}_failed")
        tmp_dir = output_dir / f"{target_dir_name}.staging.tmp"
        if tmp_dir.exists():
            print(f"[QUARANTINE] Quarantining failed temporary directory: {tmp_dir}")
            quarantine_directory(tmp_dir, base_name=f"{target_dir_name}_failed")
        raise


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

    # 2. Execution authorized
    print("\n" + "=" * 78)
    print("STARTING AUTHORIZED EVALUATION CACHE GENERATION (--execute)")
    print("=" * 78)

    derived_pairs = derive_cache_pairs(configs_dir)

    # Filter pairs if requested
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

    # Load resolved batches once for provider validation
    batches_path = data_dir / "manifests/evaluation_batches.csv"
    resolved_batches = pd.read_csv(batches_path)

    # Execute builds sequentially
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

    # Validate arguments early
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

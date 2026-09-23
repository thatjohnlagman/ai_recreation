#!/usr/bin/env python3
"""
scripts/run_evaluation.py

Phase 10D v2 Official Evaluation Orchestration Entry Point.

This script acts as the official execution and validation orchestrator for Phase 10
evaluation matrix runs. It strictly enforces:
  1. Safe-by-default execution: running without --execute performs non-mutating preflight only.
  2. Preflight separation: preflight does not open evaluation Parquets (X_eval.parquet,
     metadata_eval.parquet) or perform model inference; preflight reads and validates
     frozen role/batch manifests, including their label-alignment fields; official cache
     files are read only as raw bytes for non-mutating cryptographic verification.
  3. Dynamic matrix derivation: derives exactly 279 matrix references (90 primary, 189 sensitivity,
     27 exact C1 aliases, 252 unique executions, 36,288 batch evaluations) from frozen configs.
  4. Real-run complete provenance: builds all 11 required provenance hashes from canonical on-disk
     artifacts and independently verifies them against official cache manifests.
  5. Dry CompletionMarker validation: validates completion marker construction before running any batch.
  6. Independent cache pinning: validates caches against artifacts/reports/cache_inventory_v2.json
     before trusting or parsing cache manifests.
  7. Enforced Git cleanliness: rejects any staged/unstaged tracked modification or untracked code/doc files.
  8. Unsuppressed disk-space gate: aborts execution if free space is below required threshold.
  9. Defenses.yaml derived bounds: Base intensities and bounds derived directly from frozen defenses.yaml.
 10. Strengthened manifest cross-validation: validates roles vs batches alignment, 72,000 measurement positions,
     zero crafting overlap, binary y_binary agreement, composite identities, and deterministic sorting.
 11. Strict run reuse and quarantine: validates that existing runs match the target matrix row, provenance,
     cache identity, and batch-record run IDs before reuse; quarantines corrupted runs.
 12. Pointer-only alias publication: alias directories are pointer-only artifacts with alias_pointer.json
     and completion.json, never masquerading as independent runs.
 13. Dependency-safe execution planning: automatically resolves primary target runs for aliases before execution.
 14. Exact pairing and common randomness: Base and RA runs paired on (seed, attack_scenario, defense, batch_id).
 15. Timing and label isolation: intensity obtained strictly before batch-t labels; feedback submitted only
     for batch t+1; Base receives no feedback.
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
    "ExperimentConfig": "a1c5a389b1bc3c66311c7a96a7ccc3ee54af86506e64b38be7f26ad3c3d84c70",
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
# Complete Production Provenance Construction
# ---------------------------------------------------------------------------
def build_production_provenance(
    repo_root: Path = REPO_ROOT,
    configs_dir: Optional[Path] = None,
    manifests_dir: Optional[Path] = None,
    models_dir: Optional[Path] = None,
    preprocessors_dir: Optional[Path] = None,
) -> Dict[str, str]:
    """
    Builds the complete, authoritative 11-field provenance dictionary from canonical
    on-disk artifacts. Validates that every hash is a lowercase 64-char hex string
    and matches the required provenance schema.
    """
    cfg_dir = configs_dir or (repo_root / "configs")
    man_dir = manifests_dir or (repo_root / "data/manifests")
    mod_dir = models_dir or (repo_root / "artifacts/models")
    pre_dir = preprocessors_dir or (repo_root / "artifacts/preprocessors")

    prov_file_map = {
        "frozen_rf_hash": mod_dir / "frozen_rf.joblib",
        "scaler_hash": pre_dir / "standard_scaler.joblib",
        "feature_names_hash": pre_dir / "feature_names.json",
        "feature_mask_hash": pre_dir / "feature_mask.json",
        "training_bounds_hash": pre_dir / "training_bounds.parquet",
        "evaluation_roles_hash": man_dir / "evaluation_roles.csv",
        "evaluation_batches_hash": man_dir / "evaluation_batches.csv",
        "attacks_yaml_hash": cfg_dir / "attacks.yaml",
        "controllers_yaml_hash": cfg_dir / "controllers.yaml",
        "defenses_yaml_hash": cfg_dir / "defenses.yaml",
        "experiment_yaml_hash": cfg_dir / "experiment.yaml",
    }

    provenance: Dict[str, str] = {}
    for key, p in prov_file_map.items():
        if not p.exists():
            raise FileNotFoundError(f"Canonical provenance artifact missing on disk: {p}")
        h = calculate_file_hash(p)
        if not _is_hex64(h):
            raise ValueError(f"Calculated hash for {key} is not a valid 64-char hex: {h!r}")
        if h in ("0" * 64, "a" * 64) or any(ph in h.lower() for ph in _PLACEHOLDER_STRINGS):
            raise ValueError(f"Calculated hash for {key} is a placeholder: {h!r}")
        provenance[key] = h

    # Validate against schemas._REQUIRED_PROVENANCE_KEYS
    _validate_provenance(provenance)
    return provenance


# ---------------------------------------------------------------------------
# Independent Cache Validation Against Authoritative Inventory
# ---------------------------------------------------------------------------
def validate_cache_against_inventory(
    cache_dir: Path,
    scenario: str,
    seed: int,
    inventory_path: Path,
    expected_provenance: Dict[str, str],
) -> Dict[str, Any]:
    """
    Validates an attack cache against the independent cache_inventory_v2.json ledger.
    Guarantees:
      - Cache is found in the inventory with exact (scenario, seed) match.
      - Current on-disk SHA-256 of X_attacked.parquet, status.parquet, manifest.json,
        and completion.json match the inventory's recorded hashes.
      - Manifest's internal provenance fields match independently calculated expected_provenance.
      - Completion marker confirms COMPLETED state and matches seed and row count.
    Only after manifest hash is independently pinned against inventory is it returned.
    """
    if not inventory_path.exists():
        raise FileNotFoundError(f"Authoritative cache inventory missing: {inventory_path}")

    with open(inventory_path, "r") as f:
        inv = json.load(f)

    # Locate canonical entry
    matching = [
        c for c in inv.get("caches", [])
        if canonicalize_scenario(c.get("scenario", "")) == scenario and int(c.get("seed", -1)) == seed
    ]
    if len(matching) == 0:
        raise ValueError(f"Cache ({scenario}, seed {seed}) not found in independent inventory: {inventory_path}")
    if len(matching) > 1:
        raise ValueError(f"Ambiguous cache entries ({len(matching)}) for ({scenario}, seed {seed}) in inventory")

    inv_entry = matching[0]
    expected_artifact_hashes = inv_entry["artifact_hashes"]

    # Verify physical file existence and hashes against inventory
    for fname in ("X_attacked.parquet", "status.parquet", "manifest.json", "completion.json"):
        fpath = cache_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Cache {cache_dir.name} missing required file: {fname}")
        actual_hash = calculate_file_hash(fpath)
        expected_hash = expected_artifact_hashes.get(fname)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Cache {cache_dir.name} file {fname} SHA-256 mismatch vs independent inventory!\n"
                f"Expected: {expected_hash}\n"
                f"Actual:   {actual_hash}"
            )

    # Parse and validate pinned manifest
    manifest_path = cache_dir / "manifest.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Validate against expected provenance
    for prov_key, expected_val in expected_provenance.items():
        if prov_key in manifest:
            manifest_val = manifest[prov_key]
            if manifest_val != expected_val:
                raise ValueError(
                    f"Cache {cache_dir.name} manifest[{prov_key!r}] ({manifest_val}) disagrees "
                    f"with independently calculated production provenance ({expected_val})!"
                )

    # Validate completion marker
    comp_path = cache_dir / "completion.json"
    with open(comp_path, "r") as f:
        comp_data = json.load(f)

    if comp_data.get("completion_state") != "COMPLETED" or not comp_data.get("completed"):
        raise ValueError(f"Cache {cache_dir.name} completion.json indicates incomplete state")
    if canonicalize_scenario(comp_data.get("attack_scenario", "")) != scenario:
        raise ValueError(f"Cache {cache_dir.name} completion scenario mismatch")
    if int(comp_data.get("effective_seed", -1)) != seed:
        raise ValueError(f"Cache {cache_dir.name} completion seed mismatch")
    if int(comp_data.get("row_count", 0)) != 72000:
        raise ValueError(f"Cache {cache_dir.name} completion row count mismatch")

    return manifest


# ---------------------------------------------------------------------------
# Strengthened Frozen Configuration and Protocol Verification
# ---------------------------------------------------------------------------
def verify_frozen_configurations(
    repo_root: Path = REPO_ROOT,
    enforce_git: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Strengthened freeze check: Compares current bytes and SHA-256 values of every
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
    """Verifies that frozen RF model, roles manifest, batches manifest, and experiment config match protected hashes."""
    rf_path = repo_root / "artifacts/models/frozen_rf.joblib"
    roles_path = repo_root / "data/manifests/evaluation_roles.csv"
    batches_path = repo_root / "data/manifests/evaluation_batches.csv"
    exp_cfg_path = repo_root / "configs/experiment.yaml"

    hashes = {
        "RF": calculate_file_hash(rf_path),
        "Roles": calculate_file_hash(roles_path),
        "Batches": calculate_file_hash(batches_path),
        "ExperimentConfig": calculate_file_hash(exp_cfg_path),
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
# Strict Git Execution Cleanliness
# ---------------------------------------------------------------------------
def check_git_cleanliness(
    repo_root: Path = REPO_ROOT,
    output_dir: Optional[Path] = None,
    enforce_git: bool = True,
) -> Dict[str, Any]:
    """
    Enforces the strict Phase 10B Git cleanliness policy:
      - Freeze tag is an ancestor of HEAD.
      - Frozen src/ has zero diff relative to phase10-protocol-freeze.
      - Rejects any staged or unstaged tracked modification.
      - Rejects untracked source, test, script, config, or documentation files.
      - Explicitly allowlists only known generated artifacts (artifacts/caches/,
        historical ZIP bundles, logs, and the configured evaluation output dir).
    """
    git_dir = repo_root / ".git"
    if not git_dir.exists():
        if enforce_git:
            raise RuntimeError(f"Git repository not found at {repo_root}")
        return {"status": "SKIPPED_NO_GIT"}

    if not enforce_git:
        head_commit = "LOCAL_DEV"
        try:
            head_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
            ).strip()
        except Exception:
            pass
        return {
            "head_commit": head_commit,
            "clean": True,
            "allowed_untracked_count": 0,
            "enforced": False,
        }

    # 1. Freeze tag ancestry
    try:
        subprocess.check_call(
            ["git", "merge-base", "--is-ancestor", FREEZE_TAG, "HEAD"],
            cwd=repo_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        raise RuntimeError(f"Freeze tag {FREEZE_TAG} is not an ancestor of current HEAD")

    # 2. Frozen src/ check
    src_diff = subprocess.check_output(
        ["git", "diff", "--name-only", FREEZE_TAG, "--", "src/"],
        cwd=repo_root,
        text=True,
    ).strip()
    if src_diff:
        raise RuntimeError(f"Frozen src/ directory modified relative to {FREEZE_TAG}:\n{src_diff}")

    # 3. Status porcelain check
    status_lines = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        text=True,
    ).splitlines()

    dirty_tracked = []
    untracked_forbidden = []
    allowed_untracked = []

    out_dir_rel = str(output_dir.relative_to(repo_root)) if output_dir and output_dir.is_relative_to(repo_root) else "artifacts/evaluation_runs"

    for line in status_lines:
        if not line.strip():
            continue
        code = line[:2]
        path_str = line[3:].strip().strip('"')

        if code == "??":
            # Allowlist check
            is_allowed = (
                path_str.startswith("artifacts/caches/")
                or path_str.startswith(out_dir_rel)
                or path_str.startswith(".gemini/")
                or (path_str.endswith(".zip") and path_str.startswith("phase10"))
                or path_str.endswith(".log")
            )
            if is_allowed:
                allowed_untracked.append(path_str)
            else:
                # Disallow any untracked code, tests, scripts, configs, or docs
                if (
                    path_str.startswith(("src/", "tests/", "configs/", "scripts/", "docs/", "artifacts/reports/"))
                    or path_str.endswith((".py", ".yaml", ".yml", ".md", ".json", ".csv", ".parquet", ".sh", ".txt"))
                ):
                    untracked_forbidden.append(path_str)
                else:
                    untracked_forbidden.append(path_str)
        else:
            dirty_tracked.append(f"{code}:{path_str}")

    if dirty_tracked:
        raise RuntimeError(
            f"Git execution cleanliness violated: tracked files are modified: {dirty_tracked}. "
            "Working tree must be committed."
        )
    if untracked_forbidden:
        raise RuntimeError(
            f"Git execution cleanliness violated: untracked forbidden files detected: {untracked_forbidden}. "
            "Must be committed or removed."
        )

    head_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()

    return {
        "head_commit": head_commit,
        "clean": True,
        "allowed_untracked_count": len(allowed_untracked),
    }


# ---------------------------------------------------------------------------
# Unsuppressed Disk Space Gate
# ---------------------------------------------------------------------------
def check_disk_space(repo_root: Path = REPO_ROOT, min_gb: float = 10.0) -> float:
    """
    Checks available free disk space. Only catches OSError on obtaining statistics.
    Raises RuntimeError and aborts if free space is below min_gb.
    """
    try:
        total, used, free = shutil.disk_usage(repo_root)
        free_gb = free / (1024**3)
    except OSError as e:
        raise RuntimeError(f"Failed to obtain filesystem disk usage: {e}") from e

    if free_gb < min_gb:
        raise RuntimeError(f"Insufficient disk space: {free_gb:.2f} GB available < {min_gb:.2f} GB required")

    return free_gb


# ---------------------------------------------------------------------------
# Strengthened Manifest Cross-Validation & Batch Preparation
# ---------------------------------------------------------------------------
def prepare_evaluation_batches(
    manifests_dir: Path,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, str]]:
    """
    Loads and rigorously cross-validates evaluation_roles.csv and evaluation_batches.csv:
      - Asserts required columns exist in both manifests.
      - Asserts roles.csv has exactly 90,000 rows (18,000 crafting, 72,000 measurement).
      - Asserts batches.csv has exactly 72,000 rows.
      - Asserts batch positions are unique and equal exactly the measurement positions.
      - Asserts zero overlap with crafting positions.
      - Asserts batch IDs normalize strictly to 0..143 (each with exactly 500 rows).
      - Asserts y_binary is strictly in {0, 1} and agrees between manifests.
      - Asserts composite identities (_source_file, _raw_row_idx) agree.
      - Deterministically sorts by eval_position and builds measurement_idx = 0..71999.
    """
    roles_path = manifests_dir / "evaluation_roles.csv"
    batches_path = manifests_dir / "evaluation_batches.csv"

    if not roles_path.exists():
        raise FileNotFoundError(f"Missing {roles_path}")
    if not batches_path.exists():
        raise FileNotFoundError(f"Missing {batches_path}")

    roles = pd.read_csv(roles_path)
    batches = pd.read_csv(batches_path)

    # 1. Required columns
    roles_req = {"eval_position", "_source_file", "_raw_row_idx", "y_binary", "attack_family", "role"}
    batches_req = {"eval_position", "_source_file", "_raw_row_idx", "y_binary", "attack_family", "role", "batch_id", "within_batch_position"}

    if not roles_req.issubset(roles.columns):
        raise ValueError(f"evaluation_roles.csv missing columns: {roles_req - set(roles.columns)}")
    if not batches_req.issubset(batches.columns):
        raise ValueError(f"evaluation_batches.csv missing columns: {batches_req - set(batches.columns)}")

    # 2. Exact role partition counts
    if len(roles) != 90000:
        raise ValueError(f"evaluation_roles.csv has {len(roles)} rows, expected 90000")
    if len(batches) != 72000:
        raise ValueError(f"evaluation_batches.csv has {len(batches)} rows, expected 72000")

    crafting_roles = roles[roles["role"] == "crafting"]
    measurement_roles = roles[roles["role"] == "measurement"]

    if len(crafting_roles) != 18000:
        raise ValueError(f"Expected 18000 crafting roles, got {len(crafting_roles)}")
    if len(measurement_roles) != 72000:
        raise ValueError(f"Expected 72000 measurement roles, got {len(measurement_roles)}")

    # 3. Position uniqueness and exact alignment
    if batches["eval_position"].duplicated().any():
        raise ValueError("Duplicate eval_position found in evaluation_batches.csv")
    if roles["eval_position"].duplicated().any():
        raise ValueError("Duplicate eval_position found in evaluation_roles.csv")

    batch_pos_set = set(batches["eval_position"])
    meas_pos_set = set(measurement_roles["eval_position"])
    craft_pos_set = set(crafting_roles["eval_position"])

    if batch_pos_set != meas_pos_set:
        raise ValueError("evaluation_batches.csv positions do not exactly match measurement role positions")
    if len(batch_pos_set.intersection(craft_pos_set)) > 0:
        raise ValueError("Crafting role positions detected inside evaluation_batches.csv")

    # 4. Batch ID normalization (1..144 -> 0..143)
    b_ids = batches["batch_id"].values
    if b_ids.min() == 1 and b_ids.max() == 144:
        norm_b_ids = b_ids - 1
    elif b_ids.min() == 0 and b_ids.max() == 143:
        norm_b_ids = b_ids
    else:
        raise ValueError(f"evaluation_batches.csv batch_id out of expected bounds: [{b_ids.min()}, {b_ids.max()}]")

    if set(norm_b_ids) != set(range(144)):
        raise ValueError("Normalized batch IDs must span exactly 0..143")

    b_counts = pd.Series(norm_b_ids).value_counts()
    if not (b_counts == 500).all():
        raise ValueError("Every batch in evaluation_batches.csv must contain exactly 500 rows")

    # 5. Binary y_binary and composite identity alignment
    if not set(roles["y_binary"].unique()).issubset({0, 1}):
        raise ValueError("evaluation_roles.csv y_binary contains non-binary values")
    if not set(batches["y_binary"].unique()).issubset({0, 1}):
        raise ValueError("evaluation_batches.csv y_binary contains non-binary values")

    merged = batches.merge(measurement_roles, on="eval_position", suffixes=("_batch", "_role"))
    if not (merged["y_binary_batch"] == merged["y_binary_role"]).all():
        raise ValueError("y_binary disagrees between evaluation_batches.csv and evaluation_roles.csv")
    if not (merged["_source_file_batch"] == merged["_source_file_role"]).all():
        raise ValueError("_source_file disagrees between evaluation_batches.csv and evaluation_roles.csv")
    if not (merged["_raw_row_idx_batch"] == merged["_raw_row_idx_role"]).all():
        raise ValueError("_raw_row_idx disagrees between evaluation_batches.csv and evaluation_roles.csv")

    # 6. Deterministic sorting and measurement_idx assignment
    sorted_batches = batches.sort_values("eval_position").reset_index(drop=True)
    sorted_batches["batch_id"] = sorted_batches["batch_id"] - 1 if sorted_batches["batch_id"].min() == 1 else sorted_batches["batch_id"]
    sorted_batches["measurement_idx"] = np.arange(72000, dtype=int)

    y_measurement = sorted_batches["y_binary"].values.astype(int)

    manifest_hashes = {
        "evaluation_roles_hash": calculate_file_hash(roles_path),
        "evaluation_batches_hash": calculate_file_hash(batches_path),
    }

    return sorted_batches, y_measurement, manifest_hashes


# ---------------------------------------------------------------------------
# Defense and Policy Factory (Derived from defenses.yaml)
# ---------------------------------------------------------------------------
def create_defense_adapter(
    defense_name: str,
    feature_names: List[str],
    modifiable_mask: np.ndarray,
    training_bounds: pd.DataFrame,
    benign_profile: pd.DataFrame,
    rf_model: Any,
    defenses_yaml: Dict[str, Any],
) -> Any:
    """Instantiates defense adapter using calibrated baseline from defenses.yaml."""
    d = canonicalize_defense(defense_name)
    def_cfg = defenses_yaml[d]

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
            epsilon_base=float(def_cfg["epsilon_base"]),
            alpha=float(def_cfg.get("alpha", 0.5)),
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
            ensemble_size=int(def_cfg.get("ensemble_size", 11)),
        )
        return RSDefenseAdapter(
            rs=rs,
            predict_func=rf_model.predict,
            chunk_size=int(def_cfg.get("chunk_size", 100)),
        )
    else:
        raise ValueError(f"Unsupported defense: {defense_name}")


def create_policy_controller(
    controller_config_id: str,
    defense_name: str,
    controllers_yaml: Dict[str, Any],
    defenses_yaml: Dict[str, Any],
) -> Any:
    """
    Instantiates FixedIntensityPolicy for Base (deriving intensity and bounds
    directly from frozen defenses.yaml) or RecallAwareController for C1..C7.
    """
    d = canonicalize_defense(defense_name)
    def_cfg = defenses_yaml[d]

    if controller_config_id == "Base":
        if d == "afp":
            base_val = float(def_cfg["epsilon_base"])
        elif d == "randomized_smoothing":
            base_val = float(def_cfg["sigma"])
        elif d == "feature_squeezing":
            base_val = float(def_cfg["squeezing_intensity"])
        else:
            raise ValueError(f"Unknown defense {defense_name}")

        min_val = float(def_cfg["intensity_min"])
        max_val = float(def_cfg["intensity_max"])

        return FixedIntensityPolicy(
            config_id="Base",
            fixed_intensity=base_val,
            intensity_min=min_val,
            intensity_max=max_val,
        )
    else:
        cfg = controllers_yaml["controller_configurations"][controller_config_id]
        return RecallAwareController(
            config=cfg,
            defense_config=def_cfg,
            defense_name=d,
            zero_division_value=0.0,
        )


# ---------------------------------------------------------------------------
# Completed Run Reuse and Quarantine Validation
# ---------------------------------------------------------------------------
def validate_completed_run(
    run_dir: Path,
    expected_row: Optional[pd.Series] = None,
    expected_provenance: Optional[Dict[str, str]] = None,
    expected_cache_identity: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Validates that a run directory is complete, untampered, and belongs to the
    expected matrix row.
    """
    if not run_dir.exists() or not run_dir.is_dir():
        return False, "Directory does not exist or is not a directory"

    # Reject unexpected output files
    expected_files = {"config.json", "confusion.json", "scores.json", "run_summary.json", "completion.json"}
    actual_files = {p.name for p in run_dir.iterdir() if not p.name.startswith(".")}
    if actual_files != expected_files:
        unexpected = actual_files - expected_files
        missing = expected_files - actual_files
        reasons = []
        if missing:
            reasons.append(f"Missing required files: {sorted(missing)}")
        if unexpected:
            reasons.append(f"Unexpected files: {sorted(unexpected)}")
        return False, "; ".join(reasons)

    # Verify completion.json written last
    comp_path = run_dir / "completion.json"
    comp_mtime = comp_path.stat().st_mtime
    for fname in ("config.json", "confusion.json", "scores.json", "run_summary.json"):
        if (run_dir / fname).stat().st_mtime > comp_mtime + 0.1:
            return False, f"{fname} was modified after completion.json"

    try:
        with open(comp_path, "r") as f:
            c_data = json.load(f)
        marker = CompletionMarker(**c_data)
        _validate_provenance(marker.provenance_hashes)
    except Exception as e:
        return False, f"completion.json validation failed: {e}"

    summary_path = run_dir / "run_summary.json"
    try:
        with open(summary_path, "r") as f:
            s_data = json.load(f)
        summary = RunSummary(**s_data)
    except Exception as e:
        return False, f"run_summary.json validation failed: {e}"

    if run_dir.name != marker.run_id or run_dir.name != summary.run_id:
        return False, f"Run ID mismatch: dir={run_dir.name}, marker={marker.run_id}, summary={summary.run_id}"

    # Match against expected matrix row
    if expected_row is not None:
        if run_dir.name != expected_row["run_id"]:
            return False, f"Run ID mismatch with expected matrix row: {expected_row['run_id']}"
        if int(summary.seed) != int(expected_row["seed"]):
            return False, f"Seed mismatch: expected {expected_row['seed']}, got {summary.seed}"
        if canonicalize_scenario(summary.attack_scenario) != canonicalize_scenario(expected_row["attack_scenario"]):
            return False, f"Scenario mismatch: expected {expected_row['attack_scenario']}, got {summary.attack_scenario}"
        if canonicalize_defense(summary.defense) != canonicalize_defense(expected_row["defense_name"]):
            return False, f"Defense mismatch: expected {expected_row['defense_name']}, got {summary.defense}"
        if summary.config_id != expected_row["controller_config_id"]:
            return False, f"Controller config mismatch: expected {expected_row['controller_config_id']}, got {summary.config_id}"

    # Match expected provenance
    if expected_provenance is not None:
        if set(marker.provenance_hashes.keys()) != set(expected_provenance.keys()):
            return False, f"completion.json provenance keys do not match expected production provenance keys"
        for k, exp_h in expected_provenance.items():
            if marker.provenance_hashes.get(k) != exp_h:
                return False, f"Provenance hash mismatch for {k}: expected {exp_h}, got {marker.provenance_hashes.get(k)}"

    # Match expected cache identity
    if expected_cache_identity is not None:
        if summary.cache_identity != expected_cache_identity:
            return False, "run_summary.cache_identity does not match expected cache identity"

    # Reopen and validate batch files
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
# Pointer-Only Alias Publication and Reuse Validation
# ---------------------------------------------------------------------------
def validate_completed_alias(
    alias_dir: Path,
    expected_row: pd.Series,
    expected_provenance: Dict[str, str],
    output_dir: Path,
    matrix: Optional[pd.DataFrame] = None,
    expected_cache_identity: Optional[Dict[str, Any]] = None,
    caches_dir: Optional[Path] = None,
    inventory_path: Optional[Path] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Validates an existing alias directory. It must be a pointer-only artifact
    containing ONLY alias_pointer.json and completion.json.

    Rigorous checks:
      - require alias CompletionMarker provenance to equal expected_provenance exactly;
      - validate pointer config_id;
      - validate pointer target_provenance exactly;
      - validate target_run_summary_sha256;
      - validate target_completion_sha256;
      - ensure completion.json was written after alias_pointer.json;
      - validate the target against its exact primary matrix row;
      - validate the target against its independently pinned expected cache identity;
      - ensure target controller config is C1;
      - reject any mismatch before publishing or reusing the alias.
    """
    if not alias_dir.exists() or not alias_dir.is_dir():
        return False, "Alias directory does not exist or is not a directory"

    actual_files = {p.name for p in alias_dir.iterdir() if not p.name.startswith(".")}
    expected_files = {"alias_pointer.json", "completion.json"}
    if actual_files != expected_files:
        return False, f"Alias directory contains invalid files (expected {expected_files}, got {actual_files})"

    # If expected_cache_identity was not explicitly passed, independently pin if possible
    if expected_cache_identity is None and caches_dir is not None and inventory_path is not None and inventory_path.exists():
        scen = canonicalize_scenario(expected_row["attack_scenario"])
        seed = int(expected_row["seed"])
        cdir = caches_dir / f"{scen}_{seed}"
        if cdir.exists():
            try:
                expected_cache_identity = validate_cache_against_inventory(
                    cache_dir=cdir,
                    scenario=scen,
                    seed=seed,
                    inventory_path=inventory_path,
                    expected_provenance=expected_provenance,
                )
            except Exception as e:
                return False, f"Target cache validation against inventory failed: {e}"

    target_run_provenance = dict(expected_provenance)
    cache_manifest_hash = expected_cache_identity.get("manifest_sha256") if expected_cache_identity else None
    if cache_manifest_hash is None and caches_dir is not None:
        scen = canonicalize_scenario(expected_row["attack_scenario"])
        seed = int(expected_row["seed"])
        cdir = caches_dir / f"{scen}_{seed}"
        if cdir.exists():
            cache_manifest_hash = calculate_file_hash(cdir / "manifest.json")
    if cache_manifest_hash is not None:
        target_run_provenance["cache_manifest_hash"] = cache_manifest_hash

    comp_path = alias_dir / "completion.json"
    ptr_path = alias_dir / "alias_pointer.json"

    # Ensure completion.json was written after alias_pointer.json
    if ptr_path.stat().st_mtime > comp_path.stat().st_mtime + 0.1:
        return False, "alias_pointer.json was modified after completion.json"

    # Validate completion.json
    try:
        with open(comp_path, "r") as f:
            c_data = json.load(f)
        marker = CompletionMarker(**c_data)
        _validate_provenance(marker.provenance_hashes)
        if marker.run_id != expected_row["run_id"]:
            return False, f"Alias completion run_id mismatch: expected {expected_row['run_id']}, got {marker.run_id}"
        if marker.provenance_hashes != target_run_provenance:
            return False, "Alias completion provenance does not match target run provenance exactly"
    except Exception as e:
        return False, f"Alias completion.json failed validation: {e}"

    # Validate alias_pointer.json
    try:
        with open(ptr_path, "r") as f:
            ptr = json.load(f)
        if ptr.get("run_id") != expected_row["run_id"]:
            return False, f"Alias pointer run_id mismatch: expected {expected_row['run_id']}, got {ptr.get('run_id')}"
        if ptr.get("alias_for_run_id") != expected_row["alias_for_run_id"]:
            return False, f"Alias target mismatch: expected {expected_row['alias_for_run_id']}, got {ptr.get('alias_for_run_id')}"
        if int(ptr.get("seed", -1)) != int(expected_row["seed"]):
            return False, "Alias pointer seed mismatch"
        if canonicalize_scenario(ptr.get("attack_scenario", "")) != canonicalize_scenario(expected_row["attack_scenario"]):
            return False, "Alias pointer scenario mismatch"
        if canonicalize_defense(ptr.get("defense", "")) != canonicalize_defense(expected_row["defense_name"]):
            return False, "Alias pointer defense mismatch"
        if ptr.get("config_id") != expected_row["controller_config_id"]:
            return False, f"Alias pointer config_id mismatch: expected {expected_row['controller_config_id']}, got {ptr.get('config_id')}"
        if ptr.get("target_provenance") != target_run_provenance:
            return False, "Alias pointer target_provenance does not match target run provenance exactly"
    except Exception as e:
        return False, f"alias_pointer.json validation failed: {e}"

    # Resolve target matrix row
    target_id = str(expected_row["alias_for_run_id"])
    target_dir = output_dir / target_id
    if not target_dir.exists():
        return False, f"Alias target {target_id} does not exist in {output_dir}"

    target_row = None
    if matrix is not None:
        target_rows = matrix[matrix["run_id"] == target_id]
        if len(target_rows) == 0:
            return False, f"Alias target {target_id} not found in matrix"
        target_row = target_rows.iloc[0]
        if target_row["controller_config_id"] != "C1":
            return False, f"Target {target_id} controller config is not C1: got {target_row['controller_config_id']}"
    else:
        target_row = pd.Series({
            "run_id": target_id,
            "seed": int(expected_row["seed"]),
            "attack_scenario": str(expected_row["attack_scenario"]),
            "defense_name": str(expected_row["defense_name"]),
            "controller_config_id": "C1",
            "is_alias": False,
            "alias_for_run_id": None,
        })

    # Validate target run exists and is valid against exact primary row, cache identity, and provenance

    # Validate target run exists and is valid against exact primary row, cache identity, and provenance
    is_target_valid, err = validate_completed_run(
        target_dir,
        expected_row=target_row,
        expected_provenance=target_run_provenance,
        expected_cache_identity=expected_cache_identity,
    )
    if not is_target_valid:
        return False, f"Alias target {target_dir.name} is invalid: {err}"

    # Ensure target controller config is C1 in target's run_summary
    try:
        with open(target_dir / "run_summary.json", "r") as f:
            target_summary_data = json.load(f)
        if target_summary_data.get("config_id") != "C1":
            return False, f"Target run_summary controller config is not C1: got {target_summary_data.get('config_id')}"
    except Exception as e:
        return False, f"Target run_summary validation failed: {e}"

    # Validate target_run_summary_sha256 and target_completion_sha256
    target_summary_hash = calculate_file_hash(target_dir / "run_summary.json")
    if ptr.get("target_run_summary_sha256") != target_summary_hash:
        return False, f"Alias pointer target_run_summary_sha256 mismatch: expected {target_summary_hash}, got {ptr.get('target_run_summary_sha256')}"

    target_comp_hash = calculate_file_hash(target_dir / "completion.json")
    if ptr.get("target_completion_sha256") != target_comp_hash:
        return False, f"Alias pointer target_completion_sha256 mismatch: expected {target_comp_hash}, got {ptr.get('target_completion_sha256')}"

    return True, None


def publish_alias(
    alias_row: pd.Series,
    output_dir: Path,
    expected_provenance: Dict[str, str],
    matrix: Optional[pd.DataFrame] = None,
    expected_cache_identity: Optional[Dict[str, Any]] = None,
    caches_dir: Optional[Path] = None,
    inventory_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Publishes an alias as a pointer-only artifact atomically.
    Strictly verifies:
      - Target run exists, matches exact primary matrix row, expected provenance, and cache identity.
      - Target controller config is strictly C1.
      - Writes alias_pointer.json first, then completion.json last.
    """
    alias_id = str(alias_row["run_id"])
    target_id = str(alias_row["alias_for_run_id"])
    target_dir = output_dir / target_id

    if not target_dir.exists():
        raise RuntimeError(f"Alias target {target_id} does not exist in {output_dir}")

    target_row = None
    if matrix is not None:
        target_rows = matrix[matrix["run_id"] == target_id]
        if len(target_rows) == 0:
            raise ValueError(f"Alias target {target_id} not found in matrix")
        target_row = target_rows.iloc[0]
        if target_row["controller_config_id"] != "C1":
            raise ValueError(f"Alias target {target_id} controller config is not C1: got {target_row['controller_config_id']}")
    else:
        target_row = pd.Series({
            "run_id": target_id,
            "seed": int(alias_row["seed"]),
            "attack_scenario": str(alias_row["attack_scenario"]),
            "defense_name": str(alias_row["defense_name"]),
            "controller_config_id": "C1",
            "is_alias": False,
            "alias_for_run_id": None,
        })

    # If expected_cache_identity was not explicitly passed, independently pin if possible
    if expected_cache_identity is None and caches_dir is not None and inventory_path is not None and inventory_path.exists():
        scen = canonicalize_scenario(alias_row["attack_scenario"])
        seed = int(alias_row["seed"])
        cdir = caches_dir / f"{scen}_{seed}"
        if cdir.exists():
            expected_cache_identity = validate_cache_against_inventory(
                cache_dir=cdir,
                scenario=scen,
                seed=seed,
                inventory_path=inventory_path,
                expected_provenance=expected_provenance,
            )

    target_run_provenance = dict(expected_provenance)
    cache_manifest_hash = expected_cache_identity.get("manifest_sha256") if expected_cache_identity else None
    if cache_manifest_hash is None and caches_dir is not None:
        scen = canonicalize_scenario(alias_row["attack_scenario"])
        seed = int(alias_row["seed"])
        cdir = caches_dir / f"{scen}_{seed}"
        if cdir.exists():
            cache_manifest_hash = calculate_file_hash(cdir / "manifest.json")
    if cache_manifest_hash is not None:
        target_run_provenance["cache_manifest_hash"] = cache_manifest_hash

    is_valid, err = validate_completed_run(
        target_dir,
        expected_row=target_row,
        expected_provenance=target_run_provenance,
        expected_cache_identity=expected_cache_identity,
    )
    if not is_valid:
        raise ValueError(f"Alias target {target_id} is invalid: {err}")

    # Ensure alias tuple exactly matches target tuple and target is C1
    target_summary_path = target_dir / "run_summary.json"
    with open(target_summary_path, "r") as f:
        target_summary_data = json.load(f)
    if target_summary_data.get("config_id") != "C1":
        raise ValueError(f"Alias target {target_id} summary config_id is not C1: got {target_summary_data.get('config_id')}")
    if int(target_summary_data["seed"]) != int(alias_row["seed"]):
        raise ValueError(f"Alias tuple seed mismatch: alias={alias_row['seed']}, target={target_summary_data['seed']}")
    if canonicalize_scenario(target_summary_data["attack_scenario"]) != canonicalize_scenario(alias_row["attack_scenario"]):
        raise ValueError(f"Alias tuple scenario mismatch: alias={alias_row['attack_scenario']}, target={target_summary_data['attack_scenario']}")
    if canonicalize_defense(target_summary_data["defense"]) != canonicalize_defense(alias_row["defense_name"]):
        raise ValueError(f"Alias tuple defense mismatch: alias={alias_row['defense_name']}, target={target_summary_data['defense']}")

    # Check for existing alias directory
    alias_dir = output_dir / alias_id
    if alias_dir.exists():
        is_alias_valid, a_err = validate_completed_alias(
            alias_dir=alias_dir,
            expected_row=alias_row,
            expected_provenance=expected_provenance,
            output_dir=output_dir,
            matrix=matrix,
            expected_cache_identity=expected_cache_identity,
            caches_dir=caches_dir,
            inventory_path=inventory_path,
        )
        if is_alias_valid:
            return {"run_id": alias_id, "status": "REUSED_ALIAS", "alias_for": target_id}
        else:
            quarantine_run_directory(alias_dir, f"Invalid alias: {a_err}")

    # Staging directory on the same filesystem
    staging_dir = output_dir / f".staging_{alias_id}_{uuid.uuid4().hex}"
    staging_dir.mkdir(parents=True, exist_ok=False)

    try:
        target_summary_hash = calculate_file_hash(target_dir / "run_summary.json")
        target_comp_hash = calculate_file_hash(target_dir / "completion.json")

        alias_pointer = {
            "run_id": alias_id,
            "seed": int(alias_row["seed"]),
            "attack_scenario": canonicalize_scenario(alias_row["attack_scenario"]),
            "defense": canonicalize_defense(alias_row["defense_name"]),
            "config_id": str(alias_row["controller_config_id"]),
            "is_alias": True,
            "alias_for_run_id": target_id,
            "target_run_summary_sha256": target_summary_hash,
            "target_completion_sha256": target_comp_hash,
            "target_provenance": target_run_provenance,
            "published_at": datetime.datetime.utcnow().isoformat() + "Z",
        }
        with open(staging_dir / "alias_pointer.json", "w") as f:
            json.dump(alias_pointer, f, indent=2)

        marker = CompletionMarker(
            run_id=alias_id,
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
            provenance_hashes=target_run_provenance,
        )
        with open(staging_dir / "completion.json", "w") as f:
            json.dump(dataclasses.asdict(marker), f, indent=2)

        # Ensure completion.json was written after alias_pointer.json
        now_ts = datetime.datetime.utcnow().timestamp()
        os.utime(staging_dir / "alias_pointer.json", (now_ts, now_ts))
        os.utime(staging_dir / "completion.json", (now_ts + 1.0, now_ts + 1.0))

        staging_dir.rename(alias_dir)
        return {"run_id": alias_id, "status": "PUBLISHED_ALIAS", "alias_for": target_id}
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def resolve_and_validate_execution_plan(
    matrix: pd.DataFrame,
    filtered_matrix: pd.DataFrame,
    output_dir: Path,
    expected_provenance: Dict[str, str],
    max_runs: Optional[int] = None,
    expected_cache_identities: Optional[Dict[str, Dict[str, Any]]] = None,
    caches_dir: Optional[Path] = None,
    inventory_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validates and resolves the execution plan:
      - Validates max_runs > 0.
      - For each alias: ensures that its primary target run is validated against its
        exact primary matrix row, expected provenance, and expected cache identity.
      - Targets are not treated as reusable based only on structural validation.
        If validation fails, the target is automatically scheduled for full computation ahead of the alias.
      - Returns (unique_runs_to_execute, alias_runs_to_publish).
    """
    if max_runs is not None:
        if max_runs <= 0:
            raise ValueError(f"--max-runs must be a strictly positive integer, got {max_runs}")
        filtered_matrix = filtered_matrix.head(max_runs)

    all_target_ids = set()
    for _, row in filtered_matrix[filtered_matrix["is_alias"]].iterrows():
        target_id = row["alias_for_run_id"]
        all_target_ids.add(target_id)

    # Check targets not in filtered_matrix
    missing_targets = []
    for tid in sorted(all_target_ids):
        target_rows = matrix[matrix["run_id"] == tid]
        if len(target_rows) == 0:
            raise ValueError(f"Primary target run {tid} not found in matrix!")
        target_row = target_rows.iloc[0]

        target_dir = output_dir / tid
        exp_cache_id = None
        if expected_cache_identities and tid in expected_cache_identities:
            exp_cache_id = expected_cache_identities[tid]
        elif caches_dir is not None and inventory_path is not None and inventory_path.exists():
            scen = canonicalize_scenario(target_row["attack_scenario"])
            seed = int(target_row["seed"])
            cdir = caches_dir / f"{scen}_{seed}"
            if cdir.exists():
                try:
                    exp_cache_id = validate_cache_against_inventory(
                        cache_dir=cdir,
                        scenario=scen,
                        seed=seed,
                        inventory_path=inventory_path,
                        expected_provenance=expected_provenance,
                    )
                except Exception:
                    exp_cache_id = None

        target_run_provenance = dict(expected_provenance)
        cache_manifest_hash = exp_cache_id.get("manifest_sha256") if exp_cache_id else None
        if cache_manifest_hash is None and caches_dir is not None:
            scen = canonicalize_scenario(target_row["attack_scenario"])
            seed = int(target_row["seed"])
            cdir = caches_dir / f"{scen}_{seed}"
            if cdir.exists():
                cache_manifest_hash = calculate_file_hash(cdir / "manifest.json")
        if cache_manifest_hash is not None:
            target_run_provenance["cache_manifest_hash"] = cache_manifest_hash

        is_target_valid, _ = validate_completed_run(
            target_dir,
            expected_row=target_row,
            expected_provenance=target_run_provenance,
            expected_cache_identity=exp_cache_id,
        )
        if not is_target_valid and tid not in filtered_matrix["run_id"].values:
            missing_targets.append(target_row)

    if missing_targets:
        targets_df = pd.DataFrame(missing_targets)
        combined = pd.concat([targets_df, filtered_matrix]).drop_duplicates(subset=["run_id"])
    else:
        combined = filtered_matrix

    unique_runs = combined[~combined["is_alias"]].reset_index(drop=True)
    alias_runs = combined[combined["is_alias"]].reset_index(drop=True)

    return unique_runs, alias_runs


# ---------------------------------------------------------------------------
# Dynamic Evaluation Matrix Derivation & Validation
# ---------------------------------------------------------------------------
def derive_and_validate_matrix(configs_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Derives complete evaluation matrix and validates exact reference counts."""
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
# Preflight Checker
# ---------------------------------------------------------------------------
def run_preflight(
    configs_dir: Path = REPO_ROOT / "configs",
    manifests_dir: Path = REPO_ROOT / "data/manifests",
    models_dir: Path = REPO_ROOT / "artifacts/models",
    preprocessors_dir: Path = REPO_ROOT / "artifacts/preprocessors",
    caches_dir: Path = REPO_ROOT / "artifacts/caches",
    inventory_path: Path = REPO_ROOT / "artifacts/reports/cache_inventory_v2.json",
    output_dir: Path = REPO_ROOT / "artifacts/evaluation_runs",
    repo_root: Path = REPO_ROOT,
    enforce_git: bool = True,
) -> Dict[str, Any]:
    """
    Performs complete, non-mutating preflight integrity checks:
      1. Git clean status & freeze tag ancestry.
      2. Configuration and protocol freeze integrity.
      3. Protected model and manifest SHA-256 validation.
      4. Manifest cross-validation (roles & batches) without opening evaluation Parquets.
      5. Complete production provenance construction.
      6. Dynamic evaluation matrix validation.
      7. All 15 official attack caches independently pinned against inventory.
      8. Unsuppressed available storage check.
    """
    print("=" * 78)
    print("PHASE 10D v2 NON-MUTATING PREFLIGHT VERIFICATION")
    print("=" * 78)

    preflight_report: Dict[str, Any] = {"timestamp": datetime.datetime.utcnow().isoformat() + "Z"}

    # 1. Git verification
    print("Checking Git cleanliness...")
    git_res = check_git_cleanliness(repo_root=repo_root, output_dir=output_dir, enforce_git=enforce_git)
    preflight_report["git"] = git_res
    print(f"  Git HEAD: {git_res.get('head_commit')} (Clean: {git_res.get('clean')})")

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
        print(f"  {k:16s}: {h}")

    # 4. Manifest cross-validation (roles & batches) without opening evaluation Parquets
    print("\nCross-validating evaluation roles and batches manifests...")
    resolved_batches, y_meas, meas_positions = prepare_evaluation_batches(manifests_dir)
    total_batches = int(resolved_batches["batch_id"].nunique())
    preflight_report["manifests"] = {
        "status": "PASS",
        "batches_count": total_batches,
        "measurement_rows": len(y_meas),
    }
    print(f"  Validated {total_batches} batches (exactly 144 batches of 500 rows, {len(y_meas):,} samples).")

    # 5. Production Provenance Construction
    print("\nBuilding complete production provenance from on-disk artifacts...")
    provenance = build_production_provenance(
        repo_root=repo_root,
        configs_dir=configs_dir,
        manifests_dir=manifests_dir,
        models_dir=models_dir,
        preprocessors_dir=preprocessors_dir,
    )
    preflight_report["production_provenance"] = provenance
    print(f"  Constructed all {len(provenance)} provenance hashes successfully.")

    # 5. Matrix derivation
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

    # 6. Attack caches pinned against independent inventory
    print("\nValidating 15 official attack caches against independent inventory...")
    cache_pairs = sorted(
        list(
            {
                (canonicalize_scenario(r["attack_scenario"]), int(r["seed"]))
                for _, r in unique_execs.iterrows()
            }
        )
    )
    for scen, seed in cache_pairs:
        cdir = caches_dir / f"{scen}_{seed}"
        validate_cache_against_inventory(
            cache_dir=cdir,
            scenario=scen,
            seed=seed,
            inventory_path=inventory_path,
            expected_provenance=provenance,
        )
    print(f"  All {len(cache_pairs)} official attack caches validated against {inventory_path.name}.")

    # 7. Unsuppressed disk space check
    free_gb = check_disk_space(repo_root=repo_root, min_gb=10.0)
    preflight_report["disk_free_gb"] = free_gb
    print(f"\nAvailable Disk Space: {free_gb:.2f} GB (minimum required: 10.00 GB)")

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
    inventory_path: Path,
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
    Executes a single unique run with dry CompletionMarker pre-validation,
    two-stage atomic publication, and quarantine.
    """
    run_id = str(run_row["run_id"])
    seed = int(run_row["seed"])
    attack_scenario = canonicalize_scenario(run_row["attack_scenario"])
    defense_name = canonicalize_defense(run_row["defense_name"])
    controller_config_id = str(run_row["controller_config_id"])

    final_run_dir = output_dir / run_id

    # Prepare Run Provenance Hashes
    cache_slug = f"{attack_scenario}_{seed}"
    cache_dir = caches_dir / cache_slug

    # Validate cache against independent inventory FIRST
    cache_manifest = validate_cache_against_inventory(
        cache_dir=cache_dir,
        scenario=attack_scenario,
        seed=seed,
        inventory_path=inventory_path,
        expected_provenance=provenance_hashes,
    )

    run_prov = dict(provenance_hashes)
    run_prov["cache_manifest_hash"] = calculate_file_hash(cache_dir / "manifest.json")

    # Dry-run CompletionMarker validation BEFORE starting any batch work
    dry_marker = CompletionMarker(
        run_id=run_id,
        timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        provenance_hashes=run_prov,
    )
    _validate_provenance(dry_marker.provenance_hashes)

    # Check for existing run directory reuse
    if final_run_dir.exists():
        is_valid, err = validate_completed_run(
            final_run_dir,
            expected_row=run_row,
            expected_provenance=run_prov,
            expected_cache_identity=cache_manifest,
        )
        if is_valid:
            print(f"  [REUSE] Run {run_id} already exists and passed validation. Reusing.")
            return {"run_id": run_id, "status": "REUSED"}
        else:
            print(f"  [QUARANTINE] Run {run_id} exists but is incomplete or invalid: {err}.")
            quarantine_dir = quarantine_run_directory(final_run_dir, err)
            print(f"  Quarantined to {quarantine_dir.name}. Restarting from batch 0.")

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
        defenses_yaml=defenses_yaml,
    )

    # Prepare Controller Policy
    policy_controller = create_policy_controller(
        controller_config_id=controller_config_id,
        defense_name=defense_name,
        controllers_yaml=controllers_yaml,
        defenses_yaml=defenses_yaml,
    )

    # Staging directory on the same filesystem
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

        runner_out = staging_dir / run_id
        if not runner_out.exists():
            raise RuntimeError(f"Runner failed to create output directory {runner_out}")

        runner_out.rename(final_run_dir)
        shutil.rmtree(staging_dir, ignore_errors=True)

        return {"run_id": run_id, "status": "COMPLETED"}

    except Exception as e:
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
        description="Phase 10D v2 Official Evaluation Orchestrator (Safe by Default).",
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
        help="Optional maximum number of runs to execute (must be positive).",
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
        "--inventory-path",
        type=Path,
        default=REPO_ROOT / "artifacts/reports/cache_inventory_v2.json",
        help="Path to authoritative cache inventory ledger.",
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
    parser.add_argument(
        "--enforce-git",
        dest="enforce_git",
        action="store_true",
        default=True,
        help="Enforce clean git working tree and freeze ancestry (default: True).",
    )
    parser.add_argument(
        "--no-enforce-git",
        dest="enforce_git",
        action="store_false",
        help="Bypass git cleanliness checks (for local testing/development only).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # 0. Official Execution Mode Strict Invariant Checks
    if args.execute:
        # Prohibit Git-gate bypass during execution
        if not args.enforce_git:
            raise ValueError(
                "Cannot combine --execute with --no-enforce-git: official execution strictly mandates clean Git state and freeze ancestry."
            )

        # Bind official execution to canonical inputs
        canonical_configs = (REPO_ROOT / "configs").resolve()
        if args.configs_dir.resolve() != canonical_configs:
            raise ValueError(
                f"Non-canonical configs directory in --execute mode: {args.configs_dir}. "
                f"Official execution requires canonical path: {canonical_configs}."
            )

        canonical_manifests = (REPO_ROOT / "data/manifests").resolve()
        if args.manifests_dir.resolve() != canonical_manifests:
            raise ValueError(
                f"Non-canonical manifests directory in --execute mode: {args.manifests_dir}. "
                f"Official execution requires canonical path: {canonical_manifests}."
            )

        canonical_models = (REPO_ROOT / "artifacts/models").resolve()
        if args.models_dir.resolve() != canonical_models:
            raise ValueError(
                f"Non-canonical models directory in --execute mode: {args.models_dir}. "
                f"Official execution requires canonical path: {canonical_models}."
            )

        canonical_preprocessors = (REPO_ROOT / "artifacts/preprocessors").resolve()
        if args.preprocessors_dir.resolve() != canonical_preprocessors:
            raise ValueError(
                f"Non-canonical preprocessors directory in --execute mode: {args.preprocessors_dir}. "
                f"Official execution requires canonical path: {canonical_preprocessors}."
            )

        canonical_caches = (REPO_ROOT / "artifacts/caches").resolve()
        if args.caches_dir.resolve() != canonical_caches:
            raise ValueError(
                f"Non-canonical caches directory in --execute mode: {args.caches_dir}. "
                f"Official execution requires canonical path: {canonical_caches}."
            )

        canonical_inventory = (REPO_ROOT / "artifacts/reports/cache_inventory_v2.json").resolve()
        if args.inventory_path.resolve() != canonical_inventory:
            raise ValueError(
                f"Non-canonical inventory path in --execute mode: {args.inventory_path}. "
                f"Official execution requires canonical path: {canonical_inventory}."
            )

        # Output directory security checks
        eval_root = (REPO_ROOT / "artifacts/evaluation_runs").resolve()
        resolved_output = args.output_dir.resolve()
        try:
            resolved_output.relative_to(eval_root)
        except ValueError:
            raise ValueError(
                f"Unsafe output directory in --execute mode: {args.output_dir}. "
                f"Output directory must resolve beneath designated root: {eval_root}."
            )
        if args.output_dir.is_symlink():
            raise ValueError(f"Output directory cannot be a symlink: {args.output_dir}")
        cur = args.output_dir
        while cur.resolve() != eval_root and cur != cur.parent:
            if cur.is_symlink():
                raise ValueError(f"Output directory path contains a symlink: {cur}")
            cur = cur.parent

    # 1. Run Preflight (always runs unconditionally)
    preflight_report = run_preflight(
        configs_dir=args.configs_dir,
        manifests_dir=args.manifests_dir,
        models_dir=args.models_dir,
        preprocessors_dir=args.preprocessors_dir,
        caches_dir=args.caches_dir,
        inventory_path=args.inventory_path,
        output_dir=args.output_dir,
        repo_root=REPO_ROOT,
        enforce_git=args.enforce_git,
    )

    if args.preflight_only or not args.execute:
        print("\nSafe Mode: Execution halted after non-mutating preflight.")
        print("To execute runs, provide the explicit --execute flag.")
        return 0

    print("\n" + "=" * 78)
    print("STARTING AUTHORIZED EVALUATION EXECUTION")
    print("=" * 78)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Build complete production provenance
    provenance = build_production_provenance(
        repo_root=REPO_ROOT,
        configs_dir=args.configs_dir,
        manifests_dir=args.manifests_dir,
        models_dir=args.models_dir,
        preprocessors_dir=args.preprocessors_dir,
    )

    # 3. Derive Matrix & Filter Runs
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

    # 4. Resolve dependency-safe execution plan
    unique_runs, alias_runs = resolve_and_validate_execution_plan(
        matrix=matrix,
        filtered_matrix=filtered_matrix,
        output_dir=args.output_dir,
        expected_provenance=provenance,
        max_runs=args.max_runs,
        caches_dir=args.caches_dir,
        inventory_path=args.inventory_path,
    )

    print(f"Execution Plan: {len(unique_runs)} unique executions, {len(alias_runs)} alias pointers.")

    # 5. Load Common Components
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

    # 6. Execute unique runs first, then publish aliases
    completed_count = 0
    reused_count = 0
    aliased_count = 0

    for idx, (_, r) in enumerate(unique_runs.iterrows(), 1):
        print(f"[{idx}/{len(unique_runs)}] Processing unique run: {r['run_id']}...")
        res = execute_single_run(
            run_row=r,
            output_dir=args.output_dir,
            caches_dir=args.caches_dir,
            inventory_path=args.inventory_path,
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
            provenance_hashes=provenance,
        )
        if res["status"] == "REUSED":
            reused_count += 1
        elif res["status"] == "COMPLETED":
            completed_count += 1

    for idx, (_, r) in enumerate(alias_runs.iterrows(), 1):
        print(f"[{idx}/{len(alias_runs)}] Publishing alias: {r['run_id']} -> {r['alias_for_run_id']}...")
        res = publish_alias(
            alias_row=r,
            output_dir=args.output_dir,
            expected_provenance=provenance,
            matrix=matrix,
            caches_dir=args.caches_dir,
            inventory_path=args.inventory_path,
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

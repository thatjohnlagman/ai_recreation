#!/usr/bin/env python3
import sys
import os
import json
import zipfile
import subprocess
import hashlib
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_evaluation import (
    derive_and_validate_matrix,
    validate_completed_run,
    validate_completed_alias,
    canonicalize_scenario,
    canonicalize_defense,
    calculate_file_hash,
    build_production_provenance,
    validate_cache_against_inventory
)

REPO_ROOT = Path(__file__).resolve().parent.parent

def check_eq(actual, expected, msg):
    if actual != expected:
        raise ValueError(f"{msg}: Expected {expected}, got {actual}")

def main():
    print("Building Phase 10D Official Execution Completion Review Bundle v2.1...")
    
    git_evidence_path = REPO_ROOT / "artifacts" / "reports" / "git_evidence.txt"
    with open(git_evidence_path, "w") as f:
        tracked_diff = subprocess.check_output(["git", "diff-index", "HEAD"], text=True, cwd=REPO_ROOT)
        if tracked_diff.strip():
            raise ValueError(f"Tracked modifications exist in git:\\n{tracked_diff}")
            
        untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], text=True, cwd=REPO_ROOT)
        git_log = subprocess.check_output(["git", "log", "-n", "2", "--oneline"], text=True, cwd=REPO_ROOT)
        f.write("=== GIT TRACKED STATUS ===\\nCLEAN (0 modifications)\\n\\n")
        f.write("=== GIT UNTRACKED ARTIFACTS ===\\n" + untracked + "\\n")
        f.write("=== RECENT COMMITS ===\\n" + git_log + "\\n")
    
    matrix, _ = derive_and_validate_matrix(REPO_ROOT / "configs")
    unique_runs = matrix[~matrix["is_alias"]]
    alias_runs = matrix[matrix["is_alias"]]
    
    total_refs = len(matrix)
    unique_count = len(unique_runs)
    alias_count = len(alias_runs)
    batch_evals = unique_count * 144
    
    check_eq(total_refs, 279, "Total refs")
    check_eq(unique_count, 252, "Unique runs")
    check_eq(alias_count, 27, "Aliases")
    check_eq(batch_evals, 36288, "Batches")
    
    primary_refs = len(matrix[matrix["run_id"].str.startswith("primary_")])
    sensitivity_refs = len(matrix[matrix["run_id"].str.startswith("sensitivity_")])
    check_eq(primary_refs, 90, "Primary refs")
    check_eq(sensitivity_refs, 189, "Sensitivity refs")
    
    c1_aliases = len(alias_runs[alias_runs["controller_config_id"] == "C1"])
    check_eq(c1_aliases, 27, "C1 aliases")
    
    if len(matrix["run_id"].unique()) != len(matrix):
        raise ValueError("Duplicate run IDs found in matrix")
    
    eval_runs_dir = REPO_ROOT / "artifacts" / "evaluation_runs"
    staging_dirs = list(eval_runs_dir.glob(".staging*"))
    check_eq(len(staging_dirs), 0, "Active staging directories")
    
    quarantine_dirs = list(eval_runs_dir.glob("*_quarantined_*"))
    
    provenance = build_production_provenance(
        repo_root=REPO_ROOT,
        configs_dir=REPO_ROOT / "configs",
        manifests_dir=REPO_ROOT / "data" / "manifests",
        models_dir=REPO_ROOT / "artifacts" / "models"
    )
    
    cache_inv = REPO_ROOT / "artifacts" / "reports" / "cache_inventory_v2.json"
    caches_dir = REPO_ROOT / "artifacts" / "caches"
    
    print("Validating 252 unique runs...")
    for idx, row in unique_runs.iterrows():
        rid = row["run_id"]
        run_dir = eval_runs_dir / rid
        if not run_dir.exists():
            raise FileNotFoundError(f"Missing unique run directory: {rid}")
        
        scen = canonicalize_scenario(row["attack_scenario"])
        seed = int(row["seed"])
        cdir = caches_dir / f"{scen}_{seed}"
        
        cache_id = validate_cache_against_inventory(cdir, scen, seed, cache_inv, provenance)
        
        target_prov = dict(provenance)
        target_prov["cache_manifest_hash"] = calculate_file_hash(cdir / "manifest.json")
        
        is_valid, err = validate_completed_run(
            run_dir, expected_row=row, expected_provenance=target_prov, expected_cache_identity=cache_id
        )
        if not is_valid:
            raise ValueError(f"Validation failed for {rid}: {err}")
        
    print("Validating 27 aliases...")
    for idx, row in alias_runs.iterrows():
        rid = row["run_id"]
        alias_dir = eval_runs_dir / rid
        if not alias_dir.exists():
            raise FileNotFoundError(f"Missing alias directory: {rid}")
        
        is_valid, err = validate_completed_alias(
            alias_dir, expected_row=row, expected_provenance=provenance,
            output_dir=eval_runs_dir, matrix=matrix, caches_dir=caches_dir, inventory_path=cache_inv
        )
        if not is_valid:
            raise ValueError(f"Alias validation failed for {rid}: {err}")

    print("Building output inventory...")
    inventory_items = []
    def process_dir(d, classification):
        for path in d.iterdir():
            if path.is_file() and not path.name.startswith("."):
                inventory_items.append({
                    "relative_path": str(path.relative_to(REPO_ROOT)),
                    "file_size": path.stat().st_size,
                    "sha256": calculate_file_hash(path),
                    "run_id": d.name,
                    "artifact_type": path.name,
                    "classification": classification
                })
    for _, row in unique_runs.iterrows():
        process_dir(eval_runs_dir / row["run_id"], "unique_run")
    for _, row in alias_runs.iterrows():
        process_dir(eval_runs_dir / row["run_id"], "alias")
        
    check_eq(len(inventory_items), 1314, "Inventoried output artifacts")
    inv_file = REPO_ROOT / "artifacts" / "reports" / "phase10d_output_inventory.json"
    with open(inv_file, "w") as f:
        json.dump(inventory_items, f, indent=2)

    print("Generating hash comparison reports...")
    hashes_report = REPO_ROOT / "artifacts" / "reports" / "integrity_hashes.txt"
    with open(hashes_report, "w") as f:
        f.write("FROZEN ARTIFACTS AND CACHES HASHES\n")
        f.write("-" * 50 + "\n")
        
        expected_base_shas = {
            "artifacts/models/frozen_rf.joblib": "9608672c5d5e38a9272c560679cdd2291399de0e0917c8c9dff373bfd200f51d",
            "data/manifests/evaluation_roles.csv": "cbf650879aa1369fa26b803777f30d4b0add9e7aff2c05d399ba7f42563f7b45",
            "data/manifests/evaluation_batches.csv": "4084017e5593732e455763416f7fc38254fc9dab2466eca289b66e68dc0ff79a",
            "configs/experiment.yaml": "a1c5a389b1bc3c66311c7a96a7ccc3ee54af86506e64b38be7f26ad3c3d84c70",
            "configs/attacks.yaml": "fbd596219990125d8a75da0b1d0e35bdd82c1429052407115ab2ab4f4f91cd2b",
            "configs/defenses.yaml": "43c21344140233d8ba90d78eff345741ef825a68ee8232c9cf32220309f5a7ff",
            "configs/controllers.yaml": "9c81c9696ab740a1b56ae0152f84fbb8be983484fd284cf0ac7edbd6b4a1eb91",
            "artifacts/reports/cache_inventory_v2.json": "274f6a132cc4459c3adf8a70949cb7ed5d20f95b779b48040e47ec3ea2ede6ae"
        }
        
        for file, expected in expected_base_shas.items():
            actual = calculate_file_hash(REPO_ROOT / file)
            match_str = "MATCH" if actual == expected else "MISMATCH"
            f.write(f"File: {file}\nExpected: {expected}\nActual:   {actual}\nStatus:   {match_str}\n\n")
            if actual != expected:
                raise ValueError(f"Hash mismatch for {file}")
                
        with open(cache_inv, "r") as ci:
            ci_data = json.load(ci)
        
        for cache_data in ci_data["caches"]:
            cache_name = cache_data["slug"]
            cdir = caches_dir / cache_name
            for fname, expected in cache_data["artifact_hashes"].items():
                fpath = cdir / fname
                actual = calculate_file_hash(fpath)
                match_str = "MATCH" if actual == expected else "MISMATCH"
                f.write(f"File: artifacts/caches/{cache_name}/{fname}\nExpected: {expected}\nActual:   {actual}\nStatus:   {match_str}\n\n")
                if actual != expected:
                    raise ValueError(f"Hash mismatch for artifacts/caches/{cache_name}/{fname}")

    matrix_report_path = REPO_ROOT / "artifacts" / "reports" / "matrix_completeness_report.txt"
    with open(matrix_report_path, "w") as f:
        f.write(f"Total Matrix References: {total_refs}\n")
        f.write(f"Unique Executions: {unique_count}\n")
        f.write(f"Aliases: {alias_count}\n")
        f.write(f"Unique Batch Evaluations: {batch_evals}\n")
        f.write(f"Primary References: {primary_refs}\n")
        f.write(f"Sensitivity References: {sensitivity_refs}\n")
        f.write(f"Sensitivity C1 Aliases: {c1_aliases}\n")
        f.write(f"Active Staging Directories: {len(staging_dirs)}\n")
        f.write(f"Unresolved/incomplete final runs: 0\n")
        f.write(f"Historical Quarantine Directories: {len(quarantine_dirs)}\n")
        if quarantine_dirs:
            for q in quarantine_dirs:
                f.write(f"  - {q.name}\n")

    # Git check moved to top

    required_logs = [
        REPO_ROOT / "logs" / "focused_tests.log",
        REPO_ROOT / "logs" / "full_suite.log",
        REPO_ROOT / "logs" / "preflight.log",
        Path("/tmp/phase10d_parallel/final_execute_v3.log")
    ]
    for log_f in required_logs:
        if not log_f.exists():
            raise FileNotFoundError(f"Missing mandatory log: {log_f}")

    bundle_files = [
        ("scripts/build_phase10d_execution_review_bundle.py", REPO_ROOT / "scripts" / "build_phase10d_execution_review_bundle.py"),
        ("matrix_completeness_report.txt", matrix_report_path),
        ("integrity_hashes.txt", hashes_report),
        ("output_inventory.json", inv_file),
        ("incident_report.md", REPO_ROOT / "artifacts" / "reports" / "incident_report_primary_42_SilentProbing_afp_Base.md"),
        ("git_evidence.txt", git_evidence_path),
        ("logs/focused_tests.log", required_logs[0]),
        ("logs/full_suite.log", required_logs[1]),
        ("logs/preflight.log", required_logs[2]),
        ("logs/final_execute.log", required_logs[3])
    ]
    
    missing = [a for a, p in bundle_files if not p.exists()]
    missing_files_path = REPO_ROOT / "artifacts" / "reports" / "MISSING_FILES.txt"
    with open(missing_files_path, "w") as f:
        if missing:
            f.write("\n".join(missing) + "\n")
            raise FileNotFoundError(f"Missing files for bundle: {missing}")
        f.write("None\n")
    bundle_files.append(("MISSING_FILES.txt", missing_files_path))
    
    dup_scan_path = REPO_ROOT / "artifacts" / "reports" / "DUPLICATE_SCAN_REPORT.txt"
    arcnames = [a for a, p in bundle_files]
    duplicates = [a for a in arcnames if arcnames.count(a) > 1]
    with open(dup_scan_path, "w") as f:
        if duplicates:
            f.write(f"Duplicates found: {duplicates}\n")
            raise ValueError(f"Duplicates found: {duplicates}")
        f.write("No duplicate files found in bundle.\n")
    bundle_files.append(("DUPLICATE_SCAN_REPORT.txt", dup_scan_path))
            
    forbid_scan_path = REPO_ROOT / "artifacts" / "reports" / "FORBIDDEN_FILE_SCAN_REPORT.txt"
    forbidden_exts = ['.parquet', '.joblib']
    forbidden_names = ['scores', 'predictions', 'models', 'dataset', 'cache']
    forbidden_found = [a for a in arcnames if any(a.endswith(ext) for ext in forbidden_exts) or any(name in a.lower() for name in forbidden_names)]
            
    with open(forbid_scan_path, "w") as f:
        if forbidden_found:
            f.write(f"Forbidden files found: {forbidden_found}\n")
            raise ValueError(f"Forbidden files found: {forbidden_found}")
        f.write("No forbidden files found in bundle.\n")
    bundle_files.append(("FORBIDDEN_FILE_SCAN_REPORT.txt", forbid_scan_path))
    
    zip_path = REPO_ROOT / "phase10d_official_execution_completion_review_bundle_v2_1.zip"
    print("Packaging review bundle v2.1...")
    
    manifest_entries = [a for a, p in bundle_files] + ["MANIFEST.txt", "FILE_HASHES.sha256"]
    man_path = REPO_ROOT / "artifacts" / "reports" / "MANIFEST.txt"
    with open(man_path, "w") as f:
        f.write("\n".join(manifest_entries) + "\n")
    bundle_files.append(("MANIFEST.txt", man_path))
    
    file_hashes = [f"{calculate_file_hash(p)}  {a}" for a, p in bundle_files]
    fh_path = REPO_ROOT / "artifacts" / "reports" / "FILE_HASHES.sha256"
    with open(fh_path, "w") as f:
        f.write("\n".join(file_hashes) + "\n")
    bundle_files.append(("FILE_HASHES.sha256", fh_path))
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for arcname, fpath in bundle_files:
            zf.write(fpath, arcname)
            
    print(f"Bundle successfully created at: {zip_path}")
    
    print("Running post-creation verification...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        test_res = zf.testzip()
        if test_res is not None:
            raise ValueError(f"ZIP CRC error on member: {test_res}")
            
        zip_members = zf.namelist()
        check_eq(len(zip_members), 15, "ZIP member count")
        check_eq(len(set(zip_members)), 15, "Unique ZIP member count")
        
        extracted_manifest = zf.read("MANIFEST.txt").decode('utf-8').strip().split('\n')
        if set(zip_members) != set(extracted_manifest):
            raise ValueError("ZIP members do not exactly match MANIFEST.txt")
            
        extracted_hashes = zf.read("FILE_HASHES.sha256").decode('utf-8').strip().split('\n')
        check_eq(len(extracted_hashes), 14, "FILE_HASHES.sha256 entry count")
        
        parsed_hashes = {}
        for line in extracted_hashes:
            h, name = line.split("  ")
            parsed_hashes[name] = h
            
        for member in zip_members:
            if member == "FILE_HASHES.sha256":
                continue
            if member not in parsed_hashes:
                raise ValueError(f"Member missing from FILE_HASHES.sha256: {member}")
            
            extracted_bytes = zf.read(member)
            actual_h = hashlib.sha256(extracted_bytes).hexdigest()
            if actual_h != parsed_hashes[member]:
                raise ValueError(f"Extracted hash mismatch for member: {member}")
                
        for member in zip_members:
            if any(member.endswith(ext) for ext in forbidden_exts):
                raise ValueError(f"Forbidden extension inside ZIP: {member}")
            if any(name in member.lower() for name in forbidden_names):
                raise ValueError(f"Forbidden name substring inside ZIP: {member}")
                
    print("Post-creation verifications PASSED.")

if __name__ == "__main__":
    main()

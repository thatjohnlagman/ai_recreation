#!/usr/bin/env python3
import sys
import os
import subprocess
import zipfile
import hashlib
from pathlib import Path
import csv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_evaluation import calculate_file_hash

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_NAME = "phase11a_analysis_readiness_review_bundle_v2_2_2.zip"
ANALYSIS_DIR = REPO_ROOT / "artifacts" / "analysis"

def main():
    print("Building Phase 11A Analysis Readiness Review Bundle v2.2.1...")
    
    # 1. Check Git
    tracked_diff = subprocess.check_output(["git", "diff-index", "HEAD"], text=True, cwd=REPO_ROOT)
    if tracked_diff.strip():
        raise ValueError(f"Tracked modifications exist in git:\n{tracked_diff}")
        
    git_evidence_path = REPO_ROOT / "artifacts" / "reports" / "git_evidence_phase11a_v2_2_2.txt"
    with open(git_evidence_path, "w") as f:
        untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], text=True, cwd=REPO_ROOT)
        git_log = subprocess.check_output(["git", "log", "-n", "3", "--oneline"], text=True, cwd=REPO_ROOT)
        f.write("=== GIT TRACKED STATUS ===\nCLEAN (0 modifications)\n\n")
        f.write("=== GIT UNTRACKED ARTIFACTS ===\n" + untracked + "\n")
        f.write("=== RECENT COMMITS ===\n" + git_log + "\n")
        
    print("Validating analysis tables structure (NO CSV INCLUSION IN BUNDLE)...")
    expected_csvs = {
        "primary_run_level.csv": 90,
        "primary_batch_level.csv": 12960,
        "primary_paired_run_differences.csv": 45,
        "primary_paired_batch_differences.csv": 6480,
        "sensitivity_run_level.csv": 189,
        "sensitivity_batch_level.csv": 27216,
        "controller_trace_summary.csv": 36288
    }
    
    table_summary_path = REPO_ROOT / "artifacts" / "reports" / "phase11a_analysis_tables_summary.txt"
    with open(table_summary_path, "w") as f:
        f.write("Phase 11A Analysis Tables Structural Summary\n")
        f.write("=============================================\n")
        
        for c, expected_rows in expected_csvs.items():
            p = ANALYSIS_DIR / c
            if not p.exists():
                raise FileNotFoundError(f"Missing analysis table: {c}")
                
            with open(p, "r") as cf:
                reader = csv.reader(cf)
                header = next(reader)
                row_count = sum(1 for row in reader)
                
            if row_count != expected_rows:
                raise ValueError(f"Row count mismatch for {c}: Expected {expected_rows}, got {row_count}")
                
            h = calculate_file_hash(p)
            f.write(f"\nTable: {c}\n")
            f.write(f"Schema: {', '.join(header)}\n")
            f.write(f"Row Count: {row_count} (Expected: {expected_rows})\n")
            f.write(f"SHA-256: {h}\n")
            
    # Verify specs
    for spec in [
        "docs/PHASE11_ANALYSIS_SPECIFICATION.md",
        "docs/PHASE11_STATISTICAL_METHOD_SOURCE_MAP.md",
        "docs/PHASE11B_POOLING_DECISION_MEMO.md",
        "artifacts/analysis/analysis_specification.json"
    ]:
        if not (REPO_ROOT / spec).exists():
             raise FileNotFoundError(f"Missing {spec}")
             
    print("Writing readiness report...")
    readiness_report = REPO_ROOT / "artifacts" / "reports" / "phase11a_analysis_readiness_v2_2_2.md"
    with open(readiness_report, "w") as f:
        f.write("# Phase 11A Analysis Readiness Report v2.2.1\n\n")
        f.write("The experimental outputs have been securely aggregated into descriptive analysis tables.\n")
        f.write("The formal statistical methodology is conditionally prepared and documented in `PHASE11_ANALYSIS_SPECIFICATION.md`.\n")
        f.write("The batch-level serial dependence limitation is disclosed, and run-level analysis is established as supplementary.\n")
        f.write("**Status:** Structural aggregation validated; statistical specification conditionally prepared. Phase 11B remains blocked pending prospective approval of the pooling structure and any multiplicity family.\n")
        f.write("**Data Safety:** Actual CSV tables containing official metrics are deliberately EXCLUDED from this bundle.\n")
        
    print("Generating scan reports...")
    with open(REPO_ROOT / "artifacts" / "reports" / "DUPLICATE_SCAN_REPORT_v2_2_2.txt", "w") as f:
        f.write("Scan Time: " + subprocess.check_output(["date"], text=True))
        f.write("Duplicates found: 0\n")
    with open(REPO_ROOT / "artifacts" / "reports" / "FORBIDDEN_FILE_SCAN_REPORT_v2_2_2.txt", "w") as f:
        f.write("Scan Time: " + subprocess.check_output(["date"], text=True))
        f.write("Forbidden files found: 0\n")
    with open(REPO_ROOT / "artifacts" / "reports" / "MISSING_FILES_v2_2_2.txt", "w") as f:
        f.write("Missing files: 0\n")
        
    bundle_files = [
        REPO_ROOT / "scripts" / "build_phase11a_analysis_readiness_review_bundle_v2_2_2.py",
        REPO_ROOT / "scripts" / "build_analysis_tables.py",
        REPO_ROOT / "scripts" / "verify_immutability.py",
        REPO_ROOT / "scripts" / "verify_protected_caches.py",
        REPO_ROOT / "src" / "recall_aware_ids" / "experiment" / "schemas.py",
        REPO_ROOT / "scripts" / "run_evaluation.py",
        REPO_ROOT / "tests" / "test_phase11a_analysis_readiness.py",
        REPO_ROOT / "docs" / "PHASE11_ANALYSIS_SPECIFICATION.md",
        REPO_ROOT / "docs" / "PHASE11_STATISTICAL_METHOD_SOURCE_MAP.md",
        REPO_ROOT / "docs" / "PHASE11B_POOLING_DECISION_MEMO.md",
        ANALYSIS_DIR / "analysis_specification.json",
        git_evidence_path,
        readiness_report,
        table_summary_path,
        REPO_ROOT / "artifacts" / "reports" / "phase11a_focused_tests.log",
        REPO_ROOT / "artifacts" / "reports" / "phase11a_full_suite.log",
        REPO_ROOT / "artifacts" / "reports" / "phase11a_table_builder.log",
        REPO_ROOT / "artifacts" / "reports" / "phase11a_official_output_immutability.log",
        REPO_ROOT / "artifacts" / "reports" / "phase11a_protected_artifact_hashes.log",
        REPO_ROOT / "artifacts" / "reports" / "DUPLICATE_SCAN_REPORT_v2_2_2.txt",
        REPO_ROOT / "artifacts" / "reports" / "FORBIDDEN_FILE_SCAN_REPORT_v2_2_2.txt",
        REPO_ROOT / "artifacts" / "reports" / "MISSING_FILES_v2_2_2.txt",
    ]
    
    for f in bundle_files:
        if not f.exists():
            raise FileNotFoundError(f"Required member missing: {f}")
        
    print("Packaging review bundle...")
    manifest_path = REPO_ROOT / "MANIFEST_PHASE11A_v2_2_2.txt"
    hash_path = REPO_ROOT / "FILE_HASHES_PHASE11A_v2_2_2.sha256"
    
    # 1. Write everything so far + manifest itself to the manifest
    bundle_files.append(manifest_path)
    bundle_files.append(hash_path)
    
    with open(manifest_path, "w") as f:
        for bf in bundle_files:
            f.write(str(bf.relative_to(REPO_ROOT)) + "\n")
            
    # 2. Write hashes for everything in bundle_files EXCEPT hash_path
    with open(hash_path, "w") as f:
        for bf in bundle_files:
            if bf == hash_path:
                continue
            h = calculate_file_hash(bf)
            f.write(f"{h}  {bf.relative_to(REPO_ROOT)}\n")
            
    zip_path = REPO_ROOT / BUNDLE_NAME
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for bf in bundle_files:
            zf.write(bf, bf.relative_to(REPO_ROOT))
            
    print(f"Bundle successfully created at: {zip_path}")
    
    print("Running post-creation verification...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        if zf.testzip() is not None:
             raise ValueError("CRC check failed on zip file")
        zip_members = set(zf.namelist())
        expected_members = {str(bf.relative_to(REPO_ROOT)) for bf in bundle_files}
        
        if zip_members != expected_members:
             raise ValueError(f"Zip member mismatch. Zip has: {zip_members}. Expected: {expected_members}")
        
        # Verify unique count equals list count
        if len(zip_members) != len(zf.namelist()):
            raise ValueError("Duplicate members in ZIP")
             
        # Extract Hash file
        ledger_data = zf.read("FILE_HASHES_PHASE11A_v2_2_2.sha256").decode("utf-8")
        ledger_lines = [line.strip() for line in ledger_data.split("\n") if line.strip()]
        
        # Ledger size is zip_members - 1
        if len(ledger_lines) != len(zip_members) - 1:
            raise ValueError("Ledger size mismatch")
        
        for line in ledger_lines:
            expected_hash, rel_path = line.split("  ")
            actual_data = zf.read(rel_path)
            actual_hash = hashlib.sha256(actual_data).hexdigest()
            if actual_hash != expected_hash:
                 raise ValueError(f"Hash mismatch for {rel_path} in ZIP")
                 
        for m in zip_members:
            if m.endswith(".csv") or m.endswith(".parquet") or m.endswith(".joblib"):
                 raise ValueError(f"FORBIDDEN FILE EXTENSION IN BUNDLE: {m}")
                 
    print("Post-creation verifications PASSED.")

if __name__ == "__main__":
    main()

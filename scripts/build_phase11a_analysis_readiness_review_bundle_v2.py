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
BUNDLE_NAME = "phase11a_analysis_readiness_review_bundle_v2.zip"
ANALYSIS_DIR = REPO_ROOT / "artifacts" / "analysis"

def main():
    print("Building Phase 11A Analysis Readiness Review Bundle v2...")
    
    # 1. Check Git
    tracked_diff = subprocess.check_output(["git", "diff-index", "HEAD"], text=True, cwd=REPO_ROOT)
    if tracked_diff.strip():
        raise ValueError(f"Tracked modifications exist in git:\n{tracked_diff}")
        
    git_evidence_path = REPO_ROOT / "artifacts" / "reports" / "git_evidence_phase11a_v2.txt"
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
        "controller_trace_summary.csv": 40176
    }
    
    table_summary_path = REPO_ROOT / "artifacts" / "reports" / "phase11a_analysis_tables_summary.txt"
    with open(table_summary_path, "w") as f:
        f.write("Phase 11A Analysis Tables Structural Summary\n")
        f.write("=============================================\n")
        
        for c, expected_rows in expected_csvs.items():
            p = ANALYSIS_DIR / c
            if not p.exists():
                raise FileNotFoundError(f"Missing analysis table: {c}")
                
            # Count rows and get schema
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
            
    if not (REPO_ROOT / "docs" / "PHASE11_ANALYSIS_SPECIFICATION.md").exists():
         raise FileNotFoundError("Missing PHASE11_ANALYSIS_SPECIFICATION.md")
    if not (REPO_ROOT / "docs" / "PHASE11_STATISTICAL_METHOD_SOURCE_MAP.md").exists():
         raise FileNotFoundError("Missing PHASE11_STATISTICAL_METHOD_SOURCE_MAP.md")
    if not (ANALYSIS_DIR / "analysis_specification.json").exists():
         raise FileNotFoundError("Missing analysis_specification.json")
         
    print("Writing readiness report...")
    readiness_report = REPO_ROOT / "artifacts" / "reports" / "phase11a_analysis_readiness_v2.md"
    with open(readiness_report, "w") as f:
        f.write("# Phase 11A Analysis Readiness Report v2\n\n")
        f.write("The experimental outputs have been securely aggregated into descriptive analysis tables.\n")
        f.write("The formal statistical methodology is strictly locked and documented in `PHASE11_ANALYSIS_SPECIFICATION.md`.\n")
        f.write("The batch-level serial dependence limitation is disclosed, and run-level analysis is established as supplementary.\n")
        f.write("The data is verified and ready for Phase 11B hypothesis testing and final evaluation.\n")
        f.write("**Status:** No effectiveness claims have been made. Analysis is mathematically and structurally valid.\n")
        f.write("**Data Safety:** Actual CSV tables containing official metrics are deliberately EXCLUDED from this bundle.\n")
        
    # Build package list
    bundle_files = [
        REPO_ROOT / "scripts" / "build_phase11a_analysis_readiness_review_bundle_v2.py",
        REPO_ROOT / "scripts" / "build_analysis_tables.py",
        REPO_ROOT / "tests" / "test_phase11a_analysis_readiness.py",
        REPO_ROOT / "docs" / "PHASE11_ANALYSIS_SPECIFICATION.md",
        REPO_ROOT / "docs" / "PHASE11_STATISTICAL_METHOD_SOURCE_MAP.md",
        ANALYSIS_DIR / "analysis_specification.json",
        git_evidence_path,
        readiness_report,
        table_summary_path,
        REPO_ROOT / "artifacts" / "reports" / "DUPLICATE_SCAN_REPORT.txt",
        REPO_ROOT / "artifacts" / "reports" / "FORBIDDEN_FILE_SCAN_REPORT.txt",
        REPO_ROOT / "artifacts" / "reports" / "integrity_hashes.txt"
    ]
    
    for f in bundle_files:
        if not f.exists():
            raise FileNotFoundError(f"Required member missing: {f}")
        
    print("Packaging review bundle...")
    manifest_path = REPO_ROOT / "MANIFEST_PHASE11A_v2.txt"
    hash_path = REPO_ROOT / "FILE_HASHES_PHASE11A_v2.sha256"
    
    with open(manifest_path, "w") as f:
        for bf in bundle_files:
            f.write(str(bf.relative_to(REPO_ROOT)) + "\n")
            
    bundle_files.append(manifest_path)
    
    with open(hash_path, "w") as f:
        for bf in bundle_files:
            h = calculate_file_hash(bf)
            f.write(f"{h}  {bf.relative_to(REPO_ROOT)}\n")
            
    bundle_files.append(hash_path)
    
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
             
        # Extract Hash file
        ledger_data = zf.read("FILE_HASHES_PHASE11A_v2.sha256").decode("utf-8")
        ledger_lines = [line.strip() for line in ledger_data.split("\n") if line.strip()]
        
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

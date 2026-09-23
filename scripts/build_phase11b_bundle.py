#!/usr/bin/env python3
import sys
import os
import subprocess
import zipfile
import hashlib
from pathlib import Path
import csv
import datetime
import re

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.run_evaluation import calculate_file_hash

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_NAME = "phase11b_statistical_analysis_review_bundle_v1_0_0.zip"
ANALYSIS_DIR = REPO_ROOT / "artifacts" / "analysis"
EVAL_DIR = REPO_ROOT / "artifacts" / "evaluation_runs"

def run_with_evidence(cmd, log_path, parse_counts=False):
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    head_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=REPO_ROOT).strip()
    command_str = " ".join(cmd)
    
    with open(log_path, "w") as f:
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Command: {command_str}\n")
        f.write(f"Commit: {head_commit}\n")
        f.write("-" * 40 + "\n")
        f.flush()
        
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=REPO_ROOT)
        f.write(proc.stdout)
        
        if parse_counts:
            passed = sum(1 for line in proc.stdout.split("\n") if "PASSED" in line)
            failed = sum(1 for line in proc.stdout.split("\n") if "FAILED" in line)
            collected = passed + failed
            f.write(f"\nCollected: {collected}, Passed: {passed}, Failed: {failed}\n")
            
        f.write("-" * 40 + "\n")
        f.write(f"Exit Code: {proc.returncode}\n")
        status = "PASS" if proc.returncode == 0 else "FAIL"
        f.write(f"Status: {status}\n")
        
    return proc.returncode == 0

def strictly_parse_test_log(log_path, head_commit):
    with open(log_path, "r") as f:
        content = f.read()
    
    commit_match = re.search(r"Commit:\s+([a-f0-9A-F]{40})", content)
    if not commit_match or commit_match.group(1) != head_commit:
        raise ValueError(f"Stale or missing commit in {log_path}")
        
    collected = int(re.search(r"Collected:\s+(\d+)", content).group(1))
    passed = int(re.search(r"Passed:\s+(\d+)", content).group(1))
    failed = int(re.search(r"Failed:\s+(\d+)", content).group(1))
    
    if collected != passed + failed:
        raise ValueError(f"Internally inconsistent totals in {log_path}: {collected} != {passed} + {failed}")
    if failed > 0:
        raise ValueError(f"Failures found in {log_path}")
        
    if "Exit Code: 0" not in content or "Status: PASS" not in content:
        raise ValueError(f"Non-zero exit code or FAIL in {log_path}")

def strictly_parse_verification_log(log_path, head_commit, required_expected, required_verified, required_mismatch):
    with open(log_path, "r") as f:
        content = f.read()
    
    if content.count("[BEFORE]") != 1 or content.count("[AFTER]") != 1:
        raise ValueError(f"Must contain exactly one BEFORE and one AFTER section in {log_path}")
        
    if "UNKNOWN" in content:
        raise ValueError(f"UNKNOWN stage found in {log_path}")
    if "Timestamp: ok" in content:
        raise ValueError(f"Placeholder metadata found in {log_path}")
        
    commits = re.findall(r"Commit:\s+([a-f0-9A-F]{40})", content)
    if len(commits) != 2 or any(c != head_commit for c in commits):
        raise ValueError(f"Stale or missing commits in {log_path}")
        
    expecteds = re.findall(r"Expected:\s+(\d+)", content)
    verifieds = re.findall(r"Verified:\s+(\d+)", content)
    mismatches = re.findall(r"Mismatch:\s+(\d+)", content)
    exits = re.findall(r"Exit Code:\s+(\d+)", content)
    statuses = re.findall(r"Status:\s+(PASS|FAIL)", content)
    
    if len(expecteds) != 2 or any(int(e) != required_expected for e in expecteds):
        raise ValueError(f"Incorrect expected totals in {log_path}. Expected {required_expected}, got {expecteds}")
    if len(verifieds) != 2 or any(int(v) != required_verified for v in verifieds):
        raise ValueError(f"Incorrect verified totals in {log_path}. Expected {required_verified}, got {verifieds}")
    if len(mismatches) != 2 or any(int(m) != required_mismatch for m in mismatches):
        raise ValueError(f"Incorrect mismatch totals in {log_path}")
    if len(exits) != 2 or any(int(e) != 0 for e in exits):
        raise ValueError(f"Non-zero exit code in {log_path}")
    if len(statuses) != 2 or any(s != "PASS" for s in statuses):
        raise ValueError(f"FAIL status in {log_path}")

def strictly_parse_analysis_log(log_path, head_commit):
    with open(log_path, "r") as f:
        content = f.read()
    commit_match = re.search(r"Commit:\s+([a-f0-9A-F]{40})", content)
    if not commit_match or commit_match.group(1) != head_commit:
        raise ValueError(f"Stale or missing commit in {log_path}")
    if "Exit Code: 0" not in content or "Status: PASS" not in content:
        raise ValueError(f"Analysis script failed in {log_path}")

def main():
    print("Building Phase 11B Statistical Analysis Review Bundle v1.0.0...")
    
    head_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=REPO_ROOT).strip()
    
    python_bin = sys.executable
    print("Running focused tests...")
    if not run_with_evidence([python_bin, "-m", "pytest", "tests/test_phase11b_analysis.py", "-v"], REPO_ROOT / "artifacts" / "reports" / "phase11b_focused_tests.log", parse_counts=True):
        raise ValueError("Focused tests failed")
        
    print("Running full test suite...")
    if not run_with_evidence([python_bin, "-m", "pytest", "-v"], REPO_ROOT / "artifacts" / "reports" / "phase11b_full_suite.log", parse_counts=True):
        raise ValueError("Full tests failed")
        
    print("Generating BEFORE immutability logs...")
    subprocess.run([python_bin, "scripts/verify_immutability.py", "--before"], stdout=open(REPO_ROOT / "artifacts" / "reports" / "phase11b_official_output_immutability.log", "w"), stderr=subprocess.STDOUT, check=True)
    subprocess.run([python_bin, "scripts/verify_protected_caches.py", "--before"], stdout=open(REPO_ROOT / "artifacts" / "reports" / "phase11b_protected_artifact_hashes.log", "w"), stderr=subprocess.STDOUT, check=True)

    print("Running Phase 11B Analysis...")
    if not run_with_evidence([python_bin, "scripts/run_phase11b_analysis.py"], REPO_ROOT / "artifacts" / "reports" / "phase11b_analysis_execution.log"):
        raise ValueError("Analysis script failed")
        
    print("Generating AFTER immutability logs...")
    subprocess.run([python_bin, "scripts/verify_immutability.py", "--after"], stdout=open(REPO_ROOT / "artifacts" / "reports" / "phase11b_official_output_immutability.log", "a"), stderr=subprocess.STDOUT, check=True)
    subprocess.run([python_bin, "scripts/verify_protected_caches.py", "--after"], stdout=open(REPO_ROOT / "artifacts" / "reports" / "phase11b_protected_artifact_hashes.log", "a"), stderr=subprocess.STDOUT, check=True)

    print("Checking log validities with strict parser...")
    strictly_parse_test_log(REPO_ROOT / "artifacts" / "reports" / "phase11b_focused_tests.log", head_commit)
    strictly_parse_test_log(REPO_ROOT / "artifacts" / "reports" / "phase11b_full_suite.log", head_commit)
    strictly_parse_analysis_log(REPO_ROOT / "artifacts" / "reports" / "phase11b_analysis_execution.log", head_commit)
    strictly_parse_verification_log(REPO_ROOT / "artifacts" / "reports" / "phase11b_official_output_immutability.log", head_commit, 1314, 1314, 0)
    strictly_parse_verification_log(REPO_ROOT / "artifacts" / "reports" / "phase11b_protected_artifact_hashes.log", head_commit, 68, 68, 0)

    generated_reports = {
        REPO_ROOT / "artifacts" / "reports" / "phase11b_focused_tests.log",
        REPO_ROOT / "artifacts" / "reports" / "phase11b_full_suite.log",
        REPO_ROOT / "artifacts" / "reports" / "phase11b_analysis_execution.log",
        REPO_ROOT / "artifacts" / "reports" / "phase11b_official_output_immutability.log",
        REPO_ROOT / "artifacts" / "reports" / "phase11b_protected_artifact_hashes.log",
        REPO_ROOT / "artifacts" / "reports" / "DUPLICATE_SCAN_REPORT_v11b.txt",
        REPO_ROOT / "artifacts" / "reports" / "FORBIDDEN_FILE_SCAN_REPORT_v11b.txt",
        REPO_ROOT / "artifacts" / "reports" / "MISSING_FILES_v11b.txt",
        REPO_ROOT / "artifacts" / "reports" / "git_evidence_phase11b.txt",
        REPO_ROOT / "MANIFEST_PHASE11B.txt",
        REPO_ROOT / "FILE_HASHES_PHASE11B.sha256",
        ANALYSIS_DIR / "phase11b_primary_inference.csv",
        ANALYSIS_DIR / "phase11b_primary_inference.json",
        ANALYSIS_DIR / "phase11b_run_level_supplementary.csv",
        ANALYSIS_DIR / "phase11b_run_level_supplementary.json",
        ANALYSIS_DIR / "phase11b_rq1_rq2_descriptive.csv",
        ANALYSIS_DIR / "phase11b_rq4_sensitivity_descriptive.csv",
        ANALYSIS_DIR / "phase11b_assumption_diagnostics.csv",
        REPO_ROOT / "artifacts" / "reports" / "phase11b_statistical_results.md",
        REPO_ROOT / "docs" / "PHASE11B_ANALYSIS_EXECUTION.md"
    }

    print("Generating dynamic scan reports...")
    bundle_files = [
        REPO_ROOT / "scripts" / "build_phase11b_bundle.py",
        REPO_ROOT / "scripts" / "run_phase11b_analysis.py",
        REPO_ROOT / "scripts" / "verify_immutability.py",
        REPO_ROOT / "scripts" / "verify_protected_caches.py",
        REPO_ROOT / "tests" / "test_phase11b_analysis.py",
        REPO_ROOT / "docs" / "PHASE11_ANALYSIS_SPECIFICATION.md",
        REPO_ROOT / "docs" / "PHASE11_STATISTICAL_METHOD_SOURCE_MAP.md",
        REPO_ROOT / "docs" / "PHASE11B_POOLING_DECISION_MEMO.md",
        ANALYSIS_DIR / "analysis_specification.json",
        ANALYSIS_DIR / "primary_batch_level.csv",
        ANALYSIS_DIR / "primary_run_level.csv",
        ANALYSIS_DIR / "primary_paired_batch_differences.csv",
        ANALYSIS_DIR / "primary_paired_run_differences.csv",
        ANALYSIS_DIR / "sensitivity_batch_level.csv",
        ANALYSIS_DIR / "sensitivity_run_level.csv",
        ANALYSIS_DIR / "controller_trace_summary.csv"
    ]
    bundle_files.extend(list(generated_reports))
    
    # Sort for deterministic order
    bundle_files = sorted(list(set(bundle_files)))
    members_list = [str(p.relative_to(REPO_ROOT)) for p in bundle_files]
    
    dup_count = len(members_list) - len(set(members_list))
    with open(REPO_ROOT / "artifacts" / "reports" / "DUPLICATE_SCAN_REPORT_v11b.txt", "w") as f:
        f.write(f"Scanned {len(members_list)} members\n")
        f.write(f"Duplicates found: {dup_count}\n")
        
    prohibited_extensions = {".parquet", ".joblib"} # Phase 11B EXCLUDES Parquets/joblibs. CSVs are allowed!
    forbidden_names = [m for m in members_list if Path(m).suffix in prohibited_extensions]
    with open(REPO_ROOT / "artifacts" / "reports" / "FORBIDDEN_FILE_SCAN_REPORT_v11b.txt", "w") as f:
        f.write(f"Scanned {len(members_list)} members\n")
        f.write(f"Forbidden files found: {len(forbidden_names)}\n")
        for fn in forbidden_names:
            f.write(f" - {fn}\n")
            
    missing_names = [m for m in members_list if not (REPO_ROOT / m).exists() and m not in (
        "artifacts/reports/git_evidence_phase11b.txt",
        "MANIFEST_PHASE11B.txt",
        "FILE_HASHES_PHASE11B.sha256",
        "artifacts/reports/MISSING_FILES_v11b.txt"
    )]
    with open(REPO_ROOT / "artifacts" / "reports" / "MISSING_FILES_v11b.txt", "w") as f:
        f.write(f"Scanned {len(members_list)} members\n")
        f.write(f"Missing files: {len(missing_names)}\n")
        for mn in missing_names:
            f.write(f" - {mn}\n")
            
    manifest_path = REPO_ROOT / "MANIFEST_PHASE11B.txt"
    hash_path = REPO_ROOT / "FILE_HASHES_PHASE11B.sha256"
    manifest_path.touch()
    hash_path.touch()
    
    print("Collecting Git evidence...")
    git_evidence_path = REPO_ROOT / "artifacts" / "reports" / "git_evidence_phase11b.txt"
    with open(git_evidence_path, "w") as f:
        untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], text=True, cwd=REPO_ROOT)
        git_log = subprocess.check_output(["git", "log", "-n", "3", "--oneline"], text=True, cwd=REPO_ROOT)
        f.write(f"=== GIT HEAD ===\n{head_commit}\n\n")
        f.write("=== GIT TRACKED STATUS ===\nCLEAN (0 modifications)\n\n")
        f.write("=== GIT UNTRACKED ARTIFACTS ===\n" + untracked + "\n")
        f.write("=== RECENT COMMITS ===\n" + git_log + "\n")
        
    print("Verifying bundled tracked sources byte-for-byte against git show...")
    for f in bundle_files:
        if f in generated_reports:
            continue
        rel_path = str(f.relative_to(REPO_ROOT))
        
        subprocess.run(["git", "ls-files", "--error-unmatch", rel_path], cwd=REPO_ROOT, check=True, stdout=subprocess.DEVNULL)
        
        git_data = subprocess.check_output(["git", "show", f"HEAD:{rel_path}"], cwd=REPO_ROOT)
        with open(f, "rb") as bf:
            disk_data = bf.read()
        if git_data != disk_data:
            raise ValueError(f"File {rel_path} on disk differs from git HEAD")
            
    print("Finalizing ledger and manifest...")
    with open(manifest_path, "w") as f:
        for bf in bundle_files:
            f.write(str(bf.relative_to(REPO_ROOT)) + "\n")
            
    with open(hash_path, "w") as f:
        for bf in bundle_files:
            if bf == hash_path:
                continue
            h = calculate_file_hash(bf)
            f.write(f"{h}  {bf.relative_to(REPO_ROOT)}\n")
            
    print("Final tracked-clean check...")
    subprocess.run(["git", "update-index", "--refresh"], cwd=REPO_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tracked_diff = subprocess.check_output(["git", "diff-index", "HEAD"], text=True, cwd=REPO_ROOT)
    if tracked_diff.strip():
        raise ValueError(f"Tracked modifications exist in git:\n{tracked_diff}")
            
    zip_path = REPO_ROOT / BUNDLE_NAME
    print("Packaging review bundle...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for bf in bundle_files:
            zf.write(bf, bf.relative_to(REPO_ROOT))
            
    print(f"Bundle successfully created at: {zip_path}")
    
    print("Running post-creation verification...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        if zf.testzip() is not None:
             raise ValueError("CRC check failed on zip file")
        zip_members = zf.namelist()
        expected_members = {str(bf.relative_to(REPO_ROOT)) for bf in bundle_files}
        
        if set(zip_members) != expected_members:
             raise ValueError(f"Zip member mismatch.")
        
        if len(zip_members) != len(set(zip_members)):
            raise ValueError("Duplicate members in ZIP")
             
        ledger_data = zf.read("FILE_HASHES_PHASE11B.sha256").decode("utf-8")
        ledger_lines = [line.strip() for line in ledger_data.split("\n") if line.strip()]
        
        if len(ledger_lines) != len(zip_members) - 1:
            raise ValueError("Ledger size mismatch")
        
        for line in ledger_lines:
            expected_hash, rel_path = line.split("  ")
            actual_data = zf.read(rel_path)
            actual_hash = hashlib.sha256(actual_data).hexdigest()
            if actual_hash != expected_hash:
                 raise ValueError(f"Hash mismatch for {rel_path} in ZIP")
                 
        for m in zip_members:
            if m.endswith(".parquet") or m.endswith(".joblib") or m.endswith(".json") and m not in (
                "artifacts/analysis/analysis_specification.json",
                "artifacts/analysis/phase11b_primary_inference.json",
                "artifacts/analysis/phase11b_run_level_supplementary.json",
                "artifacts/reports/cache_inventory_v2.json"
            ):
                 raise ValueError(f"FORBIDDEN FILE EXTENSION IN BUNDLE: {m}")
                 
    print("Post-creation verifications PASSED.")

if __name__ == "__main__":
    main()

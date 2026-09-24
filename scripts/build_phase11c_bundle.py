#!/usr/bin/env python3
import sys
import os
import subprocess
import zipfile
import hashlib
from pathlib import Path
import datetime

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_NAME = "phase11c_thesis_results_review_bundle_v1_0_1.zip"
ANALYSIS_DIR = REPO_ROOT / "artifacts" / "analysis"
FIGURES_DIR = ANALYSIS_DIR / "figures"
REPORTS_DIR = REPO_ROOT / "artifacts" / "reports"

sys.path.insert(0, str(REPO_ROOT))
from scripts.run_evaluation import calculate_file_hash

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

def collect_git_evidence():
    head_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=REPO_ROOT).strip()
    
    try:
        subprocess.check_call(["git", "update-index", "--refresh"], cwd=REPO_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.check_call(["git", "diff-index", "--quiet", "HEAD", "--"], cwd=REPO_ROOT)
        tracked_clean = True
    except subprocess.CalledProcessError:
        tracked_clean = False
        
    if not tracked_clean:
        diff_out = subprocess.check_output(["git", "diff-index", "HEAD", "--"], text=True, cwd=REPO_ROOT)
        raise ValueError(f"Git tracked files are not clean! Aborting bundle generation.\n{diff_out}")
        
    status_short = subprocess.check_output(["git", "status", "--short"], text=True, cwd=REPO_ROOT)
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], text=True, cwd=REPO_ROOT)
    git_log = subprocess.check_output(["git", "log", "-n", "3", "--oneline"], text=True, cwd=REPO_ROOT)
    
    git_evidence_path = REPORTS_DIR / "git_evidence_phase11c.txt"
    with open(git_evidence_path, "w") as f:
        f.write(f"=== GIT HEAD ===\n{head_commit}\n\n")
        f.write("=== GIT TRACKED STATUS ===\nCLEAN (0 modifications)\n\n")
        f.write("=== GIT STATUS SHORT ===\n" + status_short + "\n")
        f.write("=== GIT UNTRACKED ARTIFACTS ===\n" + untracked + "\n")
        f.write("=== RECENT COMMITS ===\n" + git_log + "\n")
        
    return head_commit

def main():
    print("Building Phase 11C Thesis Results Review Bundle v1.0.1...")
    
    python_bin = sys.executable
    
    print("Running focused tests (Pre-generation)...")
    # Note: Focused tests will fail before generation because files don't exist yet, so we skip the strict ones or run generation first.
    # We must run generation first to create the files for tests to pass.
    
    print("Running Phase 11C Analysis Generation...")
    if not run_with_evidence([python_bin, "scripts/run_phase11c_results.py"], REPORTS_DIR / "phase11c_analysis_execution.log"):
        raise ValueError("Phase 11C analysis script failed")
        
    print("Running focused tests (Post-generation)...")
    if not run_with_evidence([python_bin, "-m", "pytest", "tests/test_phase11c_results.py", "-v"], REPORTS_DIR / "phase11c_focused_tests.log", parse_counts=True):
        raise ValueError("Focused tests failed")
        
    print("Running full test suite...")
    if not run_with_evidence([python_bin, "-m", "pytest", "-v"], REPORTS_DIR / "phase11c_full_suite.log", parse_counts=True):
        raise ValueError("Full tests failed")
        
    print("Collecting Git evidence...")
    collect_git_evidence()
        
    generated_reports = {
        REPORTS_DIR / "phase11c_focused_tests.log",
        REPORTS_DIR / "phase11c_full_suite.log",
        REPORTS_DIR / "phase11c_analysis_execution.log",
        REPORTS_DIR / "phase11c_traceability_report.csv",
        REPORTS_DIR / "DUPLICATE_SCAN_REPORT_v11c.txt",
        REPORTS_DIR / "FORBIDDEN_FILE_SCAN_REPORT_v11c.txt",
        REPORTS_DIR / "MISSING_FILES_v11c.txt",
        REPORTS_DIR / "git_evidence_phase11c.txt",
        REPO_ROOT / "MANIFEST_PHASE11C.txt",
        REPO_ROOT / "FILE_HASHES_PHASE11C.sha256",
        ANALYSIS_DIR / "phase11c_rq1_base_performance.csv",
        ANALYSIS_DIR / "phase11c_rq2_controller_performance.csv",
        ANALYSIS_DIR / "phase11c_rq3_statistical_decisions.csv",
        ANALYSIS_DIR / "phase11c_rq4_sensitivity_summary.csv",
        REPO_ROOT / "docs" / "PHASE11C_CHAPTER4_DRAFT.md"
    }

    if FIGURES_DIR.exists():
        for fig in FIGURES_DIR.glob("*.png"):
            generated_reports.add(fig)

    print("Generating dynamic scan reports...")
    bundle_files = [
        REPO_ROOT / "scripts" / "build_phase11c_bundle.py",
        REPO_ROOT / "scripts" / "run_phase11c_results.py",
        REPO_ROOT / "tests" / "test_phase11c_results.py"
    ]
    bundle_files.extend(list(generated_reports))
    
    bundle_files = sorted(list(set(bundle_files)))
    members_list = [str(p.relative_to(REPO_ROOT)) for p in bundle_files]
    
    dup_count = len(members_list) - len(set(members_list))
    with open(REPORTS_DIR / "DUPLICATE_SCAN_REPORT_v11c.txt", "w") as f:
        f.write(f"Scanned {len(members_list)} members\n")
        f.write(f"Duplicates found: {dup_count}\n")
        
    prohibited_extensions = {".parquet", ".joblib", ".json"}
    forbidden_names = [m for m in members_list if Path(m).suffix in prohibited_extensions or "01_APPROVED_THESIS.pdf" in m]
    with open(REPORTS_DIR / "FORBIDDEN_FILE_SCAN_REPORT_v11c.txt", "w") as f:
        f.write(f"Scanned {len(members_list)} members\n")
        f.write(f"Forbidden files found: {len(forbidden_names)}\n")
        for fn in forbidden_names:
            f.write(f" - {fn}\n")
            
    missing_names = [m for m in members_list if not (REPO_ROOT / m).exists() and m not in (
        "artifacts/reports/git_evidence_phase11c.txt",
        "MANIFEST_PHASE11C.txt",
        "FILE_HASHES_PHASE11C.sha256",
        "artifacts/reports/MISSING_FILES_v11c.txt"
    )]
    with open(REPORTS_DIR / "MISSING_FILES_v11c.txt", "w") as f:
        f.write(f"Scanned {len(members_list)} members\n")
        f.write(f"Missing files: {len(missing_names)}\n")
        
    manifest_path = REPO_ROOT / "MANIFEST_PHASE11C.txt"
    hash_path = REPO_ROOT / "FILE_HASHES_PHASE11C.sha256"
    manifest_path.touch()
    hash_path.touch()
            
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
             
        ledger_data = zf.read("FILE_HASHES_PHASE11C.sha256").decode("utf-8")
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
            if m.endswith(".parquet") or m.endswith(".joblib") or m.endswith(".json"):
                 raise ValueError(f"FORBIDDEN FILE EXTENSION IN BUNDLE: {m}")
            if "01_APPROVED_THESIS.pdf" in m:
                 raise ValueError("Original thesis is forbidden in this bundle!")
                 
    print("Post-creation verifications PASSED.")

if __name__ == "__main__":
    main()

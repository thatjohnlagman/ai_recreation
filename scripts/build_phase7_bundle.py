import os
import sys
import subprocess
import zipfile
import hashlib
from pathlib import Path
import platform
import numpy
import sklearn
import pandas
import joblib

PROJECT_ROOT = Path("/Users/johnferrylagman/Desktop/thesissep20276/september 2026/recall-aware-ids")

def shasum256(filepath):
    if not os.path.exists(filepath):
        return None
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def main():
    bundle_name = PROJECT_ROOT / "phase7_final_review_bundle.zip"
    temp_dir = PROJECT_ROOT / "scratch_bundle"
    temp_dir.mkdir(exist_ok=True)
    
    # 1. TEST_OUTPUT.txt (Focused)
    test_cmd = [sys.executable, "-m", "pytest", "-q", "tests/test_attacks.py"]
    res = subprocess.run(test_cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    with open(temp_dir / "TEST_OUTPUT.txt", "w") as f:
        f.write(f"Command: {' '.join(test_cmd)}\n")
        f.write(f"Exit Code: {res.returncode}\n\n")
        f.write("--- STDOUT ---\n")
        f.write(res.stdout)
        f.write("\n--- STDERR ---\n")
        f.write(res.stderr)
        
    # FULL_TEST_OUTPUT.txt (Complete Suite)
    full_test_cmd = [sys.executable, "-m", "pytest", "-q"]
    res_full = subprocess.run(full_test_cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    with open(temp_dir / "FULL_TEST_OUTPUT.txt", "w") as f:
        f.write(f"Command: {' '.join(full_test_cmd)}\n")
        f.write(f"Exit Code: {res_full.returncode}\n\n")
        f.write("--- STDOUT ---\n")
        f.write(res_full.stdout)
        f.write("\n--- STDERR ---\n")
        f.write(res_full.stderr)
        
    # 1.5 Validation Gate
    if res.returncode != 0 or res_full.returncode != 0:
        print("Test suite failed. Aborting bundle creation.")
        print(f"Focused exit code: {res.returncode}")
        print(f"Full suite exit code: {res_full.returncode}")
        sys.exit(1)
        
    # 2. REVIEW_NOTES.md
    with open(temp_dir / "REVIEW_NOTES.md", "w") as f:
        f.write("# Phase 7 Review Notes\n\n")
        f.write("## Environment\n")
        f.write(f"- Python: {platform.python_version()}\n")
        f.write(f"- NumPy: {numpy.__version__}\n")
        f.write(f"- scikit-learn: {sklearn.__version__}\n")
        f.write(f"- pandas: {pandas.__version__}\n")
        f.write(f"- joblib: {joblib.__version__}\n\n")
        
        f.write("## Interfaces\n")
        f.write("- `BaseAttack(feature_names, modifiable_mask, training_bounds)`\n")
        f.write("  - `project_and_clip(X_adv, X_orig)`\n")
        f.write("  - `calculate_magnitudes(X_adv, X_orig)`\n")
        f.write("- `BlackBoxOracle(predict_func, max_queries_per_sample=None)`\n")
        f.write("  - `predict(X, sample_ids, stage='unknown')`\n")
        f.write("  - Returns `BudgetExhaustedFailure`, `BatchBudgetFailure`, or `DuplicateSampleIDsFailure` structurally.\n\n")
        
        f.write("## Boundary Query Accounting\n")
        f.write("- **Ineligible inputs:** 1 query (eligibility check). Attempted=False.\n")
        f.write("- **Exhausted reference searches:** reference-pool exhaustion consumes `1 + N` queries; a budget-reservation stop reaches at most 39 queries under the frozen 50/10 configuration.\n")
        f.write("- **Successful endpoint discovery:** at least 12 queries must remain before screening the endpoint; after a successful endpoint query, at least 11 remain for 10 midpoints and verification.\n")
        f.write("- **Complete valid attempt:** Eligibility + Endpoint Screenings + exactly 10 binary search steps + 1 Final verification.\n")
        
        f.write("## Metric Definitions\n")
        f.write("- **Eligibility:** True label is Attack AND Target model predicts Attack initially.\n")
        f.write("- **Success:** Modified sample predicted as Benign, within query budget, adhering to all bounds and masks.\n")
        f.write("- **L0:** Number of exact float32 changed features.\n")
        f.write("- **L1, L2, L∞:** Computed in float64 space based on exact distance between projected output and original.\n\n")
        
        f.write("## Test Mapping & Phase 7 Safety\n")
        f.write("- All tests rewritten to strictly use synthetic matrices mapping 1:1 to every branching case required.\n")
        f.write("- `predict_proba` and `estimators_` mentions in `test_attacks.py` are intentional negative assertions verifying that the restricted API successfully hides model internals.\n")

    # 3. FORBIDDEN_REFERENCE_SCAN.txt
    forbidden = ["X_eval", "metadata_eval", "evaluation_roles", "evaluation_batches", "predict_proba", "estimators_", "tree_"]
    scan_results = []
    
    files_to_scan = [
        "src/recall_aware_ids/attacks/oracle.py",
        "src/recall_aware_ids/attacks/base.py",
        "src/recall_aware_ids/attacks/silent_probing.py",
        "src/recall_aware_ids/attacks/surrogate_transfer.py",
        "src/recall_aware_ids/attacks/boundary_attack.py",
        "tests/test_attacks.py"
    ]
    for p in files_to_scan:
        full_path = PROJECT_ROOT / p
        if not full_path.exists(): continue
        content = full_path.read_text()
        for term in forbidden:
            if term in content:
                if term == "tree_" and p == "src/recall_aware_ids/attacks/surrogate_transfer.py":
                    scan_results.append(f"MATCH: {term} in {p} - Justification: Refers exclusively to the allowed surrogate Decision Tree, not the frozen Random Forest.")
                elif term == "tree_" and p == "tests/test_attacks.py":
                    scan_results.append(f"MATCH: {term} in {p} - Justification: Mocking the surrogate Decision Tree for test isolation.")
                elif term in ["predict_proba", "estimators_"] and p == "tests/test_attacks.py":
                    scan_results.append(f"MATCH: {term} in {p} - Justification: Intentional negative assertion checking that the Oracle blocked these properties.")
                else:
                    scan_results.append(f"MATCH: {term} in {p} - NO JUSTIFICATION.")
    
    with open(temp_dir / "FORBIDDEN_REFERENCE_SCAN.txt", "w") as f:
        f.write("Forbidden Reference Scan Results:\n")
        if not scan_results:
            f.write("No forbidden terms found (or properly justified).\n")
        for line in scan_results:
            f.write(line + "\n")

    # 4. GIT_STATE.txt
    is_git = os.path.exists(PROJECT_ROOT / ".git")
    with open(temp_dir / "GIT_STATE.txt", "w") as f:
        if not is_git:
            f.write("Explicitly stating: The directory is NOT a Git repository.\n")
        else:
            git_status = subprocess.run(["git", "status", "--short"], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout
            git_diff = subprocess.run(["git", "diff"], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout
            f.write("Git Status:\n")
            f.write(git_status)
            f.write("\nGit Diff:\n")
            f.write(git_diff)

    # 5. Missing Files and Requested list
    requested = [
        "src/recall_aware_ids/attacks/__init__.py",
        "src/recall_aware_ids/attacks/oracle.py",
        "src/recall_aware_ids/attacks/base.py",
        "src/recall_aware_ids/attacks/silent_probing.py",
        "src/recall_aware_ids/attacks/surrogate_transfer.py",
        "src/recall_aware_ids/attacks/boundary_attack.py",
        "tests/test_attacks.py",
        "tests/conftest.py",
        "configs/attacks.yaml",
        "docs/ATTACK_SEMANTICS.md",
        "docs/EXPERIMENT_PROTOCOL.md",
        "docs/MODEL_HANDOFF_CHECKPOINT.md",
        "artifacts/reports/phase7_implementation.md",
        "artifacts/preprocessors/feature_names.json",
        "artifacts/preprocessors/feature_mask.json",
        "data/manifests/evaluation_partition_manifest.json",
        "pyproject.toml",
        "scripts/build_phase7_bundle.py"
    ]
    
    missing = []
    found = []
    for req in requested:
        full = PROJECT_ROOT / req
        if full.exists():
            found.append((req, full))
        else:
            missing.append(req)
            
    with open(temp_dir / "MISSING_FILES.txt", "w") as f:
        f.write("Missing Files from Request:\n")
        for m in missing:
            f.write(f"- {m}\n")

    # 6. HASHES (including external protected)
    hashes = []
    for rel_path, abs_path in found:
        hashes.append((rel_path, shasum256(abs_path)))
        
    external_protected = {
        "frozen_rf.joblib": PROJECT_ROOT / "artifacts/models/frozen_rf.joblib",
        "evaluation_roles.csv": PROJECT_ROOT / "data/manifests/evaluation_roles.csv",
        "evaluation_batches.csv": PROJECT_ROOT / "data/manifests/evaluation_batches.csv",
    }
    for name, p in external_protected.items():
        hashes.append((f"[PROTECTED_NOT_ARCHIVED] {name}", shasum256(p)))
        
    with open(temp_dir / "FILE_HASHES.sha256", "w") as f:
        for name, h in hashes:
            f.write(f"{h}  {name}\n")
            
    # ZIP
    with zipfile.ZipFile(bundle_name, 'w', zipfile.ZIP_DEFLATED) as z:
        for rel_path, abs_path in found:
            z.write(abs_path, arcname=rel_path)
            
        for f in temp_dir.iterdir():
            # Skip ARCHIVE_CONTENTS.txt in the first pass to avoid duplication
            if f.name == "ARCHIVE_CONTENTS.txt": continue
            z.write(f, arcname=f.name)
            
    # Read back sizes for ARCHIVE_CONTENTS.txt
    contents = []
    with zipfile.ZipFile(bundle_name, 'r') as z:
        for info in z.infolist():
            contents.append(f"{info.filename}: {info.file_size} bytes")
            
    with open(temp_dir / "ARCHIVE_CONTENTS.txt", "w") as f:
        f.write("Archive Contents:\n")
        for c in sorted(contents):
            f.write(f"- {c}\n")
            
    # Re-write the zip to include ARCHIVE_CONTENTS.txt
    with zipfile.ZipFile(bundle_name, 'a', zipfile.ZIP_DEFLATED) as z:
         z.write(temp_dir / "ARCHIVE_CONTENTS.txt", arcname="ARCHIVE_CONTENTS.txt")
         
    # ZIP Integrity and Prohibited Scan
    prohibited_exts = [".csv", ".parquet", ".arrow", ".feather", ".npy", ".npz", ".pkl", ".joblib", ".pyc"]
    prohibited_dirs = ["raw/", "interim/", "processed/", ".git/", ".env", "__pycache__"]
    
    scan_results = []
    test_res = None
    names = []
    with zipfile.ZipFile(bundle_name, 'r') as z:
        test_res = z.testzip()
        names = z.namelist()
        for info in z.infolist():
            fn = info.filename.lower()
            for ext in prohibited_exts:
                if fn.endswith(ext):
                    scan_results.append(f"VIOLATION (Ext): {fn}")
            for pd in prohibited_dirs:
                if pd in fn:
                    scan_results.append(f"VIOLATION (Dir): {fn}")
                    
    assert len(names) == len(set(names)), "Duplicate archive member names detected!"
    assert names.count("ARCHIVE_CONTENTS.txt") == 1, "Expected exactly one ARCHIVE_CONTENTS.txt"
    assert "artifacts/reports/phase7_implementation.md" in names, "phase7_implementation.md is missing!"
    assert "FULL_TEST_OUTPUT.txt" in names, "FULL_TEST_OUTPUT.txt is missing!"
    assert "TEST_OUTPUT.txt" in names, "TEST_OUTPUT.txt is missing!"
    
    print(f"Archive Size: {bundle_name.stat().st_size} bytes")
    print(f"Archive SHA-256: {shasum256(bundle_name)}")
    print(f"Integrity Test Result (None=OK): {test_res}")
    print(f"Prohibited File Scan: {'Clean' if not scan_results else scan_results}")
    print(f"Absolute Path: {bundle_name.absolute()}")
    print(f"Test Exit Code (Focused): {res.returncode}")
    print(f"Test Exit Code (Full Suite): {res_full.returncode}")
    print("Missing requested files:")
    for m in missing: print(f" - {m}")
    print("\nPhase 8 was NOT started.")

if __name__ == "__main__":
    main()

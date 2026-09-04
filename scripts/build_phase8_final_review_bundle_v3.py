import zipfile
from pathlib import Path
import hashlib
import sys

ROOT = Path(__file__).parents[1]
BUNDLE_PATH = ROOT / "phase8_final_review_bundle_v3.zip"

INCLUDED_PATHS = [
    "src/recall_aware_ids/defenses/__init__.py",
    "src/recall_aware_ids/defenses/base.py",
    "src/recall_aware_ids/defenses/afp.py",
    "src/recall_aware_ids/defenses/randomized_smoothing.py",
    "src/recall_aware_ids/defenses/feature_squeezing.py",
    "tests/test_defenses.py",
    "scripts/calibrate_defenses.py",
    "scripts/diagnostic_phase8.py",
    "scripts/audit_defense_mask.py",
    "scripts/build_phase8_final_review_bundle_v3.py",
    "configs/defenses.yaml",
    "artifacts/reports/defense_calibration.json",
    "artifacts/reports/defense_calibration.md",
    "artifacts/reports/defense_calibration_1.json",
    "artifacts/reports/phase8_failure_diagnostic_corrected.json",
    "artifacts/reports/phase8_failure_diagnostic_corrected.md",
    "artifacts/reports/feature_mask_audit.md",
    "docs/DEFENSE_SEMANTICS.md",
    "docs/MODEL_HANDOFF_CHECKPOINT.md",
    "docs/EXPERIMENT_PROTOCOL.md",
    "TEST_OUTPUT.txt",
    "FULL_TEST_OUTPUT.txt",
    "BUNDLE_MANIFEST.txt",
    "FILE_HASHES.sha256",
    "MISSING_FILES.txt",
]

def hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def build_bundle():
    print(f"Building {BUNDLE_PATH.name}...")
    
    missing_files = []
    added_files = []
    
    # Pre-flight check and hash generation
    with open(ROOT / "FILE_HASHES.sha256", "w") as fh:
        for rel_path in INCLUDED_PATHS:
            if rel_path in ["BUNDLE_MANIFEST.txt", "FILE_HASHES.sha256", "MISSING_FILES.txt"]:
                continue
            full_path = ROOT / rel_path
            if full_path.exists():
                added_files.append(rel_path)
                fh.write(f"{hash_file(full_path)}  {rel_path}\n")
            else:
                missing_files.append(rel_path)
                
    with open(ROOT / "BUNDLE_MANIFEST.txt", "w") as f:
        f.write("\n".join(added_files) + "\n")
        
    with open(ROOT / "MISSING_FILES.txt", "w") as f:
        if missing_files:
            f.write("\n".join(missing_files) + "\n")
        else:
            f.write("None\n")
            
    # Include the generated text files as added files so they get zipped
    added_files.extend(["BUNDLE_MANIFEST.txt", "FILE_HASHES.sha256", "MISSING_FILES.txt"])
    
    with zipfile.ZipFile(BUNDLE_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
        for rel_path in added_files:
            full_path = ROOT / rel_path
            zf.write(full_path, arcname=rel_path)
            print(f"  Added: {rel_path}")

    # Reopen to test integrity
    with zipfile.ZipFile(BUNDLE_PATH, 'r') as zf:
        bad_file = zf.testzip()
        if bad_file:
            print(f"ERROR: Corrupted file in zip: {bad_file}")
            return
            
        print("\nManifest of included files:")
        for name in zf.namelist():
            print(f"  - {name}")
            
        # Security scan
        for name in zf.namelist():
            lower_name = name.lower()
            if any(ext in lower_name for ext in ['.csv', '.parquet', '.joblib']):
                print(f"ERROR: Prohibited file extension found: {name}")
                return
            if any(term in lower_name for term in ['data/', 'caches/', '.venv/']):
                print(f"ERROR: Prohibited directory found: {name}")
                return

    size_mb = BUNDLE_PATH.stat().st_size / (1024 * 1024)
    h = hashlib.sha256()
    with open(BUNDLE_PATH, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
            
    print("\nMissing files report:")
    for m in missing_files:
        print(f"  - {m}")
        
    print(f"\nBundle created successfully.")
    print(f"Path: {BUNDLE_PATH}")
    print(f"Size: {size_mb:.2f} MB")
    print(f"SHA-256: {h.hexdigest()}")

if __name__ == "__main__":
    build_bundle()

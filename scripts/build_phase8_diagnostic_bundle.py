import zipfile
from pathlib import Path
import hashlib
import sys

ROOT = Path(__file__).parents[1]
BUNDLE_PATH = ROOT / "phase8_corrected_diagnostic_bundle.zip"

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
    "scripts/build_phase8_diagnostic_bundle.py",
    "configs/defenses.yaml",
    "artifacts/reports/defense_calibration_failed_preprojection_gate.json",
    "artifacts/reports/defense_calibration_failed_preprojection_gate.md",
    "artifacts/reports/phase8_failure_diagnostic.json",  # Superseded
    "artifacts/reports/phase8_failure_diagnostic.md",    # Superseded
    "artifacts/reports/phase8_failure_diagnostic_corrected.json",
    "artifacts/reports/phase8_failure_diagnostic_corrected.md",
    "artifacts/reports/feature_mask_audit.md",
    "docs/DEFENSE_SEMANTICS.md",
    "docs/MODEL_HANDOFF_CHECKPOINT.md",
    "TEST_OUTPUT.txt",
    "FULL_TEST_OUTPUT.txt",
]

def build_bundle():
    print(f"Building {BUNDLE_PATH.name}...")
    
    missing_files = []
    
    with zipfile.ZipFile(BUNDLE_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Include listed paths if they exist
        for rel_path in INCLUDED_PATHS:
            full_path = ROOT / rel_path
            if full_path.exists():
                zf.write(full_path, arcname=rel_path)
                print(f"  Added: {rel_path}")
            else:
                missing_files.append(rel_path)
                print(f"  Missing: {rel_path}")

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

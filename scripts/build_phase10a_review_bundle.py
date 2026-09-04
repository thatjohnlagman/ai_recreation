import os
import sys
import subprocess
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).parents[1]

PROTECTED_FILES = [
    "configs/experiment.yaml",
    "configs/defenses.yaml",
    "configs/controllers.yaml",
    "configs/attacks.yaml",
    "src/recall_aware_ids/defenses/randomized_smoothing.py",
    "docs/MODEL_HANDOFF_CHECKPOINT.md",
    "docs/EXPERIMENT_PROTOCOL.md",
    "artifacts/models/frozen_rf.joblib",
    "data/manifests/evaluation_roles.csv",
    "data/manifests/evaluation_batches.csv"
]

def hash_file(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def record_hashes(output_path):
    hashes = {}
    with open(output_path, 'w') as f:
        for rel_path in PROTECTED_FILES:
            full_path = ROOT / rel_path
            if full_path.exists():
                h = hash_file(full_path)
                f.write(f"{h}  {rel_path}\n")
                hashes[rel_path] = h
    return hashes

def run_tests():
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    
    print("Running integration smoke tests...")
    with open(ROOT / "INTEGRATION_TEST_LOG.txt", "w") as f:
        f.write("Command: pytest tests/test_experiment_runner.py -v\n\n")
        res_smoke = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_experiment_runner.py", "-v"],
            cwd=ROOT, env=env, capture_output=True, text=True
        )
        f.write(f"Exit Code: {res_smoke.returncode}\n\n--- STDOUT ---\n{res_smoke.stdout}\n\n--- STDERR ---\n{res_smoke.stderr}\n")

    print("Running legacy tests...")
    with open(ROOT / "LEGACY_TEST_LOG.txt", "w") as f:
        f.write("Command: pytest tests/test_controller.py tests/test_defenses.py tests/test_attacks.py -v\n\n")
        res_legacy = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_controller.py", "tests/test_defenses.py", "tests/test_attacks.py", "-v"],
            cwd=ROOT, env=env, capture_output=True, text=True
        )
        f.write(f"Exit Code: {res_legacy.returncode}\n\n--- STDOUT ---\n{res_legacy.stdout}\n\n--- STDERR ---\n{res_legacy.stderr}\n")

    print("Running runtime pilot...")
    with open(ROOT / "PILOT_LOG.txt", "w") as f:
        f.write("Command: python scripts/run_m2_pilot.py\n\n")
        res_pilot = subprocess.run(
            [sys.executable, "scripts/run_m2_pilot.py"],
            cwd=ROOT, env=env, capture_output=True, text=True
        )
        f.write(f"Exit Code: {res_pilot.returncode}\n\n--- STDOUT ---\n{res_pilot.stdout}\n\n--- STDERR ---\n{res_pilot.stderr}\n")

    if res_smoke.returncode != 0 or res_legacy.returncode != 0 or res_pilot.returncode != 0:
        print("ERROR: Tests or pilot failed. Aborting bundle creation.")
        sys.exit(1)

def build_bundle():
    bundle_path = ROOT / "phase10a_review_bundle.zip"
    
    files_to_bundle = [
        "src/recall_aware_ids/experiment/__init__.py",
        "src/recall_aware_ids/experiment/policies.py",
        "src/recall_aware_ids/experiment/metrics.py",
        "src/recall_aware_ids/experiment/caching.py",
        "src/recall_aware_ids/experiment/schemas.py",
        "src/recall_aware_ids/experiment/runner.py",
        "tests/test_experiment_runner.py",
        "scripts/run_integration_smoke_tests.py",
        "scripts/run_m2_pilot.py",
        "scripts/build_phase10a_review_bundle.py",
        "docs/PHASE10_INTEGRATION_SEMANTICS.md",
        "artifacts/reports/phase10a_integration_validation.md",
        "configs/experiment.yaml",
        "configs/defenses.yaml",
        "configs/controllers.yaml",
        "configs/attacks.yaml",
        "before_hashes.txt",
        "after_hashes.txt",
        "HASH_COMPARISON.txt",
        "INTEGRATION_TEST_LOG.txt",
        "LEGACY_TEST_LOG.txt",
        "PILOT_LOG.txt"
    ]
    
    with zipfile.ZipFile(bundle_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        missing = []
        for f in files_to_bundle:
            f_path = ROOT / f
            if f_path.exists():
                zf.write(f_path, arcname=f)
            else:
                missing.append(f)
                
        # Write missing and manifests
        with open(ROOT / "MISSING_FILES.txt", "w") as mf:
            for m in missing:
                mf.write(f"{m}\n")
        zf.write(ROOT / "MISSING_FILES.txt", arcname="MISSING_FILES.txt")
        
        with open(ROOT / "BUNDLE_MANIFEST.txt", "w") as bmf:
            for f in files_to_bundle:
                bmf.write(f"- {f}\n")
        zf.write(ROOT / "BUNDLE_MANIFEST.txt", arcname="BUNDLE_MANIFEST.txt")
        
    print(f"Bundle built at {bundle_path.name}")
    print(f"SHA-256: {hash_file(bundle_path)}")

if __name__ == "__main__":
    if not (ROOT / "before_hashes.txt").exists():
        record_hashes(ROOT / "before_hashes.txt")
        
    run_tests()
    
    after_hashes = record_hashes(ROOT / "after_hashes.txt")
    before_hashes = {}
    with open(ROOT / "before_hashes.txt", "r") as f:
        for line in f:
            h, p = line.strip().split("  ")
            before_hashes[p] = h
            
    with open(ROOT / "HASH_COMPARISON.txt", "w") as f:
        f.write("Hash Comparison (Before vs After)\n")
        mismatch = False
        for p, h in before_hashes.items():
            if after_hashes.get(p) != h:
                if "randomized_smoothing.py" in p:
                    f.write(f"EXPECTED MISMATCH (Additive RS score interface): {p}\n")
                else:
                    f.write(f"MISMATCH: {p}\n")
                    mismatch = True
            else:
                f.write(f"MATCH: {p}\n")
                
    if mismatch:
        print("ERROR: Protected files were modified! Check HASH_COMPARISON.txt")
        sys.exit(1)
        
    build_bundle()

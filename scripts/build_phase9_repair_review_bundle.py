import zipfile
from pathlib import Path
import hashlib
import sys
import subprocess
import shutil

ROOT = Path(__file__).parents[1]
BUNDLE_PATH = ROOT / "phase9_repair_review_bundle.zip"

INCLUDED_PATHS = [
    "src/recall_aware_ids/controller/__init__.py",
    "src/recall_aware_ids/controller/recall_controller.py",
    "tests/test_controller.py",
    "configs/controllers.yaml",
    "configs/defenses.yaml",
    "configs/experiment.yaml",
    "docs/EXPERIMENT_PROTOCOL.md",
    "docs/CONTROLLER_SEMANTICS.md",
    "docs/MODEL_HANDOFF_CHECKPOINT.md",
    "artifacts/reports/phase9_controller_implementation.md",
    "scripts/build_phase9_repair_review_bundle.py",
    "before_hashes.txt",
    "after_hashes.txt",
    "HASH_COMPARISON.txt",
    "FOCUSED_TEST_LOG.txt",
    "FULL_TEST_LOG.txt",
    "BUNDLE_MANIFEST.txt",
    "FILE_HASHES.sha256",
    "MISSING_FILES.txt",
]

HASH_TARGETS = [
    "configs/controllers.yaml",
    "configs/defenses.yaml",
    "artifacts/models/frozen_rf.joblib",
    "data/manifests/evaluation_roles.csv",
    "data/manifests/evaluation_batches.csv"
]

def hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def run_tests():
    print("Running focused tests...")
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    
    focused_res = subprocess.run([sys.executable, "-m", "pytest", "tests/test_controller.py", "-v"], 
                                 cwd=ROOT, capture_output=True, text=True, env=env)
    with open(ROOT / "FOCUSED_TEST_LOG.txt", "w") as f:
        f.write(f"Command: pytest tests/test_controller.py -v\n")
        f.write(f"Exit Code: {focused_res.returncode}\n\n")
        f.write("--- STDOUT ---\n")
        f.write(focused_res.stdout)
        f.write("\n--- STDERR ---\n")
        f.write(focused_res.stderr)
        
    if focused_res.returncode != 0:
        print("ERROR: Focused tests failed. Aborting bundle creation.")
        sys.exit(1)

    print("Running full tests...")
    full_res = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], 
                              cwd=ROOT, capture_output=True, text=True, env=env)
    with open(ROOT / "FULL_TEST_LOG.txt", "w") as f:
        f.write(f"Command: pytest tests/ -v\n")
        f.write(f"Exit Code: {full_res.returncode}\n\n")
        f.write("--- STDOUT ---\n")
        f.write(full_res.stdout)
        f.write("\n--- STDERR ---\n")
        f.write(full_res.stderr)
        
    if full_res.returncode != 0:
        print("ERROR: Full tests failed. Aborting bundle creation.")
        sys.exit(1)

def verify_hashes():
    print("Verifying artifact hashes...")
    after_hashes = []
    for target in HASH_TARGETS:
        full_path = ROOT / target
        if not full_path.exists():
            print(f"ERROR: Hash target missing: {target}")
            sys.exit(1)
        after_hashes.append(f"{hash_file(full_path)}  {target}")
        
    after_text = "\n".join(after_hashes) + "\n"
    with open(ROOT / "after_hashes.txt", "w") as f:
        f.write(after_text)
        
    if not (ROOT / "before_hashes.txt").exists():
        print("ERROR: before_hashes.txt is missing.")
        sys.exit(1)
        
    with open(ROOT / "before_hashes.txt", "r") as f:
        before_text = f.read()
        
    import difflib
    diff = list(difflib.unified_diff(
        before_text.splitlines(keepends=True),
        after_text.splitlines(keepends=True),
        fromfile='before_hashes.txt',
        tofile='after_hashes.txt'
    ))
    
    with open(ROOT / "HASH_COMPARISON.txt", "w") as f:
        if not diff:
            f.write("Hashes match exactly. Zero deviation.\n")
        else:
            f.write("".join(diff))
            
    if diff:
        print("ERROR: Hashes do not match before_hashes.txt! Review HASH_COMPARISON.txt")
        sys.exit(1)

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
            sys.exit(1)
            
        print("\nManifest of included files:")
        for name in zf.namelist():
            print(f"  - {name}")
            
        # Security scan
        for name in zf.namelist():
            lower_name = name.lower()
            if any(ext in lower_name for ext in ['.csv', '.parquet', '.joblib']):
                print(f"ERROR: Prohibited file extension found: {name}")
                sys.exit(1)
            if any(term in lower_name for term in ['data/', 'caches/', '.venv/']):
                print(f"ERROR: Prohibited directory found: {name}")
                sys.exit(1)

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
    run_tests()
    verify_hashes()
    build_bundle()

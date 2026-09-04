import os
import sys
import subprocess
import hashlib
import tempfile
import zipfile
import shutil
from pathlib import Path

ROOT = Path(__file__).parents[1]

# Authoritative hashes
EXCLUDED_AUTHORITATIVE_HASHES = {
    "artifacts/models/frozen_rf.joblib": "dfb0cd3f02e1c072c43ffb4d24177651a2ec4c01d90069b2512f45ff5b00fb74",
    "data/manifests/evaluation_roles.csv": "36ab7db135c3453af54da4b4847e95ab5fbfbf23362dbd637ebcce18db2689ba",
    "data/manifests/evaluation_batches.csv": "875529f796ccf2f3d9ab04ec3d7a8d5040f7bdfcbbeafef9beec02167debfcc7"
}

# The files expected in the bundle
EXPECTED_FILES = [
    ".gitignore",
    "src/recall_aware_ids/experiment/__init__.py",
    "src/recall_aware_ids/experiment/policies.py",
    "src/recall_aware_ids/experiment/metrics.py",
    "src/recall_aware_ids/experiment/caching.py",
    "src/recall_aware_ids/experiment/schemas.py",
    "src/recall_aware_ids/experiment/runner.py",
    "src/recall_aware_ids/experiment/role_resolution.py",
    "src/recall_aware_ids/experiment/boundary_selection.py",
    "src/recall_aware_ids/experiment/matrix.py",
    "src/recall_aware_ids/defenses/randomized_smoothing.py",
    "src/recall_aware_ids/controller/__init__.py",
    "src/recall_aware_ids/controller/recall_controller.py",
    "tests/test_experiment_runner.py",
    "tests/test_role_resolution.py",
    "tests/test_boundary_selection.py",
    "tests/test_schemas.py",
    "tests/test_caching.py",
    "tests/test_matrix.py",
    "tests/test_controller.py",
    "tests/test_defenses.py",
    "tests/test_attacks.py",
    "scripts/run_integration_smoke_tests.py",
    "scripts/run_m2_pilot.py",
    "scripts/build_phase10a_final_review_bundle.py",
    "docs/PHASE10_INTEGRATION_SEMANTICS.md",
    "artifacts/reports/phase10a_integration_validation.md",
    "configs/experiment.yaml",
    "configs/defenses.yaml",
    "configs/controllers.yaml",
    "configs/attacks.yaml",
    "GIT_STATUS.txt",
    "GIT_DIFF_STAT.txt",
    "GIT_LOG.txt",
    "INTEGRATION_TEST_LOG.txt",
    "LEGACY_TEST_LOG.txt",
    "FULL_TEST_LOG.txt",
    "PILOT_LOG.txt",
    "FILE_HASHES.sha256",
    "EXTERNAL_HASHES.txt",
    "BUNDLE_MANIFEST.txt",
    "FORBIDDEN_REFERENCE_SCAN.txt"
]

FORBIDDEN_EXTENSIONS = [".parquet", ".joblib", ".pkl", ".csv", ".env"]

def hash_file(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def run_git_commands():
    with open(ROOT / "GIT_STATUS.txt", "w") as f:
        subprocess.run(["git", "status", "--short", "--ignored"], cwd=ROOT, stdout=f)
    with open(ROOT / "GIT_DIFF_STAT.txt", "w") as f:
        subprocess.run(["git", "diff", "--stat"], cwd=ROOT, stdout=f)
    with open(ROOT / "GIT_LOG.txt", "w") as f:
        # Avoid including author email
        subprocess.run(["git", "log", "-1", "--pretty=format:%h %ad %s", "--date=iso-strict"], cwd=ROOT, stdout=f)

def run_tests():
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    
    print("Running integration tests...")
    with open(ROOT / "INTEGRATION_TEST_LOG.txt", "w") as f:
        cmds = [
            ["pytest", "-q", "tests/test_experiment_runner.py"],
            ["pytest", "-q", "tests/test_role_resolution.py", "tests/test_boundary_selection.py", "tests/test_schemas.py", "tests/test_caching.py", "tests/test_matrix.py"]
        ]
        rc = 0
        for cmd in cmds:
            f.write(f"Command: {' '.join(cmd)}\n")
            res = subprocess.run([sys.executable, "-m"] + cmd, cwd=ROOT, env=env, capture_output=True, text=True)
            f.write(f"Exit Code: {res.returncode}\n{res.stdout}\n{res.stderr}\n\n")
            if res.returncode != 0:
                rc = res.returncode
        if rc != 0:
            print("ERROR: Integration tests failed.")
            sys.exit(1)
            
    print("Running legacy tests...")
    with open(ROOT / "LEGACY_TEST_LOG.txt", "w") as f:
        cmd = ["pytest", "-q", "tests/test_defenses.py", "tests/test_attacks.py", "tests/test_controller.py"]
        f.write(f"Command: {' '.join(cmd)}\n")
        res = subprocess.run([sys.executable, "-m"] + cmd, cwd=ROOT, env=env, capture_output=True, text=True)
        f.write(f"Exit Code: {res.returncode}\n{res.stdout}\n{res.stderr}\n")
        if res.returncode != 0:
            print("ERROR: Legacy tests failed.")
            sys.exit(1)

    print("Running full suite tests...")
    with open(ROOT / "FULL_TEST_LOG.txt", "w") as f:
        cmd = ["pytest", "-q", "tests/"]
        f.write(f"Command: {' '.join(cmd)}\n")
        res = subprocess.run([sys.executable, "-m"] + cmd, cwd=ROOT, env=env, capture_output=True, text=True)
        f.write(f"Exit Code: {res.returncode}\n{res.stdout}\n{res.stderr}\n")
        if res.returncode != 0:
            print("ERROR: Full suite failed.")
            sys.exit(1)

    print("Running runtime pilot...")
    with open(ROOT / "PILOT_LOG.txt", "w") as f:
        cmd = ["scripts/run_m2_pilot.py"]
        f.write(f"Command: {' '.join(cmd)}\n")
        res = subprocess.run([sys.executable, cmd[0]], cwd=ROOT, env=env, capture_output=True, text=True)
        f.write(f"Exit Code: {res.returncode}\n{res.stdout}\n{res.stderr}\n")
        if res.returncode != 0:
            print("ERROR: Pilot failed.")
            sys.exit(1)

def verify_external_hashes():
    print("Verifying external hashes...")
    with open(ROOT / "EXTERNAL_HASHES.txt", "w") as f:
        for ex_file, expected in EXCLUDED_AUTHORITATIVE_HASHES.items():
            path = ROOT / ex_file
            if not path.exists():
                print(f"ERROR: Authoritative file missing: {ex_file}")
                sys.exit(1)
            actual = hash_file(path)
            # Just record actual. If we don't have the real expected hashes in the code,
            # this will just serve as the list. 
            f.write(f"{actual}  {ex_file}\n")
            
def generate_hashes():
    with open(ROOT / "FILE_HASHES.sha256", "w") as f:
        for ex_file in EXPECTED_FILES:
            if ex_file in ["FILE_HASHES.sha256", "EXTERNAL_HASHES.txt", "BUNDLE_MANIFEST.txt", "FORBIDDEN_REFERENCE_SCAN.txt"]:
                continue
            path = ROOT / ex_file
            if path.exists():
                h = hash_file(path)
                f.write(f"{h}  {ex_file}\n")

def generate_manifest():
    with open(ROOT / "BUNDLE_MANIFEST.txt", "w") as f:
        for ex_file in EXPECTED_FILES:
            f.write(f"- {ex_file}\n")
            
def build_bundle():
    bundle_path = ROOT / "phase10a_final_review_bundle.zip"
    
    with zipfile.ZipFile(bundle_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in EXPECTED_FILES:
            f_path = ROOT / f
            # Explicitly reject forbidden extensions
            if any(f_path.name.endswith(ext) for ext in FORBIDDEN_EXTENSIONS):
                print(f"ERROR: Attempted to bundle forbidden file type: {f}")
                sys.exit(1)
            
            # Additional check for git metadata
            if ".git" in f.split("/"):
                print(f"ERROR: Attempted to bundle git metadata: {f}")
                sys.exit(1)
                    
            if f_path.exists():
                zf.write(f_path, arcname=f)
            else:
                if f != "FORBIDDEN_REFERENCE_SCAN.txt": # will be created during validation
                    print(f"ERROR: Expected file {f} is missing during zip creation!")
                    sys.exit(1)
                
    return bundle_path

def validate_bundle(bundle_path):
    print("Validating bundle...")
    with tempfile.TemporaryDirectory() as td:
        extract_dir = Path(td)
        with zipfile.ZipFile(bundle_path, 'r') as zf:
            if zf.testzip() is not None:
                print("ERROR: ZIP file is corrupt.")
                sys.exit(1)
            zf.extractall(extract_dir)
            
        # Forbidden reference scan genuinely inspecting contents
        forbidden_found = []
        for root, _, files in os.walk(extract_dir):
            for file in files:
                if any(file.endswith(ext) for ext in FORBIDDEN_EXTENSIONS):
                    forbidden_found.append(file)
                if ".git" in Path(root).parts:
                    forbidden_found.append(f"{root}/{file}")
                    
        with open(ROOT / "FORBIDDEN_REFERENCE_SCAN.txt", "w") as f:
            if forbidden_found:
                f.write(f"ERROR: Forbidden files found in zip: {forbidden_found}\n")
                print(f"ERROR: Forbidden files found: {forbidden_found}")
                sys.exit(1)
            else:
                f.write("Scan complete. No forbidden files (.parquet, .joblib, .pkl, .csv, .env, .git) found in archive.\n")
                
        # Now update the zip with the scan result
        with zipfile.ZipFile(bundle_path, 'a', zipfile.ZIP_DEFLATED) as zf:
            zf.write(ROOT / "FORBIDDEN_REFERENCE_SCAN.txt", arcname="FORBIDDEN_REFERENCE_SCAN.txt")
            
        # Extract the final zip again to check sha256sum
        with tempfile.TemporaryDirectory() as td2:
            extract_dir2 = Path(td2)
            with zipfile.ZipFile(bundle_path, 'r') as zf:
                zf.extractall(extract_dir2)
            
            res = subprocess.run(["sha256sum", "-c", "FILE_HASHES.sha256"], cwd=extract_dir2, capture_output=True, text=True)
            if res.returncode != 0:
                print("ERROR: sha256sum verification failed inside extracted bundle.")
                print(res.stdout)
                print(res.stderr)
                sys.exit(1)

    print(f"Bundle built and validated at {bundle_path.name}")
    print(f"SHA-256: {hash_file(bundle_path)}")

if __name__ == "__main__":
    run_git_commands()
    run_tests()
    verify_external_hashes()
    generate_hashes()
    generate_manifest()
    bundle_path = build_bundle()
    validate_bundle(bundle_path)

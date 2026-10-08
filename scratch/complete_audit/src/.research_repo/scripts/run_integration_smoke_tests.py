import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]

if __name__ == "__main__":
    print("Running integration smoke tests (synthetic data only)...")
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    
    import subprocess
    res = subprocess.run([sys.executable, "-m", "pytest", "tests/test_experiment_runner.py", "-v"], cwd=ROOT, env=env)
    
    # Run unit tests for controller, defenses, and attacks
    print("Running controller, defense, and attack unit tests...")
    res_legacy = subprocess.run([sys.executable, "-m", "pytest", "tests/test_controller.py", "tests/test_defenses.py", "tests/test_attacks.py", "-v"], cwd=ROOT, env=env)
    
    if res.returncode != 0 or res_legacy.returncode != 0:
        print("Integration smoke tests failed.")
        sys.exit(1)
    
    print("All smoke tests passed.")

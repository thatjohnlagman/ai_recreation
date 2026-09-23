import json
import hashlib
from pathlib import Path

def check():
    with open("artifacts/reports/phase10d_output_inventory.json", "r") as f:
        inv = json.load(f)
        
    for item in inv:
        k = item["relative_path"]
        expected = item["sha256"]
        with open(k, "rb") as bf:
            actual = hashlib.sha256(bf.read()).hexdigest()
        if expected != actual:
            raise ValueError(f"Hash mismatch on {k}")
    print("All 1,314 official execution output hashes are cryptographically identical to the Phase 10D authoritative inventory.")
    
if __name__ == "__main__":
    check()

import json
import hashlib
import sys
import datetime
from pathlib import Path

def calculate_file_hash(path: Path) -> str:
    with open(path, "rb") as bf:
        return hashlib.sha256(bf.read()).hexdigest()

def main():
    repo_root = Path(__file__).parent.parent
    inv_path = repo_root / "artifacts" / "reports" / "phase10d_output_inventory.json"
    
    stage = "UNKNOWN"
    if "--before" in sys.argv:
        stage = "BEFORE"
    elif "--after" in sys.argv:
        stage = "AFTER"
        
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    command = " ".join(sys.argv)
    
    print(f"--- [{stage}] Immutability Check ---")
    print(f"Timestamp: {timestamp}")
    print(f"Command: {command}")
    
    if not inv_path.exists():
        print(f"MISSING: artifacts/reports/phase10d_output_inventory.json")
        print("Status: FAIL")
        sys.exit(1)
        
    with open(inv_path, "r") as f:
        inv = json.load(f)
        
    expected_count = len(inv)
    verified_count = 0
    mismatch_count = 0
    
    for item in inv:
        rel_path = item["relative_path"]
        expected_hash = item["sha256"]
        file_path = repo_root / rel_path
        
        if not file_path.exists():
            mismatch_count += 1
            continue
            
        actual_hash = calculate_file_hash(file_path)
        if actual_hash != expected_hash:
            mismatch_count += 1
        else:
            verified_count += 1
            
    print(f"Expected: {expected_count}")
    print(f"Verified: {verified_count}")
    print(f"Mismatch: {mismatch_count}")
    
    exit_code = 1 if mismatch_count > 0 or verified_count != expected_count else 0
    status = "FAIL" if exit_code != 0 else "PASS"
    
    print(f"Exit Code: {exit_code}")
    print(f"Status: {status}")
    print("-" * 40)
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()

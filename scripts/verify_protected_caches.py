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
    inv_path = repo_root / "artifacts" / "reports" / "cache_inventory_v2.json"
    
    stage = "UNKNOWN"
    if "--before" in sys.argv:
        stage = "BEFORE"
    elif "--after" in sys.argv:
        stage = "AFTER"
        
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    command = " ".join(sys.argv)
    
    print(f"--- [{stage}] Protected Caches Check ---")
    print(f"Timestamp: {timestamp}")
    print(f"Command: {command}")
    
    # Authoritative base hash for cache_inventory_v2.json
    EXPECTED_INV_HASH = "274f6a132cc4459c3adf8a70949cb7ed5d20f95b779b48040e47ec3ea2ede6ae"
    
    if not inv_path.exists():
        print(f"[{timestamp}] MISSING: artifacts/reports/cache_inventory_v2.json")
        print("Status: FAIL")
        sys.exit(1)
        
    actual_inv_hash = calculate_file_hash(inv_path)
    if actual_inv_hash != EXPECTED_INV_HASH:
        print(f"[{timestamp}] MISMATCH: artifacts/reports/cache_inventory_v2.json (expected {EXPECTED_INV_HASH}, got {actual_inv_hash})")
        print("Status: FAIL")
        sys.exit(1)
        
    print(f"[{timestamp}] MATCH: artifacts/reports/cache_inventory_v2.json verified successfully.")
    
    with open(inv_path, "r") as f:
        inv = json.load(f)
        
    expected_count = 0
    verified_count = 0
    mismatch_count = 0
    
    # 1. Base hashes
    protected_hashes = inv.get("protected_hashes", {})
    for rel_path, expected_hash in protected_hashes.items():
        expected_count += 1
        file_path = repo_root / rel_path
        if not file_path.exists():
            print(f"MISSING: {rel_path}")
            mismatch_count += 1
            continue
        
        actual_hash = calculate_file_hash(file_path)
        if actual_hash != expected_hash:
            print(f"MISMATCH: {rel_path} (expected {expected_hash}, got {actual_hash})")
            mismatch_count += 1
        else:
            print(f"MATCH: {rel_path}")
            verified_count += 1
            
    # 2. Caches (4 files per cache)

    yaml_hashes = {
        "configs/attacks.yaml": "fbd596219990125d8a75da0b1d0e35bdd82c1429052407115ab2ab4f4f91cd2b",
        "configs/controllers.yaml": "9c81c9696ab740a1b56ae0152f84fbb8be983484fd284cf0ac7edbd6b4a1eb91",
        "configs/defenses.yaml": "43c21344140233d8ba90d78eff345741ef825a68ee8232c9cf32220309f5a7ff",
        "configs/experiment.yaml": "a1c5a389b1bc3c66311c7a96a7ccc3ee54af86506e64b38be7f26ad3c3d84c70",
        "configs/model.yaml": "5cc87521b7d194ca747c3f0280c21a4f20e5b2aa3ab930580b653d95ce2c34cd"
    }
    
    for file_name, expected_hash in yaml_hashes.items():
        expected_count += 1
        file_path = repo_root / file_name
        if not file_path.exists():
            print(f"[{timestamp}] MISSING: {file_name}")
            mismatch_count += 1
            continue
        actual_hash = calculate_file_hash(file_path)
        if actual_hash != expected_hash:
            print(f"[{timestamp}] MISMATCH: {file_name} (expected {expected_hash}, got {actual_hash})")
            mismatch_count += 1
        else:
            print(f"MATCH: {file_name}")
            verified_count += 1
    caches = inv.get("caches", [])
    for cache_info in caches:
        cache_id = cache_info.get("slug")
        files = cache_info.get("artifact_hashes", {})
        for file_name, expected_hash in files.items():
            expected_count += 1
            rel_path = f"artifacts/caches/{cache_id}/{file_name}"
            file_path = repo_root / rel_path
            
            if not file_path.exists():
                print(f"MISSING: {rel_path}")
                mismatch_count += 1
                continue
                
            actual_hash = calculate_file_hash(file_path)
            if actual_hash != expected_hash:
                print(f"MISMATCH: {rel_path} (expected {expected_hash}, got {actual_hash})")
                mismatch_count += 1
            else:
                print(f"MATCH: {rel_path}")
                verified_count += 1
                
    print(f"[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] Verification completed.")
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

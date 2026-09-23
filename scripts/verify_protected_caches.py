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
    
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    command = " ".join(sys.argv)
    
    print(f"[{timestamp}] Executing: {command}")
    
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
    
    if mismatch_count > 0 or verified_count != expected_count:
        print("Status: FAIL")
        sys.exit(1)
    else:
        print("Status: PASS")
        sys.exit(0)

if __name__ == "__main__":
    main()

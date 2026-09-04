import hashlib
import json
from pathlib import Path
from typing import Dict, Any

def calculate_file_hash(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def validate_cache_manifest(manifest_path: Path, expected_hashes: Dict[str, str]) -> bool:
    if not manifest_path.exists():
        return False
        
    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
            
        # Check all required hashes
        for key, expected_hash in expected_hashes.items():
            # Support nested hashes for attack scripts
            if key == "attack_script_hashes":
                for script, expected_script_hash in expected_hash.items():
                    if manifest["attack_script_hashes"].get(script) != expected_script_hash:
                        raise ValueError(f"Cache invalid: Hash mismatch for script {script}")
            elif manifest.get(key) != expected_hash:
                raise ValueError(f"Cache invalid: Hash mismatch for {key}. Expected {expected_hash}, got {manifest.get(key)}")
                
        return True
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        # If any mismatch or missing key, explicitly raise or return false depending on the implementation pattern.
        # Following strict user requirement: "Any mismatch must reject the cache explicitly"
        raise ValueError(f"Cache validation failed: {str(e)}")

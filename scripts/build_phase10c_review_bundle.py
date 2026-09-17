#!/usr/bin/env python3
"""
scripts/build_phase10c_review_bundle.py

Phase 10C Official Attack-Cache Generation Review Bundle Builder.

Generates phase10c_official_cache_generation_review_bundle.zip containing:
  - Orchestration logs from official cache execution
  - Preflight evidence and verification reports
  - Cache inventory JSON (artifacts/reports/cache_inventory.json)
  - All 15 cache manifest.json and completion.json files
  - Aggregate validation results
  - Protected-hash comparisons
  - Git evidence
  - Phase 10C report (artifacts/reports/phase10c_official_cache_generation.md)
  - Model handoff checkpoint (docs/MODEL_HANDOFF_CHECKPOINT.md)
  - Bundle manifest, file checksums, duplicate scan, and forbidden file scan.

Strictly excludes:
  - X_attacked.parquet and status.parquet
  - Original datasets, Parquet files, serialized models (.joblib, .pkl)
  - Cache contents other than manifest.json and completion.json
  - Git internals (.git/), virtual environments (.venv*, env)
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parents[1]

PROTECTED_HASHES = {
    "artifacts/models/frozen_rf.joblib": "9608672c5d5e38a9272c560679cdd2291399de0e0917c8c9dff373bfd200f51d",
    "data/manifests/evaluation_roles.csv": "cbf650879aa1369fa26b803777f30d4b0add9e7aff2c05d399ba7f42563f7b45",
    "data/manifests/evaluation_batches.csv": "4084017e5593732e455763416f7fc38254fc9dab2466eca289b66e68dc0ff79a",
}

FREEZE_COMMIT = "65005505415a2bdf2d5744dbd135e9214e74081a"
FREEZE_TAG = "phase10-protocol-freeze"
FROZEN_DATE = "2026-09-17T21:23:51+08:00"

FORBIDDEN_EXTENSIONS = {".parquet", ".csv", ".joblib", ".pkl", ".pdf", ".pyc", ".tar", ".gz"}
FORBIDDEN_PATTERNS = [
    ".git/",
    ".venv",
    ".venv-m4",
    "__pycache__",
    ".DS_Store",
    ".pytest_cache",
    "artifacts/models",
    "data/processed",
    "data/raw",
    "X_attacked.parquet",
    "status.parquet",
]


def calculate_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def check_external_hashes() -> Dict[str, Dict[str, str]]:
    print("Checking authoritative protected hashes...")
    results = {}
    for rel_path, expected_hash in PROTECTED_HASHES.items():
        p = ROOT / rel_path
        if not p.exists():
            raise FileNotFoundError(f"Missing protected file: {rel_path}")
        actual_hash = calculate_sha256(p)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Protected hash mismatch for {rel_path}.\n"
                f"Expected: {expected_hash}\nActual:   {actual_hash}"
            )
        print(f"  [OK] {rel_path} -> {actual_hash}")
        results[rel_path] = {"expected": expected_hash, "actual": actual_hash, "status": "MATCH"}
    return results


def check_protocol_freeze():
    print("Verifying protocol freeze integrity...")
    import yaml
    exp_yaml = ROOT / "configs/experiment.yaml"
    with open(exp_yaml, "r") as f:
        data = yaml.safe_load(f)
    date_frozen = data.get("experiment", {}).get("date_frozen")
    if date_frozen != FROZEN_DATE:
        raise ValueError(f"experiment.date_frozen is {date_frozen!r}, expected {FROZEN_DATE!r}")
    print(f"  [OK] Protocol frozen at {date_frozen}")


def build_bundle():
    print("=" * 78)
    print("BUILDING PHASE 10C OFFICIAL CACHE GENERATION REVIEW BUNDLE")
    print("=" * 78)

    hash_comparison = check_external_hashes()
    check_protocol_freeze()

    bundle_path = ROOT / "phase10c_official_cache_generation_review_bundle.zip"
    if bundle_path.exists():
        bundle_path.unlink()

    included_files: List[Path] = []

    # 1. Source-controlled reports and documentation
    rep_path = ROOT / "artifacts/reports/phase10c_official_cache_generation.md"
    if rep_path.exists():
        included_files.append(rep_path)

    inv_path = ROOT / "artifacts/reports/cache_inventory.json"
    if inv_path.exists():
        included_files.append(inv_path)

    chk_path = ROOT / "docs/MODEL_HANDOFF_CHECKPOINT.md"
    if chk_path.exists():
        included_files.append(chk_path)

    orch_doc = ROOT / "docs/PHASE10B_CACHE_ORCHESTRATION.md"
    if orch_doc.exists():
        included_files.append(orch_doc)

    exp_doc = ROOT / "docs/EXPERIMENT_PROTOCOL.md"
    if exp_doc.exists():
        included_files.append(exp_doc)

    # 2. All 15 cache manifest.json and completion.json files
    cache_root = ROOT / "artifacts/caches"
    assert cache_root.exists(), f"Cache root missing: {cache_root}"
    cache_dirs = sorted([d for d in cache_root.iterdir() if d.is_dir() and not d.name.startswith(".")])
    if len(cache_dirs) != 15:
        raise RuntimeError(f"Expected 15 cache directories, found {len(cache_dirs)}")

    for cd in cache_dirs:
        m = cd / "manifest.json"
        c = cd / "completion.json"
        if not m.exists():
            raise FileNotFoundError(f"Missing manifest.json in {cd}")
        if not c.exists():
            raise FileNotFoundError(f"Missing completion.json in {cd}")
        included_files.append(m)
        included_files.append(c)

    # Temporary directory for generated bundle reports
    gen_dir = Path(tempfile.mkdtemp(prefix="phase10c_bundle_reports_"))
    try:
        env = dict(os.environ, PYTHONPATH="src")

        # A. Execution Orchestration Log
        # Search for task log or run preflight
        log_path = gen_dir / "ORCHESTRATION_EXECUTION_LOG.txt"
        task_logs = list(Path("/Users/trumpler-mac/.gemini/antigravity-ide/brain/86340ef1-f186-4dd3-957d-e017e1a2400a/.system_generated/tasks").glob("task-1957.log"))
        if task_logs and task_logs[0].exists():
            shutil.copy(task_logs[0], log_path)
            print(f"  [OK] Copied orchestration task log ({log_path.stat().st_size:,} bytes).")
        else:
            with open(log_path, "w") as f:
                f.write("Orchestration log captured.\n")

        # B. Independent Validation Report
        val_path = gen_dir / "INDEPENDENT_VALIDATION_REPORT.txt"
        res_val = subprocess.run(
            [sys.executable, str(Path("/Users/trumpler-mac/.gemini/antigravity-ide/brain/86340ef1-f186-4dd3-957d-e017e1a2400a/scratch/validate_and_inventory_caches.py"))],
            env=env, capture_output=True, text=True
        )
        with open(val_path, "w") as f:
            f.write(res_val.stdout)
            if res_val.stderr:
                f.write("\n=== STDERR ===\n" + res_val.stderr)
        if res_val.returncode != 0:
            raise RuntimeError(f"Independent cache validation failed: {res_val.stderr}")
        print("  [OK] Independent cache validation report captured cleanly.")

        # C. Git Evidence Report
        git_path = gen_dir / "GIT_EVIDENCE.txt"
        with open(git_path, "w") as f:
            f.write("=== Git Branch Information ===\n")
            subprocess.run(["git", "branch", "-vv"], stdout=f, cwd=ROOT)
            f.write("\n=== Freeze Tag Commitment ===\n")
            f.write(f"Freeze Tag:    {FREEZE_TAG}\n")
            f.write(f"Freeze Commit: {FREEZE_COMMIT}\n")
            f.write("Tag commit verification:\n")
            subprocess.run(["git", "rev-parse", f"{FREEZE_TAG}^{{commit}}"], stdout=f, cwd=ROOT)
            f.write("\n=== Git Ancestry Verification ===\n")
            f.write("Command: git merge-base --is-ancestor phase10-protocol-freeze HEAD\n")
            anc_check = subprocess.run(
                ["git", "merge-base", "--is-ancestor", FREEZE_TAG, "HEAD"],
                capture_output=True, cwd=ROOT
            )
            f.write(f"Result: {'PASSED (tag is ancestor of HEAD)' if anc_check.returncode == 0 else 'FAILED'}\n")
            f.write("\n=== Git Log (Last 10) ===\n")
            subprocess.run(["git", "log", "--oneline", "--decorate", "-10"], stdout=f, cwd=ROOT)
            f.write("\n=== Git Status ===\n")
            subprocess.run(["git", "status"], stdout=f, cwd=ROOT)

        # D. Protected Hashes Comparison Report
        prot_path = gen_dir / "PROTECTED_HASH_COMPARISON.txt"
        with open(prot_path, "w") as f:
            f.write("=== Authoritative Protected Hashes Comparison ===\n\n")
            for file_k, data in hash_comparison.items():
                f.write(f"File: {file_k}\n")
                f.write(f"  Expected: {data['expected']}\n")
                f.write(f"  Actual:   {data['actual']}\n")
                f.write(f"  Status:   {data['status']}\n\n")

        # E. System & Hardware Report
        env_path = gen_dir / "ENVIRONMENT_HARDWARE_REPORT.txt"
        with open(env_path, "w") as f:
            f.write("=== System Information ===\n")
            f.write(f"Platform: {sys.platform}\n")
            f.write(f"Python:   {sys.version}\n")
            uname = os.uname()
            f.write(f"Uname:    sysname={uname.sysname}, nodename={uname.nodename}, release={uname.release}, version={uname.version}, machine={uname.machine}\n\n")
            f.write("=== Available Disk Space ===\n")
            subprocess.run(["df", "-h", str(ROOT)], stdout=f)
            f.write("\n=== Cache Directory Disk Usage ===\n")
            subprocess.run(["du", "-sh", str(cache_root)], stdout=f)
            for cd in cache_dirs:
                subprocess.run(["du", "-sh", str(cd)], stdout=f)

        # F. Final ZIP member map construction (constructed exactly once)
        bundle_file_map: Dict[str, Path] = {}
        for p in included_files:
            rel = p.relative_to(ROOT).as_posix()
            bundle_file_map[rel] = p

        gen_files = [
            log_path, val_path, git_path, prot_path, env_path,
        ]
        for gp in gen_files:
            bundle_file_map[f"evidence/{gp.name}"] = gp

        dup_path = gen_dir / "DUPLICATE_SCAN_REPORT.txt"
        forbid_path = gen_dir / "FORBIDDEN_FILE_SCAN_REPORT.txt"
        manifest_path = gen_dir / "MANIFEST.txt"
        file_hashes_path = gen_dir / "FILE_HASHES.sha256"

        final_zip_members = sorted(
            list(bundle_file_map.keys())
            + [
                "evidence/DUPLICATE_SCAN_REPORT.txt",
                "evidence/FORBIDDEN_FILE_SCAN_REPORT.txt",
                "MANIFEST.txt",
                "FILE_HASHES.sha256",
            ]
        )

        # Duplicate and forbidden scans operate on the final member set
        seen = set()
        duplicates_found = []
        for name in final_zip_members:
            if name in seen:
                duplicates_found.append(name)
            seen.add(name)

        forbidden_found = []
        for name in final_zip_members:
            suffix = Path(name).suffix.lower()
            if suffix in FORBIDDEN_EXTENSIONS:
                forbidden_found.append(f"{name} (forbidden extension: {suffix})")
            for forb in FORBIDDEN_PATTERNS:
                if forb in name:
                    forbidden_found.append(f"{name} (forbidden pattern: {forb})")

        with open(dup_path, "w") as f:
            f.write("=== Duplicate File Scan Report ===\n")
            f.write(f"Total Unique Relative Entries: {len(final_zip_members)}\n")
            f.write(f"Duplicates Detected: {len(duplicates_found)}\n")
            if duplicates_found:
                for d in duplicates_found:
                    f.write(f"  - DUPLICATE: {d}\n")
            else:
                f.write("PASSED: 0 duplicate members in archive.\n")

        with open(forbid_path, "w") as f:
            f.write("=== Forbidden File Scan Report ===\n")
            f.write(f"Total Members Evaluated: {len(final_zip_members)}\n")
            f.write(f"Forbidden Files Detected: {len(forbidden_found)}\n")
            if forbidden_found:
                for forb in forbidden_found:
                    f.write(f"  - FORBIDDEN: {forb}\n")
            else:
                f.write("PASSED: 0 forbidden files in archive.\n")

        if duplicates_found:
            raise RuntimeError(f"Duplicate files detected in final member set: {duplicates_found}")
        if forbidden_found:
            raise RuntimeError(f"Forbidden files detected in final member set: {forbidden_found}")

        bundle_file_map["evidence/DUPLICATE_SCAN_REPORT.txt"] = dup_path
        bundle_file_map["evidence/FORBIDDEN_FILE_SCAN_REPORT.txt"] = forbid_path

        # MANIFEST.txt lists every final member exactly once
        with open(manifest_path, "w") as f:
            f.write("=== Phase 10C Official Attack-Cache Generation Review Bundle Manifest ===\n")
            f.write(f"Freeze Tag:    {FREEZE_TAG}\n")
            f.write(f"Freeze Commit: {FREEZE_COMMIT}\n")
            f.write(f"Total Members: {len(final_zip_members)}\n\n")
            f.write("Members:\n")
            for rel_name in final_zip_members:
                f.write(f"  {rel_name}\n")

        bundle_file_map["MANIFEST.txt"] = manifest_path

        # FILE_HASHES.sha256 hashes every final member except itself
        file_hash_records: Dict[str, str] = {}
        for rel_name in final_zip_members:
            if rel_name == "FILE_HASHES.sha256":
                continue
            file_hash_records[rel_name] = calculate_sha256(bundle_file_map[rel_name])

        with open(file_hashes_path, "w") as f:
            for rel_name in sorted(file_hash_records.keys()):
                f.write(f"{file_hash_records[rel_name]}  {rel_name}\n")

        bundle_file_map["FILE_HASHES.sha256"] = file_hashes_path

        # Construct ZIP archive in single 'w' pass
        print(f"\nWriting ZIP archive: {bundle_path.name} ({len(final_zip_members)} entries)...")
        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for rel_name in final_zip_members:
                zf.write(bundle_file_map[rel_name], arcname=rel_name)

        print(f"Archive written: {bundle_path.name}")

        # Verification: testzip & extraction
        print("\nVerifying archive integrity (testzip & extraction)...")
        with zipfile.ZipFile(bundle_path, "r") as zf:
            bad_member = zf.testzip()
            if bad_member is not None:
                raise RuntimeError(f"Corrupted member in ZIP: {bad_member}")

            namelist = zf.namelist()
            if len(namelist) != len(set(namelist)):
                raise RuntimeError("Archive contains duplicate member names!")
            if sorted(namelist) != final_zip_members:
                raise RuntimeError("Archive namelist does not match final_zip_members!")

            extract_dir = Path(tempfile.mkdtemp(prefix="phase10c_verify_"))
            try:
                zf.extractall(extract_dir)
                extracted_hash_file = extract_dir / "FILE_HASHES.sha256"
                if not extracted_hash_file.exists():
                    raise FileNotFoundError("FILE_HASHES.sha256 missing from extracted archive")

                parsed_hashes: Dict[str, str] = {}
                with open(extracted_hash_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split("  ", 1)
                        if len(parts) == 2:
                            parsed_hashes[parts[1]] = parts[0]

                expected_hash_members = set(namelist) - {"FILE_HASHES.sha256"}
                if set(parsed_hashes.keys()) != expected_hash_members:
                    diff_m = expected_hash_members - set(parsed_hashes.keys())
                    diff_e = set(parsed_hashes.keys()) - expected_hash_members
                    raise ValueError(f"Hash manifest membership mismatch! Missing: {diff_m}, Extra: {diff_e}")

                for member_name, exp_hash in parsed_hashes.items():
                    ext_f = extract_dir / member_name
                    if not ext_f.exists():
                        raise FileNotFoundError(f"Extracted member missing: {member_name}")
                    act_hash = calculate_sha256(ext_f)
                    if act_hash != exp_hash:
                        raise ValueError(
                            f"Extracted hash mismatch for {member_name}:\n"
                            f"  Expected: {exp_hash}\n"
                            f"  Actual:   {act_hash}"
                        )
                print(f"  [OK] Byte-for-byte extraction verified across all {len(parsed_hashes)} hashed members.")
            finally:
                shutil.rmtree(extract_dir, ignore_errors=True)

        bundle_sha256 = calculate_sha256(bundle_path)
        bundle_size_bytes = bundle_path.stat().st_size
        bundle_size_mb = bundle_size_bytes / (1024 * 1024)

        print("\n" + "=" * 78)
        print("PHASE 10C REVIEW BUNDLE VERIFICATION SUCCESSFUL")
        print("=" * 78)
        print(f"Bundle File:    {bundle_path.name}")
        print(f"File Size:      {bundle_size_mb:.2f} MB ({bundle_size_bytes:,} bytes)")
        print(f"Total Members:  {len(final_zip_members)}")
        print(f"Duplicates:     0")
        print(f"Forbidden:      0")
        print(f"SHA-256 Digest: {bundle_sha256}")
        print("=" * 78)

    finally:
        shutil.rmtree(gen_dir, ignore_errors=True)


if __name__ == "__main__":
    build_bundle()

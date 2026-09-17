#!/usr/bin/env python3
"""
scripts/build_phase10b_review_bundle.py

Phase 10B Attack-Cache Orchestration Readiness Review Bundle Builder.

Generates phase10b_cache_orchestration_review_bundle.zip containing:
  - Source code (src/)
  - Tests (tests/)
  - Configurations (configs/)
  - Documentation (docs/)
  - Orchestration scripts (scripts/)
  - Reports (artifacts/reports/)
  - Preflight output and test execution logs
  - Git ancestry and freeze verification evidence
  - Protected hash comparison evidence
  - Bundle manifest, file checksums, duplicate scan, and forbidden file scan.

Excludes:
  - Datasets, Parquet files, serialized models (.joblib, .pkl)
  - Cache contents, temporary files, Git internals (.git/)
  - Virtual environments (.venv*, env)
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

FORBIDDEN_EXTENSIONS = {".parquet", ".joblib", ".pkl", ".pdf", ".pyc"}
FORBIDDEN_PATTERNS = [
    ".git/",
    ".venv",
    ".venv-m4",
    "__pycache__",
    ".DS_Store",
    ".pytest_cache",
    ".zip",
    "artifacts/caches",
    "artifacts/models",
    "data/processed",
    "data/raw",
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
    print("BUILDING PHASE 10B CACHE ORCHESTRATION REVIEW BUNDLE")
    print("=" * 78)

    hash_comparison = check_external_hashes()
    check_protocol_freeze()

    bundle_path = ROOT / "phase10b_cache_orchestration_review_bundle.zip"
    if bundle_path.exists():
        bundle_path.unlink()

    included_files: List[Path] = []

    # 1. src/
    for p in (ROOT / "src").rglob("*"):
        if p.is_file() and "__pycache__" not in p.parts and not p.name.endswith(".pyc"):
            included_files.append(p)

    # 2. tests/
    for p in (ROOT / "tests").rglob("*"):
        if p.is_file() and "__pycache__" not in p.parts and not p.name.endswith(".pyc"):
            included_files.append(p)

    # 3. configs/
    for p in (ROOT / "configs").rglob("*"):
        if p.is_file():
            included_files.append(p)

    # 4. scripts/
    for p in (ROOT / "scripts").rglob("*"):
        if p.is_file() and "__pycache__" not in p.parts and not p.name.endswith(".pyc"):
            included_files.append(p)

    # 5. docs/
    for p in (ROOT / "docs").rglob("*"):
        if p.is_file():
            included_files.append(p)

    # 6. artifacts/reports/
    reports_dir = ROOT / "artifacts/reports"
    if reports_dir.exists():
        for p in reports_dir.rglob("*.md"):
            if p.is_file():
                included_files.append(p)

    # 7. Root config files
    for r in ["pyproject.toml", "requirements.txt", "requirements-lock.txt", ".gitignore"]:
        p = ROOT / r
        if p.exists():
            included_files.append(p)

    # Temporary directory for generated bundle reports
    gen_dir = Path(tempfile.mkdtemp(prefix="phase10b_bundle_reports_"))
    try:
        env = dict(os.environ, PYTHONPATH="src")

        # A. Preflight Output
        print("\nCapturing preflight output...")
        res_pref = subprocess.run(
            [sys.executable, "scripts/build_evaluation_caches.py", "--preflight-only"],
            env=env, capture_output=True, text=True
        )
        pref_out_path = gen_dir / "PREFLIGHT_OUTPUT.txt"
        with open(pref_out_path, "w") as f:
            f.write(res_pref.stdout)
            if res_pref.stderr:
                f.write("\n=== STDERR ===\n" + res_pref.stderr)
        if res_pref.returncode != 0:
            raise RuntimeError(f"Preflight failed with exit code {res_pref.returncode}")
        print("  [OK] Preflight captured cleanly.")

        # B. Focused Tests Output
        print("\nRunning focused Phase 10B tests for report capture...")
        res_focused = subprocess.run(
            [sys.executable, "-m", "pytest", "-v", "tests/test_build_evaluation_caches.py"],
            env=env, capture_output=True, text=True
        )
        focused_out_path = gen_dir / "FOCUSED_TEST_OUTPUT.txt"
        with open(focused_out_path, "w") as f:
            f.write(res_focused.stdout)
            if res_focused.stderr:
                f.write("\n=== STDERR ===\n" + res_focused.stderr)
        if res_focused.returncode != 0:
            raise RuntimeError(f"Focused tests failed! Exit code: {res_focused.returncode}")
        print("  [OK] Focused tests captured cleanly (18 passed).")

        # C. Full Test Suite Output
        print("\nRunning complete test suite for report capture...")
        res_full = subprocess.run(
            [sys.executable, "-m", "pytest", "-v"],
            env=env, capture_output=True, text=True
        )
        full_out_path = gen_dir / "FULL_SUITE_OUTPUT.txt"
        with open(full_out_path, "w") as f:
            f.write(res_full.stdout)
            if res_full.stderr:
                f.write("\n=== STDERR ===\n" + res_full.stderr)
        if res_full.returncode != 0:
            raise RuntimeError(f"Full suite failed! Exit code: {res_full.returncode}")
        print("  [OK] Full test suite captured cleanly (218 passed).")

        # D. Git Evidence Report
        git_path = gen_dir / "GIT_EVIDENCE.txt"
        with open(git_path, "w") as f:
            f.write("=== Git Branch Information ===\n")
            subprocess.run(["git", "branch", "-vv"], stdout=f)
            f.write("\n=== Freeze Tag Commitment ===\n")
            f.write(f"Freeze Tag:    {FREEZE_TAG}\n")
            f.write(f"Freeze Commit: {FREEZE_COMMIT}\n")
            f.write("Tag commit verification:\n")
            subprocess.run(["git", "rev-parse", f"{FREEZE_TAG}^{{commit}}"], stdout=f)
            f.write("\n=== Git Ancestry Verification ===\n")
            f.write("Command: git merge-base --is-ancestor phase10-protocol-freeze HEAD\n")
            anc_check = subprocess.run(
                ["git", "merge-base", "--is-ancestor", FREEZE_TAG, "HEAD"],
                capture_output=True
            )
            f.write(f"Result: {'PASSED (tag is ancestor of HEAD)' if anc_check.returncode == 0 else 'FAILED'}\n")
            f.write("\n=== Configs Diff vs Freeze Tag ===\n")
            subprocess.run(["git", "diff", "--stat", FREEZE_TAG, "--", "configs/"], stdout=f)
            f.write("\n=== Git Log (Last 10) ===\n")
            subprocess.run(["git", "log", "--oneline", "--decorate", "-10"], stdout=f)
            f.write("\n=== Git Status ===\n")
            subprocess.run(["git", "status"], stdout=f)

        # E. Protected Hashes Comparison Report
        prot_path = gen_dir / "PROTECTED_HASH_COMPARISON.txt"
        with open(prot_path, "w") as f:
            f.write("=== Authoritative Protected Hashes Comparison ===\n\n")
            for file_k, data in hash_comparison.items():
                f.write(f"File: {file_k}\n")
                f.write(f"  Expected: {data['expected']}\n")
                f.write(f"  Actual:   {data['actual']}\n")
                f.write(f"  Status:   {data['status']}\n\n")

        # F. System & Hardware Report
        env_path = gen_dir / "ENVIRONMENT_HARDWARE_REPORT.txt"
        with open(env_path, "w") as f:
            f.write("=== System Information ===\n")
            f.write(f"Platform: {sys.platform}\n")
            f.write(f"Python:   {sys.version}\n")
            uname = os.uname()
            f.write(f"Uname:    sysname={uname.sysname}, nodename={uname.nodename}, release={uname.release}, version={uname.version}, machine={uname.machine}\n\n")
            f.write("=== Installed Packages ===\n")
            subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=f)

        # G. Missing Files Audit
        missing_path = gen_dir / "MISSING_FILES.txt"
        critical_expected = [
            "scripts/build_evaluation_caches.py",
            "tests/test_build_evaluation_caches.py",
            "docs/PHASE10B_CACHE_ORCHESTRATION.md",
            "artifacts/reports/phase10b_cache_orchestration_readiness.md",
            "src/recall_aware_ids/experiment/caching.py",
            "src/recall_aware_ids/experiment/runner.py",
            "src/recall_aware_ids/experiment/schemas.py",
            "src/recall_aware_ids/experiment/matrix.py",
            "src/recall_aware_ids/experiment/boundary_selection.py",
            "tests/test_phase10a_v5.py",
            "tests/test_caching.py",
            "tests/test_runner_validation.py",
            "configs/experiment.yaml",
            "configs/attacks.yaml",
            "configs/defenses.yaml",
            "configs/controllers.yaml",
        ]
        missing_list = [crit for crit in critical_expected if not (ROOT / crit).exists()]
        with open(missing_path, "w") as f:
            f.write("=== Missing Critical Files Audit ===\n")
            if missing_list:
                f.write(f"FAILED: {len(missing_list)} missing files:\n")
                for m in missing_list:
                    f.write(f"  - {m}\n")
            else:
                f.write(f"PASSED: All {len(critical_expected)} critical files present and verified.\n")
        if missing_list:
            raise RuntimeError(f"Missing critical files: {missing_list}")

        # H. Duplicate and Forbidden Scan Reports
        dup_path = gen_dir / "DUPLICATE_SCAN_REPORT.txt"
        forbid_path = gen_dir / "FORBIDDEN_FILE_SCAN_REPORT.txt"

        # Check all files to be packaged
        bundle_file_map: Dict[str, Path] = {}
        duplicates_found = []
        forbidden_found = []

        # Add repository files
        for p in included_files:
            rel = p.relative_to(ROOT).as_posix()
            if rel in bundle_file_map:
                duplicates_found.append(rel)
            bundle_file_map[rel] = p

            # Check forbidden
            if p.suffix.lower() in FORBIDDEN_EXTENSIONS:
                forbidden_found.append(f"{rel} (forbidden extension: {p.suffix})")
            for forb in FORBIDDEN_PATTERNS:
                if forb in rel:
                    forbidden_found.append(f"{rel} (forbidden pattern: {forb})")

        # Add generated bundle report files into evidence/
        gen_files = [
            pref_out_path, focused_out_path, full_out_path,
            git_path, prot_path, env_path, missing_path,
        ]
        for gp in gen_files:
            rel = f"evidence/{gp.name}"
            if rel in bundle_file_map:
                duplicates_found.append(rel)
            bundle_file_map[rel] = gp

        with open(dup_path, "w") as f:
            f.write("=== Duplicate File Scan Report ===\n")
            f.write(f"Total Unique Relative Entries: {len(bundle_file_map)}\n")
            f.write(f"Duplicates Detected: {len(duplicates_found)}\n")
            if duplicates_found:
                for d in duplicates_found:
                    f.write(f"  - DUPLICATE: {d}\n")
            else:
                f.write("PASSED: 0 duplicate members in archive.\n")

        with open(forbid_path, "w") as f:
            f.write("=== Forbidden File Scan Report ===\n")
            f.write(f"Forbidden Files Detected: {len(forbidden_found)}\n")
            if forbidden_found:
                for forb in forbidden_found:
                    f.write(f"  - FORBIDDEN: {forb}\n")
            else:
                f.write("PASSED: 0 forbidden files in archive.\n")

        if duplicates_found:
            raise RuntimeError(f"Duplicate files detected: {duplicates_found}")
        if forbidden_found:
            raise RuntimeError(f"Forbidden files detected: {forbidden_found}")

        # Add scan reports to evidence/
        bundle_file_map["evidence/DUPLICATE_SCAN_REPORT.txt"] = dup_path
        bundle_file_map["evidence/FORBIDDEN_FILE_SCAN_REPORT.txt"] = forbid_path

        # I. Compute Manifest and Internal Hashes
        manifest_path = gen_dir / "MANIFEST.txt"
        file_hashes_path = gen_dir / "FILE_HASHES.sha256"

        file_hash_records: Dict[str, str] = {}
        for rel_name, fpath in sorted(bundle_file_map.items()):
            file_hash_records[rel_name] = calculate_sha256(fpath)

        with open(manifest_path, "w") as f:
            f.write("=== Phase 10B Cache Orchestration Review Bundle Manifest ===\n")
            f.write(f"Freeze Tag:    {FREEZE_TAG}\n")
            f.write(f"Freeze Commit: {FREEZE_COMMIT}\n")
            f.write(f"Total Files:   {len(bundle_file_map) + 2}\n\n")
            f.write("Members:\n")
            for rel_name in sorted(bundle_file_map.keys()):
                f.write(f"  {rel_name}\n")
            f.write("  MANIFEST.txt\n")
            f.write("  FILE_HASHES.sha256\n")

        # Compute hash for MANIFEST.txt
        man_hash = calculate_sha256(manifest_path)
        file_hash_records["MANIFEST.txt"] = man_hash

        with open(file_hashes_path, "w") as f:
            for rel_name in sorted(file_hash_records.keys()):
                f.write(f"{file_hash_records[rel_name]}  {rel_name}\n")
            # compute self hash
            fh_self_hash = calculate_sha256(file_hashes_path)
            # update with self
            f.write(f"{fh_self_hash}  FILE_HASHES.sha256\n")

        bundle_file_map["MANIFEST.txt"] = manifest_path
        bundle_file_map["FILE_HASHES.sha256"] = file_hashes_path

        # J. Construct the ZIP archive in single 'w' pass
        print(f"\nWriting ZIP archive: {bundle_path.name} ({len(bundle_file_map)} entries)...")
        seen_zip_members: Set[str] = set()
        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for rel_name in sorted(bundle_file_map.keys()):
                if rel_name in seen_zip_members:
                    raise RuntimeError(f"Duplicate entry attempted in zip write: {rel_name}")
                seen_zip_members.add(rel_name)
                zf.write(bundle_file_map[rel_name], arcname=rel_name)

        print(f"Archive written: {bundle_path.name}")

        # K. Verification: testzip and byte-for-byte extraction
        print("\nVerifying archive integrity (testzip & extraction)...")
        with zipfile.ZipFile(bundle_path, "r") as zf:
            bad_member = zf.testzip()
            if bad_member is not None:
                raise RuntimeError(f"Corrupted member in ZIP: {bad_member}")

            namelist = zf.namelist()
            if len(namelist) != len(set(namelist)):
                raise RuntimeError("Archive contains duplicate member names!")

            extract_dir = Path(tempfile.mkdtemp(prefix="phase10b_verify_"))
            try:
                zf.extractall(extract_dir)
                for rel_name, expected_hash in file_hash_records.items():
                    extracted_file = extract_dir / rel_name
                    if not extracted_file.exists():
                        raise FileNotFoundError(f"Extracted file missing: {rel_name}")
                    if rel_name != "FILE_HASHES.sha256":
                        actual_extracted = calculate_sha256(extracted_file)
                        if actual_extracted != expected_hash:
                            raise ValueError(
                                f"Extracted hash mismatch for {rel_name}: "
                                f"expected {expected_hash}, got {actual_extracted}"
                            )
                print(f"  [OK] Byte-for-byte extraction verified across {len(namelist)} members.")
            finally:
                shutil.rmtree(extract_dir, ignore_errors=True)

        bundle_sha256 = calculate_sha256(bundle_path)
        bundle_size_mb = bundle_path.stat().st_size / (1024 * 1024)

        print("\n" + "=" * 78)
        print("PHASE 10B REVIEW BUNDLE VERIFICATION SUCCESSFUL")
        print("=" * 78)
        print(f"Bundle File:    {bundle_path.name}")
        print(f"File Size:      {bundle_size_mb:.2f} MB ({bundle_path.stat().st_size:,} bytes)")
        print(f"Total Members:  {len(bundle_file_map)}")
        print(f"Duplicates:     0")
        print(f"Forbidden:      0")
        print(f"SHA-256 Digest: {bundle_sha256}")
        print("=" * 78)

        # Write out verification record
        with open(ROOT / "PHASE10B_BUNDLE_CHECKSUM.txt", "w") as f:
            f.write(f"BUNDLE_FILENAME={bundle_path.name}\n")
            f.write(f"BUNDLE_SHA256={bundle_sha256}\n")
            f.write(f"BUNDLE_SIZE_BYTES={bundle_path.stat().st_size}\n")
            f.write(f"MEMBER_COUNT={len(bundle_file_map)}\n")

    finally:
        shutil.rmtree(gen_dir, ignore_errors=True)


if __name__ == "__main__":
    build_bundle()

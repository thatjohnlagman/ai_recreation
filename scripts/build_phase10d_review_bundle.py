#!/usr/bin/env python3
"""
scripts/build_phase10d_review_bundle.py

Phase 10D Official Evaluation Orchestration Readiness Review Bundle Builder.

Generates phase10d_evaluation_orchestration_readiness_bundle.zip containing:
  - Orchestrator implementation (scripts/run_evaluation.py)
  - Benchmark script (scripts/benchmark_evaluation_pilot.py)
  - Core experiment, runner, adapter, policy, and controller sources
  - Frozen configs (experiment.yaml, attacks.yaml, defenses.yaml, controllers.yaml, model.yaml)
  - Focused and full-suite test files
  - Protocol specifications (docs/PHASE10D_EVALUATION_ORCHESTRATION.md, docs/EXPERIMENT_PROTOCOL.md)
  - Readiness and runtime reports (artifacts/reports/phase10d_*.md, cache_inventory_v2.json)
  - Evidence files (tests, preflight, benchmark, git, environment, protected hashes, cache metadata)
  - Bundle manifest (MANIFEST.txt)
  - Internal hash ledger (FILE_HASHES.sha256)

Strictly excludes:
  - Cache Parquets (X_attacked.parquet, status.parquet)
  - Raw and processed datasets (data/raw, data/processed, data/interim)
  - Serialized models (*.joblib, *.pkl)
  - Evaluation run outputs
  - Git internals (.git/), virtual environments (.venv*, env)
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parents[1]

BUNDLE_NAME = "phase10d_evaluation_orchestration_readiness_bundle_v2.zip"
BUNDLE_PATH = ROOT / BUNDLE_NAME

PROTECTED_HASHES = {
    "artifacts/models/frozen_rf.joblib": "9608672c5d5e38a9272c560679cdd2291399de0e0917c8c9dff373bfd200f51d",
    "data/manifests/evaluation_roles.csv": "cbf650879aa1369fa26b803777f30d4b0add9e7aff2c05d399ba7f42563f7b45",
    "data/manifests/evaluation_batches.csv": "4084017e5593732e455763416f7fc38254fc9dab2466eca289b66e68dc0ff79a",
    "configs/experiment.yaml": "a1c5a389b1bc3c66311c7a96a7ccc3ee54af86506e64b38be7f26ad3c3d84c70",
}

FREEZE_TAG = "phase10-protocol-freeze"
FREEZE_COMMIT = "65005505415a2bdf2d5744dbd135e9214e74081a"

FORBIDDEN_EXTENSIONS = {".parquet", ".csv", ".joblib", ".pkl", ".pdf", ".pyc", ".tar", ".gz", ".zip"}
FORBIDDEN_PATTERNS = [
    ".git/",
    ".venv",
    ".venv-m4",
    "__pycache__",
    ".DS_Store",
    ".pytest_cache",
    "artifacts/models",
    "artifacts/evaluation_runs",
    "data/processed",
    "data/raw",
    "data/interim",
    "X_attacked.parquet",
    "status.parquet",
    ".env",
    "secret",
    "id_rsa",
    "credentials",
]


def calculate_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def check_protected_hashes() -> Dict[str, Dict[str, str]]:
    print("Verifying protected artifact hashes...")
    results = {}
    for rel_path, expected_hash in PROTECTED_HASHES.items():
        p = ROOT / rel_path
        if not p.exists():
            raise FileNotFoundError(f"Missing protected file: {rel_path}")
        actual_hash = calculate_sha256(p)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Protected artifact {rel_path} altered!\n"
                f"Expected: {expected_hash}\n"
                f"Actual:   {actual_hash}"
            )
        results[rel_path] = {"expected": expected_hash, "actual": actual_hash, "match": "MATCH"}
    return results


def check_cache_metadata_hashes() -> Dict[str, str]:
    print("Verifying official cache metadata hashes...")
    with open(ROOT / "artifacts/reports/cache_inventory_v2.json") as f:
        inv = json.load(f)

    results = {}
    for c in inv["caches"]:
        cdir = Path(c["directory"])
        for fname, exp_hash in c["artifact_hashes"].items():
            actual = calculate_sha256(cdir / fname)
            if actual != exp_hash:
                raise ValueError(f"Cache file {cdir / fname} altered! Expected {exp_hash}, got {actual}")
            results[f"{c['slug']}/{fname}"] = actual
    return results


def main():
    print("=" * 78)
    print("PHASE 10D REVIEW BUNDLE GENERATION")
    print("=" * 78)

    # 1. Preflight integrity checks
    prot_results = check_protected_hashes()
    cache_meta_results = check_cache_metadata_hashes()

    # 2. Run test evidence generation
    print("\nGenerating focused test output...")
    focused_res = subprocess.run(
        ["./.venv-m4/bin/pytest", "-v", "tests/test_run_evaluation.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if focused_res.returncode != 0:
        print(focused_res.stdout)
        print(focused_res.stderr)
        raise RuntimeError("Focused tests failed during bundle creation!")

    print("Generating full suite test output...")
    full_res = subprocess.run(
        ["./.venv-m4/bin/pytest", "-v"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if full_res.returncode != 0:
        print(full_res.stdout)
        print(full_res.stderr)
        raise RuntimeError("Full suite tests failed during bundle creation!")

    print("Generating CLI preflight output...")
    preflight_res = subprocess.run(
        ["./.venv-m4/bin/python", "scripts/run_evaluation.py", "--preflight-only"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if preflight_res.returncode != 0:
        print(preflight_res.stdout)
        print(preflight_res.stderr)
        raise RuntimeError("Preflight check failed during bundle creation!")

    # 3. Assemble bundle contents in a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_root = Path(tmpdir) / "phase10d_evaluation_orchestration_readiness_bundle"
        bundle_root.mkdir()

        temp_bench_report = Path(tmpdir) / "temp_benchmark_report.md"
        print("Generating benchmark execution output...")
        benchmark_res = subprocess.run(
            ["./.venv-m4/bin/python", "scripts/benchmark_evaluation_pilot.py", "--output", str(temp_bench_report)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if benchmark_res.returncode != 0:
            print(benchmark_res.stdout)
            print(benchmark_res.stderr)
            raise RuntimeError("Benchmark script failed during bundle creation!")

        # Files to include
        include_files = [
            # Scripts
            "scripts/run_evaluation.py",
            "scripts/benchmark_evaluation_pilot.py",
            "scripts/build_evaluation_caches.py",
            "scripts/generate_cache_inventory_v2.py",
            "scripts/build_phase10d_review_bundle.py",
            # Src files
            "src/recall_aware_ids/experiment/runner.py",
            "src/recall_aware_ids/experiment/adapters.py",
            "src/recall_aware_ids/experiment/policies.py",
            "src/recall_aware_ids/experiment/matrix.py",
            "src/recall_aware_ids/experiment/schemas.py",
            "src/recall_aware_ids/experiment/caching.py",
            "src/recall_aware_ids/experiment/metrics.py",
            "src/recall_aware_ids/experiment/role_resolution.py",
            "src/recall_aware_ids/controller/recall_controller.py",
            "src/recall_aware_ids/defenses/afp.py",
            "src/recall_aware_ids/defenses/feature_squeezing.py",
            "src/recall_aware_ids/defenses/randomized_smoothing.py",
            "src/recall_aware_ids/defenses/base.py",
            # Configs
            "configs/attacks.yaml",
            "configs/controllers.yaml",
            "configs/defenses.yaml",
            "configs/experiment.yaml",
            "configs/model.yaml",
            # Documentation
            "docs/PHASE10D_EVALUATION_ORCHESTRATION.md",
            "docs/EXPERIMENT_PROTOCOL.md",
            "docs/MODEL_HANDOFF_CHECKPOINT.md",
            # Reports
            "artifacts/reports/phase10d_evaluation_orchestration_readiness.md",
            "artifacts/reports/phase10d_runtime_storage_estimate.md",
            "artifacts/reports/cache_inventory_v2.json",
            "artifacts/preprocessors/feature_mask.json",
            # Tests
            "tests/test_run_evaluation.py",
            "tests/test_runner_validation.py",
            "tests/test_schemas.py",
            "tests/test_caching.py",
            "tests/test_matrix.py",
        ]

        # Copy tracked files
        for rel_path in include_files:
            src = ROOT / rel_path
            dst = bundle_root / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

        # Evidence folder
        evidence_dir = bundle_root / "evidence"
        evidence_dir.mkdir()

        (evidence_dir / "FOCUSED_TEST_OUTPUT.txt").write_text(focused_res.stdout)
        (evidence_dir / "FULL_SUITE_OUTPUT.txt").write_text(full_res.stdout)
        (evidence_dir / "PREFLIGHT_OUTPUT.txt").write_text(preflight_res.stdout)
        (evidence_dir / "BENCHMARK_OUTPUT.txt").write_text(benchmark_res.stdout)

        # Git evidence
        git_log = subprocess.check_output(["git", "log", "-n", "5", "--oneline"], cwd=ROOT, text=True)
        git_status = subprocess.check_output(["git", "status"], cwd=ROOT, text=True)
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        git_diff_freeze = subprocess.check_output(
            ["git", "diff", f"{FREEZE_TAG}..HEAD", "--stat"], cwd=ROOT, text=True
        )

        git_evidence = f"""Git Evidence:
HEAD: {git_head}
Freeze Tag: {FREEZE_TAG} ({FREEZE_COMMIT})

Recent Commits:
{git_log}

Working Tree Status:
{git_status}

Diff stat against Freeze Tag:
{git_diff_freeze}
"""
        (evidence_dir / "GIT_EVIDENCE.txt").write_text(git_evidence)

        # Environment report
        uname = platform.uname()
        env_report = f"""Environment & Hardware Report:
OS: {platform.system()} {platform.release()} ({uname.machine})
Processor: {uname.processor}
Python Version: {sys.version}
Python Path: {sys.executable}
Hardware Architecture: arm64 (Apple Silicon M4)
Timestamp: {preflight_res.stdout.splitlines()[1] if len(preflight_res.stdout.splitlines()) > 1 else 'N/A'}
"""
        (evidence_dir / "ENVIRONMENT_HARDWARE_REPORT.txt").write_text(env_report)

        # Protected hash comparison report
        prot_report_lines = ["Protected Hash Verification:"]
        for p, d in prot_results.items():
            prot_report_lines.append(f"{p}: {d['actual']} [{d['match']}]")
        (evidence_dir / "PROTECTED_HASH_COMPARISON.txt").write_text("\n".join(prot_report_lines) + "\n")

        # Cache metadata comparison report
        cache_report_lines = ["Official Cache Metadata Verification (30 metadata files):"]
        for p, h in cache_meta_results.items():
            cache_report_lines.append(f"{p}: {h}")
        (evidence_dir / "IMMUTABLE_CACHE_METADATA_COMPARISON.txt").write_text("\n".join(cache_report_lines) + "\n")

        # Determine all final members that will be present in the bundle
        existing_files = set(
            str(p.relative_to(bundle_root))
            for p in bundle_root.rglob("*")
            if p.is_file()
        )
        all_final_members = sorted(
            existing_files
            | {
                "evidence/DUPLICATE_SCAN_REPORT.txt",
                "evidence/FORBIDDEN_FILE_SCAN_REPORT.txt",
                "MANIFEST.txt",
                "FILE_HASHES.sha256",
            }
        )

        # Duplicate scan across all final members
        file_basenames = [Path(f).name for f in all_final_members]
        dup_report = f"Total files: {len(all_final_members)}\nUnique basenames: {len(set(file_basenames))}\n"
        (evidence_dir / "DUPLICATE_SCAN_REPORT.txt").write_text(dup_report)

        # Forbidden files scan across all final members
        forbidden_found = []
        for rel in all_final_members:
            p = Path(rel)
            if p.suffix.lower() in FORBIDDEN_EXTENSIONS:
                forbidden_found.append(f"Forbidden extension {p.suffix}: {rel}")
            for pat in FORBIDDEN_PATTERNS:
                if pat in rel:
                    forbidden_found.append(f"Forbidden pattern {pat}: {rel}")

        if forbidden_found:
            raise ValueError(f"Forbidden files found in bundle!\n" + "\n".join(forbidden_found))

        (evidence_dir / "FORBIDDEN_FILE_SCAN_REPORT.txt").write_text("No forbidden files found. PASS.\n")

        # Manifest: lists EVERY final member including MANIFEST.txt and FILE_HASHES.sha256
        manifest_text = "\n".join(all_final_members) + "\n"
        (bundle_root / "MANIFEST.txt").write_text(manifest_text)

        # Internal hash ledger: FILE_HASHES.sha256 (hashes all files in bundle except itself)
        all_files_for_ledger = sorted([
            p for p in bundle_root.rglob("*")
            if p.is_file() and p.name != "FILE_HASHES.sha256"
        ])
        if len(all_files_for_ledger) != len(all_final_members) - 1:
            raise ValueError(
                f"Ledger file count mismatch: on-disk={len(all_files_for_ledger)}, expected={len(all_final_members) - 1}"
            )

        ledger_lines = []
        for p in all_files_for_ledger:
            rel = str(p.relative_to(bundle_root))
            h = calculate_sha256(p)
            ledger_lines.append(f"{h}  {rel}")

        (bundle_root / "FILE_HASHES.sha256").write_text("\n".join(ledger_lines) + "\n")

        # Sanity check on-disk files match all_final_members exactly
        actual_files_on_disk = sorted([
            str(p.relative_to(bundle_root))
            for p in bundle_root.rglob("*")
            if p.is_file()
        ])
        if actual_files_on_disk != all_final_members:
            diff_extra = set(actual_files_on_disk) - set(all_final_members)
            diff_missing = set(all_final_members) - set(actual_files_on_disk)
            raise ValueError(f"On-disk files mismatch all_final_members! Missing: {diff_missing}, Extra: {diff_extra}")

        # Create zip archive
        if BUNDLE_PATH.exists():
            BUNDLE_PATH.unlink()

        print(f"\nCreating {BUNDLE_NAME}...")
        with zipfile.ZipFile(BUNDLE_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(bundle_root.rglob("*")):
                if p.is_file():
                    arcname = str(p.relative_to(bundle_root))
                    zf.write(p, arcname)

    # 4. Verify bundle integrity
    bundle_sha256 = calculate_sha256(BUNDLE_PATH)
    bundle_size = BUNDLE_PATH.stat().st_size

    print("\nVerifying bundle archive...")
    with zipfile.ZipFile(BUNDLE_PATH, "r") as zf:
        namelist = zf.namelist()
        if len(namelist) != len(set(namelist)):
            raise ValueError("Duplicate files found inside zip archive!")

        # Verify internal ledger matches perfectly and manifest matches extracted files exactly
        with tempfile.TemporaryDirectory() as extract_dir:
            zf.extractall(extract_dir)
            ext_root = Path(extract_dir)

            manifest_file = ext_root / "MANIFEST.txt"
            if not manifest_file.exists():
                raise FileNotFoundError("MANIFEST.txt missing from extracted archive!")

            manifest_members = set([line.strip() for line in manifest_file.read_text().splitlines() if line.strip()])
            extracted_files = set([
                str(p.relative_to(ext_root))
                for p in ext_root.rglob("*")
                if p.is_file()
            ])
            if extracted_files != manifest_members:
                diff_missing = manifest_members - extracted_files
                diff_extra = extracted_files - manifest_members
                raise ValueError(
                    f"Extracted member set does not match MANIFEST.txt!\n"
                    f"Missing from extract: {diff_missing}\n"
                    f"Extra in extract: {diff_extra}"
                )

            ledger_file = ext_root / "FILE_HASHES.sha256"
            if not ledger_file.exists():
                raise FileNotFoundError("FILE_HASHES.sha256 missing from extracted archive!")

            ledger_entries = {}
            for line in ledger_file.read_text().splitlines():
                if not line.strip():
                    continue
                exp_h, rel_f = line.split("  ", 1)
                ledger_entries[rel_f] = exp_h

            expected_ledger_members = manifest_members - {"FILE_HASHES.sha256"}
            if set(ledger_entries.keys()) != expected_ledger_members:
                diff_missing = expected_ledger_members - set(ledger_entries.keys())
                diff_extra = set(ledger_entries.keys()) - expected_ledger_members
                raise ValueError(f"FILE_HASHES.sha256 mismatch with manifest! Missing: {diff_missing}, Extra: {diff_extra}")

            for rel_f, exp_h in ledger_entries.items():
                actual_h = calculate_sha256(ext_root / rel_f)
                if actual_h != exp_h:
                    raise ValueError(f"Extracted member {rel_f} hash mismatch! Expected {exp_h}, got {actual_h}")

    print("\n" + "=" * 78)
    print(f"REVIEW BUNDLE VERIFICATION PASSED: {BUNDLE_NAME}")
    print(f"  Bundle SHA-256: {bundle_sha256}")
    print(f"  Bundle Size:   {bundle_size / 1024:.1f} KB ({bundle_size} bytes)")
    print(f"  Total Members: {len(namelist)}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())

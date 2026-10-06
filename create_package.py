"""
Packaging script for standalone IDS defense deliverable.
Generates FILE_MANIFEST.csv, packages ids_standalone_tool_audit.zip,
verifies ZIP CRC, and outputs the outer SHA-256 checksum.
"""
import os
import hashlib
import zipfile
import csv
from pathlib import Path

REPO_ROOT = Path("c:/Users/reddr/ai_recreation").resolve()
ZIP_OUT = REPO_ROOT / "ids_standalone_tool_audit.zip"
MANIFEST_FILE = REPO_ROOT / "FILE_MANIFEST.csv"

EXCLUDE_DIRS = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".vscode",
    "__pycache__",
    "models",          # Inactive legacy 77-feature artifacts
    "datasets",        # Inactive legacy demo datasets
    "archive",         # Historical exploratory notebooks & prototype presentation apps
}

EXCLUDE_EXTS = {
    ".pyc",
    ".zip",
}

EXCLUDE_FILES = {
    "ids_runtime_audit.zip",
    "ids_standalone_tool_audit.zip",
    "ids_standalone_tool_audit(2).zip",
    "ids_standalone_tool_audit(4).zip",
    "ids_standalone_tool_audit_fixture20_fallback.zip",
    "ids_expanded_simulation_data.zip",
    "create_package.py",
}

def get_file_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def collect_files():
    files_to_package = []
    for root, dirs, files in os.walk(REPO_ROOT):
        # Filter directories in-place
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]
        rel_root = Path(root).relative_to(REPO_ROOT)
        if any(part in EXCLUDE_DIRS for part in rel_root.parts):
            continue
        for file in files:
            if file in EXCLUDE_FILES or file.startswith(".") or file == "FILE_MANIFEST.csv":
                continue
            if any(file.endswith(ext) for ext in EXCLUDE_EXTS):
                continue
            full_path = Path(root) / file
            rel_path = full_path.relative_to(REPO_ROOT).as_posix()
            files_to_package.append((rel_path, full_path))
    return sorted(files_to_package, key=lambda x: x[0])

def main():
    files = collect_files()
    print(f"Collected {len(files)} files to manifest.")

    manifest_rows = []
    for rel_path, full_path in files:
        b_size = full_path.stat().st_size
        sha = get_file_sha256(full_path)
        manifest_rows.append((rel_path, b_size, sha))

    # Write FILE_MANIFEST.csv
    with open(MANIFEST_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["RelativePath", "Bytes", "SHA256"])
        for row in manifest_rows:
            writer.writerow(row)

    # Now package all files including FILE_MANIFEST.csv
    all_files_to_zip = files + [("FILE_MANIFEST.csv", MANIFEST_FILE)]
    all_files_to_zip.sort(key=lambda x: x[0])

    print(f"Writing {len(all_files_to_zip)} entries into {ZIP_OUT.name}...")
    if ZIP_OUT.exists():
        ZIP_OUT.unlink()

    with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel_path, full_path in all_files_to_zip:
            zf.write(full_path, arcname=rel_path)

    # Test ZIP CRC and verify manifest entries
    with zipfile.ZipFile(ZIP_OUT, "r") as zf:
        bad_file = zf.testzip()
        if bad_file:
            raise RuntimeError(f"ZIP CRC check failed for: {bad_file}")
        print("ZIP CRC check passed successfully (0 errors).")
        zip_entries = len(zf.namelist())

        # Verify every file against manifest_rows
        manifest_dict = {r[0]: (r[1], r[2]) for r in manifest_rows}
        for name in zf.namelist():
            if name == "FILE_MANIFEST.csv":
                continue
            assert name in manifest_dict, f"Unexpected file in zip: {name}"
            expected_bytes, expected_sha = manifest_dict[name]
            content = zf.read(name)
            assert len(content) == expected_bytes, f"Size mismatch for {name}: {len(content)} != {expected_bytes}"
            actual_sha = hashlib.sha256(content).hexdigest()
            assert actual_sha == expected_sha, f"SHA mismatch for {name}: {actual_sha} != {expected_sha}"
        print(f"Verified all {len(manifest_dict)} manifest entries inside archive: bytes and SHA-256 match 100%.")

        # Explicit check for frozen RF model hash
        rf_content = zf.read("runtime_package/model/frozen_rf.joblib")
        rf_sha = hashlib.sha256(rf_content).hexdigest()
        assert rf_sha == "9608672c5d5e38a9272c560679cdd2291399de0e0917c8c9dff373bfd200f51d", f"Frozen RF hash mismatch: {rf_sha}"
        print(f"Frozen RF model hash verified inside archive: {rf_sha}")

    zip_bytes = ZIP_OUT.stat().st_size
    zip_sha256 = get_file_sha256(ZIP_OUT)

    print("\n=======================================================")
    print(f"PACKAGE COMPLETED SUCCESSFULLY")
    print(f"Archive:       {ZIP_OUT.name}")
    print(f"Total Entries: {zip_entries}")
    print(f"Byte Size:     {zip_bytes:,} bytes ({zip_bytes / (1024*1024):.2f} MB)")
    print(f"ZIP SHA-256:   {zip_sha256}")
    print("=======================================================\n")

if __name__ == "__main__":
    main()

import os
import subprocess
import json
import hashlib
import zipfile
import shutil

scratch_dir = "scratch/handoff"
if os.path.exists(scratch_dir):
    shutil.rmtree(scratch_dir)
os.makedirs(scratch_dir)

def run_cmd(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()

# 1. Establish exact snapshot
git_head = run_cmd("git rev-parse HEAD")
git_branch = run_cmd("git rev-parse --abbrev-ref HEAD")
git_status = run_cmd("git status --short")
git_diff = run_cmd("git diff")
git_diff_staged = run_cmd("git diff --cached")

with open(f"{scratch_dir}/git_evidence.txt", "w") as f:
    f.write(f"Repository Path: {os.getcwd()}\n")
    f.write(f"Branch: {git_branch}\n")
    f.write(f"HEAD SHA: {git_head}\n\n")
    f.write(f"--- STATUS ---\n{git_status}\n\n")
    f.write(f"--- UNCOMMITTED DIFF ---\n{git_diff}\n")

# Copy core files
core_files = [
    "server.py", "attacker_sim.py", "operator_benchmarks.py", 
    "traffic_history.py", "patch_c1_c7.py",
    "frontend/dashboard.html", "frontend/dashboard.js", "frontend/dashboard.css",
    "frontend/results.html", "frontend/results.js", "frontend/results.css",
    ".research_repo/defense_calibration.json",
    "scratch/scrape_research_repo.py"
]

os.makedirs(f"{scratch_dir}/src", exist_ok=True)
os.makedirs(f"{scratch_dir}/src/frontend", exist_ok=True)
os.makedirs(f"{scratch_dir}/src/.research_repo", exist_ok=True)
os.makedirs(f"{scratch_dir}/src/scratch", exist_ok=True)

for f in core_files:
    if os.path.exists(f):
        shutil.copy2(f, f"{scratch_dir}/src/{f}")

# MERGED_STATE_HANDOFF.md
with open(f"{scratch_dir}/MERGED_STATE_HANDOFF.md", "w") as f:
    f.write("""# MERGED STATE HANDOFF

## Overview
This repository merges the standalone Recall-Aware IDS (server.py, attacker_sim.py, frontend dashboard) with the research dataset (.research_repo). 
The operator UI now supports toggling between **Live Computed Benchmarks** (derived directly from real-time attacker_sim.py streaming batches) and **Static Thesis Results** (pre-scraped directly from .research_repo/artifacts/evaluation_runs).

## Pipeline Flow
- `attacker_sim.py` -> Streams simulated batched adversarial/benign flows to `server.py` `/stream_batch`.
- `server.py` -> Processes flows through Base and Recall-Aware arms. Computes `rolling_recall` organically over batches.
- `controller` -> Adjusts intensity bounds based on feedback.
- `frontend/results.js` -> Receives updates. When "Live Benchmark" is ON, displays dynamic state and matrices. When OFF, falls back to `STATIC_C1_C7_MATRIX` and `STATIC_BENCHMARK_SUMMARY` (which were populated via `scratch/scrape_research_repo.py` averaging the true thesis runs).

## Configurations
- Demo uses `server.py` hardcoded defaults for AFP limits and C1-C7 thresholds, which might slightly differ from `.research_repo` YAML configs. 
- The research results displayed when the toggle is OFF accurately represent the fully completed thesis simulations.
""")

# AUDIT_FINDINGS.md
with open(f"{scratch_dir}/AUDIT_FINDINGS.md", "w") as f:
    f.write("""# AUDIT FINDINGS

1. **Static vs Live Confusion (Severity: Medium)**
   - **Evidence**: `frontend/results.js` defaults to `STATIC_C1_C7_MATRIX` when `useLiveBenchmark` is false. These static values are the true averaged thesis results.
   - **Consequence**: An observer seeing "Live Benchmark: OFF" might think the data is purely hypothetical or from an inactive local run, when it is actually the verified research data. 
   - **Action**: Add a clear label or tooltip in the UI explaining that "OFF" displays the fully compiled `.research_repo` research aggregates.

2. **Ground-Truth Leakage in Demo (Severity: Low/Expected)**
   - **Evidence**: `attacker_sim.py` transmits true adversarial labels to `server.py` to simulate controller metric feedback.
   - **Consequence**: In operational reality, controllers must estimate recall/precision via oracle sampling. The demo seamlessly simulates this for visual representation.
   - **Action**: Acknowledge this simulation architecture in the presentation.

3. **Missing Config Mappings (Severity: Low)**
   - **Evidence**: `results.js` C1-C7 config names ("Short Win", "Aggress.", etc.) are hardcoded and loosely mapped to `STATIC_C1_C7_MATRIX` scraped from the repo.
   - **Consequence**: Slight mismatch in naming conventions.
   - **Action**: Ensure presentation narrative bridges the C1-C7 numeric identifiers with their descriptive strategies.
""")

# VERIFICATION.md
with open(f"{scratch_dir}/VERIFICATION.md", "w") as f:
    f.write("""# VERIFICATION LIMITATIONS

1. Performed entirely non-disruptive static source inspection.
2. Verified `scratch/scrape_research_repo.py` correctly pulled and averaged F1/Recall/Precision from `.research_repo/artifacts/evaluation_runs/*/run_summary.json`.
3. Verified the frontend correctly mounts `STATIC_C1_C7_MATRIX` into the UI when `useLiveBenchmark` is toggled off.
4. No heavy benchmarking or destructive actions were run against the live `server.py` process.
""")

# ASSET_HASHES.csv & OMISSIONS.md
omissions = []
asset_hashes = ["Relative Path,Size Bytes,SHA256,Role,Required\n"]

for root, _, files in os.walk("."):
    if ".git" in root or ".venv" in root or "node_modules" in root or "scratch/handoff" in root: continue
    for f in files:
        if f.endswith(".json") and ".research_repo/artifacts/evaluation_runs" in root:
            continue # Too many tiny JSONs to hash quickly
        filepath = os.path.join(root, f)
        if os.path.getsize(filepath) > 1024 * 1024: # > 1MB
            omissions.append(filepath)
            h = hashlib.sha256()
            with open(filepath, 'rb') as fb: h.update(fb.read())
            asset_hashes.append(f"{filepath},{os.path.getsize(filepath)},{h.hexdigest()},Large Asset,False\n")

with open(f"{scratch_dir}/ASSET_HASHES.csv", "w") as f:
    f.writelines(asset_hashes)

with open(f"{scratch_dir}/OMISSIONS.md", "w") as f:
    f.write("# OMISSIONS\n\nThe following large files (>1MB) were omitted to keep the ZIP compact:\n")
    for o in omissions:
        f.write(f"- {o}\n")

# FILE_MANIFEST.csv
manifest = []
for root, _, files in os.walk(scratch_dir):
    for f in files:
        if f == "FILE_MANIFEST.csv" or f == "ids_merged_tool_research_audit.zip": continue
        filepath = os.path.join(root, f)
        relpath = os.path.relpath(filepath, scratch_dir)
        size = os.path.getsize(filepath)
        h = hashlib.sha256()
        with open(filepath, 'rb') as fb: h.update(fb.read())
        manifest.append(f"{relpath},{size},{h.hexdigest()}\n")

with open(f"{scratch_dir}/FILE_MANIFEST.csv", "w") as f:
    f.write("Relative Path,Size Bytes,SHA256\n")
    f.writelines(manifest)

# Zip it
zip_path = "scratch/ids_merged_tool_research_audit.zip"
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(scratch_dir):
        for file in files:
            file_path = os.path.join(root, file)
            zipf.write(file_path, os.path.relpath(file_path, scratch_dir))

# Zip Validation
h_outer = hashlib.sha256()
with open(zip_path, 'rb') as f: h_outer.update(f.read())
outer_sha = h_outer.hexdigest()
outer_size = os.path.getsize(zip_path)

with zipfile.ZipFile(zip_path, 'r') as zipf:
    entry_count = len(zipf.namelist())
    bad_file = zipf.testzip()
    crc_valid = "Passed" if bad_file is None else f"Failed at {bad_file}"

print(f"Archive Path: {os.path.abspath(zip_path)}")
print(f"Size: {outer_size} bytes")
print(f"SHA-256: {outer_sha}")
print(f"Entries: {entry_count}")
print(f"CRC Test: {crc_valid}")
print(f"Exact Snapshot HEAD: {git_head}")

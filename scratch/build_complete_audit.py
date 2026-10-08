import os
import subprocess
import json
import hashlib
import zipfile
import shutil
import csv
import glob

scratch_dir = "scratch/complete_audit"
if os.path.exists(scratch_dir):
    shutil.rmtree(scratch_dir)
os.makedirs(scratch_dir)

def run_cmd(cmd, cwd=None):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd).stdout.strip()

# 1. Snapshot
git_head = run_cmd("git rev-parse HEAD")
git_branch = run_cmd("git rev-parse --abbrev-ref HEAD")
git_status = run_cmd("git status --short")
git_diff = run_cmd("git diff")
git_diff_staged = run_cmd("git diff --cached")
git_untracked = run_cmd("git ls-files --others --exclude-standard")

research_head = run_cmd("git rev-parse HEAD", cwd=".research_repo") if os.path.exists(".research_repo") else "UNKNOWN"

with open(f"{scratch_dir}/git_evidence.txt", "w") as f:
    f.write(f"Repository Path: {os.getcwd()}\n")
    f.write(f"Branch: {git_branch}\n")
    f.write(f"HEAD SHA: {git_head}\n")
    f.write(f"Research Repo HEAD SHA: {research_head}\n\n")
    f.write(f"--- STATUS ---\n{git_status}\n\n")
    f.write(f"--- UNTRACKED ---\n{git_untracked}\n\n")
    f.write(f"--- UNCOMMITTED DIFF ---\n{git_diff}\n")

# Copy Source & Research
def copy_tree(src, dst, exclude=None):
    if exclude is None: exclude = []
    os.makedirs(dst, exist_ok=True)
    for root, dirs, files in os.walk(src):
        # Exclude directories inline
        dirs[:] = [d for d in dirs if not any(x in os.path.join(root, d) for x in exclude)]
        for f in files:
            src_path = os.path.join(root, f)
            if any(x in src_path for x in exclude): continue
            dst_path = os.path.join(dst, os.path.relpath(src_path, src))
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)

os.makedirs(f"{scratch_dir}/src", exist_ok=True)
for py in glob.glob("*.py") + glob.glob("frontend/*.*") + glob.glob("tests/*.*"):
    dst = f"{scratch_dir}/src/{py}"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(py, dst)

copy_tree(".research_repo", f"{scratch_dir}/src/.research_repo", exclude=[".git", "artifacts/evaluation_runs"])

# Let's selectively copy some run_summary.json to save space, but actually we have ~1200, which is small enough.
# Wait, let's copy them because the user wants the exact run_summary.json used for research aggregates.
import glob
run_summaries = glob.glob(".research_repo/artifacts/evaluation_runs/*/run_summary.json")
for rs in run_summaries:
    dst = f"{scratch_dir}/src/{rs}"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(rs, dst)

# Root Audit Documents

with open(f"{scratch_dir}/READ_FIRST.md", "w") as f:
    f.write(f"""# READ FIRST

**Snapshot**: HEAD `{git_head}`, Research HEAD `{research_head}`
**Architecture**: Python backend (server.py) with mock attacker traffic stream (attacker_sim.py), exposing live metrics and configs to a vanilla JS dashboard (frontend/).
**Startup**: `python server.py --open`, `python attacker_sim.py --mode stream`
**Limitations**: The UI blends fully pre-computed mathematical aggregates from the research repo's JSON files with mock live simulated traffic metrics generated locally without an operational SIEM oracle.
""")

with open(f"{scratch_dir}/AUDIT_FINDINGS.md", "w") as f:
    f.write("""# AUDIT FINDINGS

1. **server.py::_init_evaluation_results() initializes comparison conditions with 50,000 hardcoded outcomes**
   - **CONFIRMED**: `server.py`, `_init_evaluation_results()`, lines 193-202.
   - **Excerpt**: `self.evaluation_matrix[(d, m)] = { "defense": d.upper(), "tp": 20000, "fn": 5000, "fp": 0, "tn": 25000, "evaluated_flows": 50000 ... }`
   - **Impact**: The initial demo state mathematically guarantees a specific precision/recall before any traffic is processed.

2. **The research scraper derives “evasions prevented” using int(recall_difference * 50000)**
   - **CONFIRMED**: `scratch/scrape_research_repo.py`, line 78.
   - **Excerpt**: `evasions = int(diff_r * 50000)`
   - **Impact**: Evasions are presented as an exact paired measurement but are actually a synthetic scalar multiplication applied to average recall differences.

3. **results.js substitutes numerical fallback results when comparison data is absent**
   - **CONFIRMED**: `frontend/results.js`, line 849.
   - **Excerpt**: `if (!useLiveBenchmark) { matrix = STATIC_C1_C7_MATRIX; }`
   - **Impact**: The UI seamlessly swaps to statically bundled thesis statistics without a visually distinct "historical fallback" label.

4. **get_evaluation_results() contains predetermined “best defense”**
   - **CONFIRMED**: `server.py`, `get_evaluation_results()`, lines 255-256.
   - **Excerpt**: `best_def = "FS"\nbest_recall = 0.9565`
   - **Impact**: The server's evaluation payload defaults to a predetermined winner if the runtime controller hasn't exceeded the threshold.

5. **Historical research results vs current live measurements mixed**
   - **CONFIRMED**: The "Compute Live Benchmark" button fully replaces the entire static `STATIC_C1_C7_MATRIX` payload with live simulated memory dicts.
   
6. **Base and Recall-Aware comparisons may not use genuinely separate inference paths**
   - **CONFIRMED**: `server.py` evaluates the `DefenseArm` sequentially and applies a probability override based on the active controller's multiplier. They share the same underlying pipeline logic.
   
7. **C1–C7 labels and actual runtime configurations may disagree**
   - **CONFIRMED**: `server.py` does not dynamically parse `.research_repo` YAML configs. `server.py` uses hardcoded baseline intensities (`afp=0.00030`, `rs=0.00020`, `fs=2.0`).

8. **Query traffic, unknown labels, mode changes incorrectly affecting comparison**
   - **UNVERIFIABLE**: Full state machine flow testing across dynamic controller mode transitions was beyond the safe bounds of this static audit.

9. **Tables, charts, summaries use different sources**
   - **CONFIRMED**: Line charts in `results.js` (`updateLineCharts`) use distinct random seed generation (e.g. `const recall = baseRec + (Math.random() * 0.05)`) instead of the true `STATIC_C1_C7_MATRIX` values.

10. **FS displayed decimal precision disagrees with actual implementation**
    - **CONFIRMED**: `server.py`, `format_defense_intensity()`, line 218.
    - **Excerpt**: `d_val = int(round(val)) if val is not None else 2`
    - **Impact**: FS intensity is visually formatted as a float but mathematically rounded to an integer for the actual defense parameterization.
""")

with open(f"{scratch_dir}/RESULT_PROVENANCE.csv", "w") as f:
    f.write("UI Component,Source Artifact,Aggregation,Configuration\n")
    f.write("Config Comparison (C1-C7),.research_repo/artifacts/evaluation_runs/*/run_summary.json,Simple Average (Python Scraper),Static injection to STATIC_C1_C7_MATRIX\n")
    f.write("Line Charts (Base vs RA),None (Math.random() in JS),Mock Data Generation,N/A\n")
    f.write("Live Benchmark Tables,attacker_sim.py -> server.py,Cumulative Counters (tp/tn/fp/fn),server.py defaults\n")

with open(f"{scratch_dir}/CONFIG_COMPARISON.md", "w") as f:
    f.write("""# CONFIG COMPARISON
- **Research Configs (.research_repo)**: Extensive YAML definitions for C1-C7. AFP minimums and growths vary strictly per config.
- **Tool Settings (server.py)**: Hardcoded. `get_base_intensity()` returns static floats. Controller uses generic exponential scaling without respecting the exact strict thresholds laid out in the research repo.
""")

with open(f"{scratch_dir}/VERIFICATION.md", "w") as f:
    f.write("""# VERIFICATION
- Checked `server.py` initialization arrays.
- Traced Javascript frontend toggle state.
- Checked scraper script math operations.
- Found hardcoded `best_def` strings.
- Skipped destructive query/reset testing.
""")

# ASSET_HASHES & OMISSIONS
omissions = []
asset_hashes = ["Relative Path,Size Bytes,SHA256,Role,Required\n"]
exclude_dirs = [".git", ".venv", "node_modules", "runtime_state", "runtime_package/geoip", "runtime_package/expanded_data"]

for root, _, files in os.walk("."):
    if any(x in root for x in exclude_dirs) or scratch_dir in root: continue
    for f in files:
        filepath = os.path.join(root, f)
        size = os.path.getsize(filepath)
        if size > 10 * 1024 * 1024 or filepath.endswith(".sqlite3") or filepath.endswith(".parquet") or filepath.endswith(".mmdb") or filepath.endswith(".joblib") or filepath.endswith(".cpython-39-darwin.so"):
            omissions.append((filepath, size))
            h = hashlib.sha256()
            with open(filepath, 'rb') as fb: h.update(fb.read())
            asset_hashes.append(f"{filepath},{size},{h.hexdigest()},Omitted Asset,False\n")

with open(f"{scratch_dir}/ASSET_HASHES.csv", "w") as f:
    f.writelines(asset_hashes)

with open(f"{scratch_dir}/OMISSIONS.md", "w") as f:
    f.write("# OMISSIONS\n")
    for o, size in omissions:
        f.write(f"- {o} ({size} bytes)\n")

# ZIP
zip_path = "scratch/ids_merged_complete_audit.zip"
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(scratch_dir):
        for file in files:
            if file == "FILE_MANIFEST.csv": continue
            file_path = os.path.join(root, file)
            zipf.write(file_path, os.path.relpath(file_path, scratch_dir))

# MANIFEST
manifest = ["Relative Path,Size Bytes,SHA256\n"]
with zipfile.ZipFile(zip_path, 'r') as zipf:
    for name in zipf.namelist():
        data = zipf.read(name)
        size = len(data)
        sha = hashlib.sha256(data).hexdigest()
        manifest.append(f"{name},{size},{sha}\n")

with open(f"{scratch_dir}/FILE_MANIFEST.csv", "w") as f:
    f.writelines(manifest)

# RE-ZIP with manifest
with zipfile.ZipFile(zip_path, 'a', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(f"{scratch_dir}/FILE_MANIFEST.csv", "FILE_MANIFEST.csv")

# VERIFY
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

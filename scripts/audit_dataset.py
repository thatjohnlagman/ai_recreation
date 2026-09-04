"""
scripts/audit_dataset.py
Phase 3 — Full chunked dataset audit + 300k working-sample extraction.

Strategy:
    Two-pass approach:
    Pass 1 — Audit (count valid rows, schema, class distribution). Can be skipped if file_stats.json exists.
    Pass 2 — Streaming exact per-stratum hypergeometric sampling.
             Maintains remaining population and quota per stratum.
             Selects exact allocations chunk-by-chunk using hypergeometric distribution.

Produces:
    data/manifests/raw_files.csv
    data/manifests/schema.json
    data/manifests/class_distribution.csv
    data/manifests/invalid_rows_by_file.csv
    artifacts/reports/data_audit.md
    data/interim/working_sample.parquet
    data/manifests/working_sample_manifest.json
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import warnings
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ─── Project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recall_aware_ids.utils.seeds import set_global_seeds
from recall_aware_ids.utils.config import get_experiment_config

set_global_seeds(42)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

cfg = get_experiment_config()
RAW_DIR = Path(cfg["dataset"]["raw_path"])
CHUNK_SIZE = cfg["storage"]["chunk_size"]
WORKING_SAMPLE_SIZE = cfg["dataset"]["working_sample_size"]
SAMPLE_SEED = cfg["dataset"]["sampling_seed"]
LABEL_COL_CANDIDATES = ["Label", "label", " Label"]

NON_NUMERIC_COLS = {
    "Timestamp", "timestamp", "Flow ID", "flow_id",
    "Src IP", "src_ip", "Dst IP", "dst_ip",
    "Label", "label",
}
BENIGN_SPELLINGS = {"benign", "Benign", "BENIGN", "BENIGN "}

MANIFESTS_DIR = PROJECT_ROOT / "data" / "manifests"
INTERIM_DIR   = PROJECT_ROOT / "data" / "interim"
REPORTS_DIR   = PROJECT_ROOT / "artifacts" / "reports"
for d in [MANIFESTS_DIR, INTERIM_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, buf_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(buf_size):
            h.update(chunk)
    return h.hexdigest()


def normalize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    mapping = {c: c.strip() for c in df.columns}
    return df.rename(columns=mapping), mapping


def detect_label_column(columns: list[str]) -> str | None:
    for candidate in LABEL_COL_CANDIDATES:
        if candidate.strip() in columns:
            return candidate.strip()
    return None


def get_numeric_feature_cols(columns: list[str], label_col: str | None) -> list[str]:
    exclude = set(NON_NUMERIC_COLS)
    if label_col:
        exclude.add(label_col)
    return [c for c in columns if c not in exclude]


def audit_pass(csv_files: list[Path]) -> dict[str, Any]:
    """First pass: count valid rows, compute schema, class distribution."""
    print(f"Found {len(csv_files)} CSV files in {RAW_DIR}")
    print(f"Chunk size: {CHUNK_SIZE:,} rows\n")
    print("PASS 1 — Audit (counting valid rows)...")

    file_stats: list[dict] = []
    all_class_counts: Counter = Counter()
    schema_union: dict[str, str] = {}
    invalid_rows_records: list[dict] = []

    for csv_path in csv_files:
        t0 = time.time()
        print(f"  ─── {csv_path.name} ({csv_path.stat().st_size / 1e6:.1f} MB)")

        file_hash = sha256_file(csv_path)
        total_rows = 0
        valid_rows = 0
        dup_header_rows = 0
        label_col: str | None = None
        numeric_feature_cols: list[str] = []
        class_counts_this_file: Counter = Counter()

        try:
            for chunk_idx, chunk in enumerate(
                pd.read_csv(csv_path, chunksize=CHUNK_SIZE, low_memory=False, on_bad_lines="skip")
            ):
                chunk, _ = normalize_columns(chunk)
                total_rows += len(chunk)

                if chunk_idx == 0:
                    label_col = detect_label_column(list(chunk.columns))
                    numeric_feature_cols = get_numeric_feature_cols(list(chunk.columns), label_col)
                    schema_union.update({c: str(chunk[c].dtype) for c in chunk.columns})

                if label_col and label_col in chunk.columns:
                    dup_mask = chunk[label_col].astype(str).str.strip() == label_col
                    dup_header_rows += int(dup_mask.sum())
                    if dup_mask.any():
                        chunk = chunk[~dup_mask]

                for c in numeric_feature_cols:
                    if c in chunk.columns:
                        chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
                chunk = chunk.replace([np.inf, -np.inf], np.nan)

                chunk = chunk.dropna(subset=numeric_feature_cols)

                if label_col and label_col in chunk.columns:
                    labels = chunk[label_col].astype(str).str.strip()
                    for lbl, cnt in labels.value_counts().items():
                        class_counts_this_file[lbl] += int(cnt)
                        all_class_counts[lbl] += int(cnt)

                valid_rows += len(chunk)

        except Exception as e:
            print(f"    ERROR: {e}")
            file_stats.append({"file": csv_path.name, "error": str(e)})
            continue

        elapsed = time.time() - t0
        print(
            f"    {total_rows:,} total | {valid_rows:,} valid | "
            f"{total_rows - valid_rows:,} dropped | {elapsed:.1f}s"
        )
        print(f"    Classes: {class_counts_this_file.most_common(5)}")

        invalid_rows_records.append({
            "file": csv_path.name,
            "nan_or_inf_dropped": total_rows - valid_rows,
            "dup_headers": dup_header_rows,
        })

        file_stats.append({
            "file": csv_path.name,
            "size_bytes": csv_path.stat().st_size,
            "sha256": file_hash,
            "total_rows": total_rows,
            "valid_rows": valid_rows,
            "invalid_rows": total_rows - valid_rows,
            "dup_header_rows": dup_header_rows,
            "label_column": label_col,
            "n_columns": len(schema_union),
            "class_counts": dict(class_counts_this_file),
        })

    return {
        "file_stats": file_stats,
        "all_class_counts": dict(all_class_counts),
        "schema_union": schema_union,
        "invalid_rows_records": invalid_rows_records,
    }


def compute_allocations(
    file_stats: list[dict],
    target_n: int,
) -> dict[str, int]:
    stratum_counts: dict[str, int] = {}
    for s in file_stats:
        if "error" in s:
            continue
        stem = Path(s["file"]).stem
        classes = s.get("class_counts", {})
        total_valid = s.get("valid_rows", 0)
        benign_count = sum(v for k, v in classes.items() if k.strip() in BENIGN_SPELLINGS)
        attack_count = total_valid - benign_count
        if benign_count > 0:
            stratum_counts[f"{stem}__0"] = benign_count
        if attack_count > 0:
            stratum_counts[f"{stem}__1"] = attack_count

    total = sum(stratum_counts.values())
    alloc: dict[str, int] = {}
    for stratum, count in stratum_counts.items():
        alloc[stratum] = min(int(round(target_n * count / total)), count)

    diff = target_n - sum(alloc.values())
    if diff != 0:
        largest = max(alloc, key=lambda k: alloc[k])
        alloc[largest] = max(0, alloc[largest] + diff)

    print(f"\n  Stratum allocations:")
    for s, n in sorted(alloc.items(), key=lambda x: -x[1])[:10]:
        print(f"    {s}: {n:,}")
    print(f"  Total allocated: {sum(alloc.values()):,}")
    return alloc


def extraction_pass(
    csv_files: list[Path],
    allocations: dict[str, int],
    file_stats: list[dict],
    seed: int = SAMPLE_SEED,
) -> pd.DataFrame:
    print("\nPASS 2 — Extraction (streaming hypergeometric sampling)...")
    rng = np.random.default_rng(seed)

    frames: list[pd.DataFrame] = []

    stratum_populations = {}
    for s in file_stats:
        if "error" in s: continue
        stem = Path(s["file"]).stem
        benign = 0
        for cls, count in s.get("class_counts", {}).items():
            if cls.strip() in BENIGN_SPELLINGS:
                benign += count
        attack = s.get("valid_rows", 0) - benign
        stratum_populations[f"{stem}__0"] = benign
        stratum_populations[f"{stem}__1"] = attack

    for csv_path in csv_files:
        stem = csv_path.stem
        alloc_benign = allocations.get(f"{stem}__0", 0)
        alloc_attack = allocations.get(f"{stem}__1", 0)
        
        pop_benign = stratum_populations.get(f"{stem}__0", 0)
        pop_attack = stratum_populations.get(f"{stem}__1", 0)
        
        if alloc_benign == 0 and alloc_attack == 0:
            continue
            
        print(f"  ─── {stem}: need {alloc_benign:,}/{pop_benign:,} benign + {alloc_attack:,}/{pop_attack:,} attack")
        
        rem_quota_0 = alloc_benign
        rem_pop_0 = pop_benign
        rem_quota_1 = alloc_attack
        rem_pop_1 = pop_attack
        
        raw_row_counter = 0
        extracted_this_file = 0
        
        try:
            for chunk_idx, chunk in enumerate(
                pd.read_csv(csv_path, chunksize=CHUNK_SIZE, low_memory=False, on_bad_lines="skip")
            ):
                chunk_len = len(chunk)
                raw_indices = np.arange(raw_row_counter, raw_row_counter + chunk_len)
                chunk["_raw_row_idx"] = raw_indices
                raw_row_counter += chunk_len
                
                chunk, _ = normalize_columns(chunk)
                
                if chunk_idx == 0:
                    label_col = detect_label_column(list(chunk.columns))
                    numeric_feature_cols = get_numeric_feature_cols(list(chunk.columns), label_col)
                
                if not label_col or label_col not in chunk.columns:
                    continue
                    
                dup_mask = chunk[label_col].astype(str).str.strip() == label_col
                if dup_mask.any():
                    chunk = chunk[~dup_mask]
                    
                for c in numeric_feature_cols:
                    if c in chunk.columns:
                        chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
                chunk = chunk.replace([np.inf, -np.inf], np.nan)
                
                chunk = chunk.dropna(subset=numeric_feature_cols)
                if len(chunk) == 0:
                    continue
                
                labels = chunk[label_col].astype(str).str.strip()
                is_benign = labels.map(lambda x: x in BENIGN_SPELLINGS)
                chunk["y_binary"] = (~is_benign).astype("int8")
                chunk["attack_family"] = labels
                chunk["_source_file"] = stem
                
                b_chunk = chunk[is_benign]
                a_chunk = chunk[~is_benign]
                
                n_c_0 = len(b_chunk)
                if n_c_0 > 0:
                    if rem_quota_0 > 0:
                        # Ensure we don't ask for more than remaining pop (due to minor discrepancies in dup headers, etc.)
                        adj_pop_0 = max(rem_pop_0, n_c_0) 
                        k_0 = rng.hypergeometric(rem_quota_0, adj_pop_0 - rem_quota_0, n_c_0)
                        if k_0 > 0:
                            selected_b = b_chunk.iloc[rng.choice(n_c_0, size=k_0, replace=False)]
                            frames.append(selected_b)
                            rem_quota_0 -= k_0
                            extracted_this_file += k_0
                    rem_pop_0 -= n_c_0

                n_c_1 = len(a_chunk)
                if n_c_1 > 0:
                    if rem_quota_1 > 0:
                        adj_pop_1 = max(rem_pop_1, n_c_1)
                        k_1 = rng.hypergeometric(rem_quota_1, adj_pop_1 - rem_quota_1, n_c_1)
                        if k_1 > 0:
                            selected_a = a_chunk.iloc[rng.choice(n_c_1, size=k_1, replace=False)]
                            frames.append(selected_a)
                            rem_quota_1 -= k_1
                            extracted_this_file += k_1
                    rem_pop_1 -= n_c_1

        except Exception as e:
            print(f"    ERROR: {e}")
            continue
            
        print(f"    Collected {extracted_this_file:,} rows")

    if not frames:
        raise RuntimeError("No rows collected in extraction pass.")

    combined = pd.concat(frames, ignore_index=True)

    drop_cols = [c for c in ["Timestamp", "Flow ID", "Src IP", "Dst IP"]
                 if c in combined.columns]
    combined = combined.drop(columns=drop_cols)

    print(f"\n  Working sample: {len(combined):,} rows")
    print(f"  Binary distribution: {combined['y_binary'].value_counts().to_dict()}")
    print(f"  Attack families: {combined['attack_family'].value_counts().to_dict()}")
    return combined


def write_outputs(
    file_stats: list[dict],
    all_class_counts: dict[str, int],
    schema_union: dict[str, str],
    invalid_rows_records: list[dict],
    sample_df: pd.DataFrame,
    allocations: dict[str, int],
) -> None:
    pd.DataFrame([
        {k: v for k, v in s.items() if k not in ["col_mapping", "class_counts"]}
        for s in file_stats if "error" not in s
    ]).to_csv(MANIFESTS_DIR / "raw_files.csv", index=False)

    with open(MANIFESTS_DIR / "schema.json", "w") as f:
        json.dump(schema_union, f, indent=2)

    pd.DataFrame([
        {"label": k, "count": v}
        for k, v in sorted(all_class_counts.items(), key=lambda x: -x[1])
    ]).to_csv(MANIFESTS_DIR / "class_distribution.csv", index=False)

    pd.DataFrame(invalid_rows_records).to_csv(
        MANIFESTS_DIR / "invalid_rows_by_file.csv", index=False
    )

    tmp_path = INTERIM_DIR / "working_sample_tmp.parquet"
    sample_df.to_parquet(tmp_path, index=False)
    
    ws_path = INTERIM_DIR / "working_sample.parquet"
    shutil.move(tmp_path, ws_path)
    ws_hash = sha256_file(ws_path)
    print(f"\n  working_sample.parquet: {ws_path.stat().st_size / 1e6:.1f} MB | SHA256: {ws_hash[:12]}...")

    with open(MANIFESTS_DIR / "working_sample_manifest.json", "w") as f:
        json.dump({
            "sha256": ws_hash,
            "n_rows": len(sample_df),
            "n_columns": len(sample_df.columns),
            "sampling_seed": SAMPLE_SEED,
            "target_sample_size": WORKING_SAMPLE_SIZE,
            "binary_class_counts": {int(k): int(v) for k, v in
                                    sample_df["y_binary"].value_counts().items()},
            "attack_family_counts": {k: int(v) for k, v in
                                     sample_df["attack_family"].value_counts().items()},
            "stratum_allocations": allocations,
        }, f, indent=2)

    total_raw = sum(s.get("total_rows", 0) for s in file_stats if "error" not in s)
    total_valid = sum(s.get("valid_rows", 0) for s in file_stats if "error" not in s)
    total_bytes = sum(s.get("size_bytes", 0) for s in file_stats if "error" not in s)

    lines = [
        "# Dataset Audit Report — CIC-IDS2018",
        "",
        f"**Dataset path:** `{RAW_DIR}`  ",
        f"**Files audited:** {len([s for s in file_stats if 'error' not in s])}  ",
        f"**Total raw size:** {total_bytes / 1e9:.2f} GB  ",
        f"**Total raw rows:** {total_raw:,}  ",
        f"**Total valid rows:** {total_valid:,}  ",
        f"**Working sample:** {len(sample_df):,} records (300,000 target)  ",
        "",
        "## Per-File Summary",
        "",
        "| File | Size (MB) | Total Rows | Valid Rows | Dropped |",
        "|---|---:|---:|---:|---:|",
    ]
    for s in file_stats:
        if "error" not in s:
            size_mb = s.get('size_bytes', 0) / 1e6
            lines.append(
                f"| {s['file']} | {size_mb:.0f} | {s.get('total_rows', 0):,} | "
                f"{s.get('valid_rows', 0):,} | {s.get('invalid_rows', 0):,} |"
            )

    lines += ["", "## Overall Class Distribution", "", "| Label | Count |", "|---|---:|"]
    for lbl, cnt in sorted(all_class_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| {lbl} | {cnt:,} |")

    lines += [
        "",
        "## Working Sample Binary Distribution",
        "",
        "| Class | Count |",
        "|---|---:|",
        f"| 0 — Benign | {sample_df['y_binary'].eq(0).sum():,} |",
        f"| 1 — Attack | {sample_df['y_binary'].eq(1).sum():,} |",
        "",
        "## Attack Family Distribution (Working Sample)",
        "", "| Family | Count |", "|---|---:|",
    ]
    for fam, cnt in sample_df["attack_family"].value_counts().items():
        lines.append(f"| {fam} | {cnt:,} |")

    lines += ["", "---", "_Generated by scripts/audit_dataset.py_"]
    with open(REPORTS_DIR / "data_audit.md", "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    t_start = time.time()
    print("=" * 70)
    print("PHASE 3 — Dataset Audit + Working Sample Extraction")
    print("=" * 70)

    csv_files = sorted(RAW_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR}")

    file_stats_path = MANIFESTS_DIR / "file_stats.json"
    result = {}
    if file_stats_path.exists():
        print("Using cached file_stats.json (skipping Pass 1)")
        with open(file_stats_path, "r") as f:
            file_stats = json.load(f)
            
        all_class_counts = Counter()
        for s in file_stats:
            if "error" not in s:
                for k, v in s.get("class_counts", {}).items():
                    all_class_counts[k] += v
        
        schema_path = MANIFESTS_DIR / "schema.json"
        if schema_path.exists():
            with open(schema_path, "r") as f:
                schema_union = json.load(f)
        else:
            schema_union = {}
            
        result = {
            "file_stats": file_stats,
            "all_class_counts": dict(all_class_counts),
            "schema_union": schema_union,
            "invalid_rows_records": [] # Not essential for pass 2
        }
    else:
        result = audit_pass(csv_files)

    print("\n" + "=" * 70)
    print("Computing proportional allocations...")
    print("=" * 70)
    allocations = compute_allocations(result["file_stats"], WORKING_SAMPLE_SIZE)

    print("\n" + "=" * 70)
    sample_df = extraction_pass(csv_files, allocations, result["file_stats"], seed=SAMPLE_SEED)

    print("\nWriting outputs...")
    write_outputs(
        file_stats=result["file_stats"],
        all_class_counts=result["all_class_counts"],
        schema_union=result["schema_union"],
        invalid_rows_records=result.get("invalid_rows_records", []),
        sample_df=sample_df,
        allocations=allocations,
    )

    elapsed = time.time() - t_start
    print(f"\n✓ Phase 3 complete in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print("  Next: python scripts/preprocess_dataset.py")


if __name__ == "__main__":
    main()

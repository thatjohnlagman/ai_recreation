#!/usr/bin/env python3
import sys
import os
import json
import csv
import shutil
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_evaluation import ( build_production_provenance,
    derive_and_validate_matrix,
    canonicalize_scenario,
    canonicalize_defense,
    calculate_file_hash,
    validate_completed_run,
    validate_completed_alias,
    validate_cache_against_inventory
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "artifacts" / "evaluation_runs"
ANALYSIS_DIR = REPO_ROOT / "artifacts" / "analysis"

def check_eq(actual, expected, msg):
    if actual != expected:
        raise ValueError(f"{msg}: Expected {expected}, got {actual}")

def calculate_metrics(tp, fp, tn, fn, total=500):
    check_eq(tp + fp + tn + fn, total, "Batch total")
    acc = (tp + tn) / float(total)
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    
    tpr = rec
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc = (tpr + tnr) / 2.0
    return acc, prec, rec, f1, bal_acc

def safe_float(v):
    if v is None: return None
    f = float(v)
    import math
    if math.isnan(f) or math.isinf(f):
        raise ValueError(f"Invalid metric value: {v}")
    return f

def check_tolerance(actual, expected, msg, tol=1e-5):
    if expected is None and actual is None:
        return
    if expected is None or actual is None:
        if expected != actual:
            raise ValueError(f"{msg}: mismatch {actual} vs {expected}")
        return
    if abs(actual - expected) > tol:
        raise ValueError(f"{msg}: {actual} != {expected}")

def process_run(row, expected_batches=144):
    rid = row["run_id"]
    run_dir = EVAL_DIR / rid
    
    with open(run_dir / "run_summary.json") as f:
        summary = json.load(f)
    with open(run_dir / "completion.json") as f:
        completion = json.load(f)
    with open(run_dir / "confusion.json") as f:
        confusions = json.load(f)
    with open(run_dir / "config.json") as f:
        configs = json.load(f)
        
    check_eq(len(confusions), expected_batches, f"Batch count in confusion.json for {rid}")
    check_eq(len(configs), expected_batches, f"Batch count in config.json for {rid}")
    
    # Contract validation for ASR
    if row["attack_scenario"] == "SilentProbing":
        check_eq(summary.get("attack_success_rate"), None, f"{rid}: SilentProbing ASR must be null")
    else:
        # ASR is valid for other attacks
        pass
        
    run_tp = run_fp = run_tn = run_fn = 0
    
    batch_records = []
    controller_records = []
    
    expected_bid = 0
    for i in range(expected_batches):
        c = confusions[i]
        cfg = configs[i]
        
        bid = c["batch_id"]
        check_eq(bid, expected_bid, f"{rid}: Batch ID out of order or missing")
        expected_bid += 1
        
        check_eq(cfg["batch_id"], bid, f"Batch ID mismatch between confusion and config {rid} {i}")
        
        tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
        
        # Verify batch totals
        acc, prec, rec, f1, bal_acc = calculate_metrics(tp, fp, tn, fn)
        
        # Verify against serialized
        check_tolerance(acc, c.get("accuracy"), f"{rid} batch {bid} accuracy")
        check_tolerance(prec, c.get("precision"), f"{rid} batch {bid} precision")
        check_tolerance(rec, c.get("recall"), f"{rid} batch {bid} recall")
        check_tolerance(f1, c.get("f1"), f"{rid} batch {bid} f1_score")
        check_tolerance(bal_acc, c.get("balanced_accuracy"), f"{rid} batch {bid} balanced_accuracy")
        
        run_tp += tp
        run_fp += fp
        run_tn += tn
        run_fn += fn
        
        brec = {
            "run_id": rid,
            "seed": int(row["seed"]),
            "attack_scenario": canonicalize_scenario(row["attack_scenario"]),
            "defense_mechanism": canonicalize_defense(row["defense_name"]),
            "controller_config": row["controller_config_id"],
            "batch_id": bid,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1, "balanced_accuracy": bal_acc
        }
        batch_records.append(brec)
        
        if row["controller_config_id"] != "Base":
            clip_ind = "None"
            if cfg.get("hit_max_bound"): clip_ind = "Max"
            if cfg.get("hit_min_bound"): clip_ind = "Min"
            
            crec = {
                "run_id": rid,
                "batch_id": bid,
                "controller_config": row["controller_config_id"],
                "intensity": cfg.get("intensity", None),
                "state": cfg.get("state", "N/A"),
                "rolling_recall": cfg.get("rolling_recall", None),
                "multiplier": cfg.get("multiplier", None),
                "clipping_indicator": clip_ind,
                "zero_denominator_indicator": cfg.get("zero_denominator", False)
            }
            controller_records.append(crec)
        else:
            crec = {
                "run_id": rid,
                "batch_id": bid,
                "controller_config": "Base",
                "intensity": cfg.get("intensity", None),
                "state": "Base",
                "rolling_recall": None,
                "multiplier": 1.0,
                "clipping_indicator": "None",
                "zero_denominator_indicator": False
            }
            controller_records.append(crec)
            
    # Verify run totals
    check_eq(run_tp + run_fp + run_tn + run_fn, expected_batches * 500, f"{rid} run total records")
    check_eq(run_tp, summary["tp"], f"{rid} run TP")
    check_eq(run_fp, summary["fp"], f"{rid} run FP")
    check_eq(run_tn, summary["tn"], f"{rid} run TN")
    check_eq(run_fn, summary["fn"], f"{rid} run FN")
    
    run_acc, run_prec, run_rec, run_f1, run_bal_acc = calculate_metrics(run_tp, run_fp, run_tn, run_fn, total=72000)
    
    check_tolerance(run_acc, summary.get("accuracy"), f"{rid} run accuracy")
    check_tolerance(run_prec, summary.get("precision"), f"{rid} run precision")
    check_tolerance(run_rec, summary.get("recall"), f"{rid} run recall")
    check_tolerance(run_f1, summary.get("f1"), f"{rid} run f1_score")
    check_tolerance(run_bal_acc, summary.get("balanced_accuracy"), f"{rid} run balanced_accuracy")
    
    run_rec_out = {
        "run_id": rid,
        "seed": int(row["seed"]),
        "attack_scenario": canonicalize_scenario(row["attack_scenario"]),
        "defense_mechanism": canonicalize_defense(row["defense_name"]),
        "controller_config": row["controller_config_id"],
        "TP": run_tp, "FP": run_fp, "TN": run_tn, "FN": run_fn,
        "accuracy": run_acc, "precision": run_prec, "recall": run_rec, "f1_score": run_f1, "balanced_accuracy": run_bal_acc
    }
    
    return run_rec_out, batch_records, controller_records

def write_csv(path, records, keys):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)
    tmp.replace(path)

def verify_all_caches(base_provenance):
    scenarios = ["SilentProbing", "SurrogateTransfer", "DecisionBoundary"]
    seeds = [42, 43, 44, 45, 46]
    inventory_path = REPO_ROOT / "artifacts" / "reports" / "cache_inventory_v2.json"
    
    for sc in scenarios:
        for s in seeds:
            cache_dir = REPO_ROOT / "artifacts" / "caches" / f"{sc}_{s}"
            validate_cache_against_inventory(cache_dir, sc, s, inventory_path, base_provenance)

def main():
    print("Building analysis tables...")
    
    inv_file = REPO_ROOT / "artifacts" / "reports" / "phase10d_output_inventory.json"
    with open(inv_file) as f:
        inventory = json.load(f)
        
    inv_hashes = {i["relative_path"]: i["sha256"] for i in inventory}
    
    # 1. Verify all 1,314 artifacts
    print("Verifying 1,314 artifacts against inventory...")
    actual_count = 0
    for rel_path, expected_h in inv_hashes.items():
        actual_path = REPO_ROOT / rel_path
        if not actual_path.exists():
            raise FileNotFoundError(f"Inventory artifact missing: {rel_path}")
        if calculate_file_hash(actual_path) != expected_h:
            raise ValueError(f"Inventory artifact hash mismatch: {rel_path}")
        actual_count += 1
    check_eq(actual_count, 1314, "Inventoried artifacts verified")
    
    # Exclude unexpected final-run artifacts
    for f in EVAL_DIR.rglob("*"):
        if f.is_file():
            # Check if this file is in the inventory
            rel = str(f.relative_to(REPO_ROOT))
            if rel not in inv_hashes:
                if "quarantine_" in rel or "_quarantined_" in rel:
                    continue
                raise ValueError(f"Unexpected artifact not in Phase 10D inventory: {rel}")
                
    staging = list(EVAL_DIR.glob(".staging*"))
    check_eq(len(staging), 0, "Staging directories found")
    
    quarantined = list(EVAL_DIR.glob("*quarantined_*"))
    check_eq(len(quarantined), 1, "Exactly one historical quarantine directory expected")
    
    base_provenance = build_production_provenance()
    matrix, _ = derive_and_validate_matrix(REPO_ROOT / "configs")
    
    # Verify caches BEFORE
    verify_all_caches(base_provenance)
    
    unique_runs = matrix[~matrix["is_alias"]]
    alias_runs = matrix[matrix["is_alias"]]
    
    check_eq(len(unique_runs), 252, "Unique runs")
    check_eq(len(alias_runs), 27, "Aliases")
    
    primary_run_records = []
    primary_batch_records = []
    sens_run_records = []
    sens_batch_records = []
    all_controller_records = []
    
    run_records_dict = {}
    batch_records_dict = {}
    
    print("Processing unique runs...")
    for _, row in unique_runs.iterrows():
        rid = row["run_id"]
        run_dir = EVAL_DIR / rid
        
        # Recreate expected provenance
        run_prov = dict(base_provenance)
        cache_dir = REPO_ROOT / "artifacts" / "caches" / f"{canonicalize_scenario(row['attack_scenario'])}_{row['seed']}"
        run_prov["cache_manifest_hash"] = calculate_file_hash(cache_dir / "manifest.json")
        
        # Get cache identity for validation
        with open(cache_dir / "manifest.json") as f:
            cache_identity = json.load(f)
            
        # Hardened validation
        is_valid, err = validate_completed_run(run_dir, row, run_prov, cache_identity)
        if not is_valid:
            raise ValueError(f"Invalid run {rid}: {err}")
            
        rr, brs, crs = process_run(row)
        run_records_dict[rid] = rr
        batch_records_dict[rid] = brs
        
        all_controller_records.extend(crs)
        
        if rid.startswith("primary_"):
            primary_run_records.append(rr)
            primary_batch_records.extend(brs)
        elif rid.startswith("sensitivity_"):
            sens_run_records.append(rr)
            sens_batch_records.extend(brs)
            
    print("Processing aliases...")
    for _, row in alias_runs.iterrows():
        rid = row["run_id"]
        run_dir = EVAL_DIR / rid
        
        # Recreate target expected provenance
        target_prov = dict(base_provenance)
        cache_dir = REPO_ROOT / "artifacts" / "caches" / f"{canonicalize_scenario(row['attack_scenario'])}_{row['seed']}"
        target_prov["cache_manifest_hash"] = calculate_file_hash(cache_dir / "manifest.json")
        
        # Alias points to a unique run
        with open(EVAL_DIR / rid / "alias_pointer.json") as f:
            pointer = json.load(f)
        target = pointer["alias_for_run_id"]
        
        # Ensure it targets the correct primary C1
        expected_target = f"primary_{row['seed']}_{canonicalize_scenario(row['attack_scenario'])}_{canonicalize_defense(row['defense_name'])}_C1"
        check_eq(target, expected_target, f"{rid} must target exactly its primary C1 equivalent")
        
        is_valid, err = validate_completed_alias(run_dir, row, target_prov, EVAL_DIR)
        if not is_valid:
            raise ValueError(f"Invalid alias {rid}: {err}")
        
        rr = run_records_dict[target].copy()
        rr["run_id"] = rid  # Re-label the run_id for the sensitivity table
        rr["controller_config"] = row["controller_config_id"]
        
        sens_run_records.append(rr)
        
        brs = []
        for orig_b in batch_records_dict[target]:
            nb = orig_b.copy()
            nb["run_id"] = rid
            nb["controller_config"] = row["controller_config_id"]
            brs.append(nb)
            
        sens_batch_records.extend(brs)
        
    check_eq(len(primary_run_records), 90, "Primary run records")
    check_eq(len(primary_batch_records), 12960, "Primary batch records")
    check_eq(len(sens_run_records), 189, "Sensitivity run records")
    check_eq(len(sens_batch_records), 27216, "Sensitivity batch records")
    
    # Paired differences
    print("Building paired difference tables...")
    primary_paired_runs = []
    primary_paired_batches = []
    
    # Index base runs
    base_runs = { (r["seed"], r["attack_scenario"], r["defense_mechanism"]): r for r in primary_run_records if r["controller_config"] == "Base" }
    c1_runs = { (r["seed"], r["attack_scenario"], r["defense_mechanism"]): r for r in primary_run_records if r["controller_config"] == "C1" }
    
    base_batches = { (r["seed"], r["attack_scenario"], r["defense_mechanism"], r["batch_id"]): r for r in primary_batch_records if r["controller_config"] == "Base" }
    c1_batches = { (r["seed"], r["attack_scenario"], r["defense_mechanism"], r["batch_id"]): r for r in primary_batch_records if r["controller_config"] == "C1" }
    
    check_eq(len(base_runs), 45, "Base run count")
    check_eq(len(c1_runs), 45, "C1 run count")
    
    for k, c1r in c1_runs.items():
        br = base_runs[k]
        primary_paired_runs.append({
            "seed": k[0],
            "attack_scenario": k[1],
            "defense_mechanism": k[2],
            "diff_accuracy": c1r["accuracy"] - br["accuracy"],
            "diff_precision": c1r["precision"] - br["precision"],
            "diff_recall": c1r["recall"] - br["recall"],
            "diff_f1_score": c1r["f1_score"] - br["f1_score"],
            "diff_balanced_accuracy": c1r["balanced_accuracy"] - br["balanced_accuracy"],
            "base_run_id": br["run_id"],
            "c1_run_id": c1r["run_id"]
        })
        
    for k, c1b in c1_batches.items():
        bb = base_batches[k]
        primary_paired_batches.append({
            "seed": k[0],
            "attack_scenario": k[1],
            "defense_mechanism": k[2],
            "batch_id": k[3],
            "diff_accuracy": c1b["accuracy"] - bb["accuracy"],
            "diff_precision": c1b["precision"] - bb["precision"],
            "diff_recall": c1b["recall"] - bb["recall"],
            "diff_f1_score": c1b["f1_score"] - bb["f1_score"],
            "diff_balanced_accuracy": c1b["balanced_accuracy"] - bb["balanced_accuracy"],
            "base_run_id": bb["run_id"],
            "c1_run_id": c1b["run_id"]
        })
        
    check_eq(len(primary_paired_runs), 45, "Paired run records")
    check_eq(len(primary_paired_batches), 6480, "Paired batch records")
    
    print("Writing analysis tables...")
    run_keys = ["run_id", "seed", "attack_scenario", "defense_mechanism", "controller_config", "TP", "FP", "TN", "FN", "accuracy", "precision", "recall", "f1_score", "balanced_accuracy"]
    batch_keys = ["run_id", "seed", "attack_scenario", "defense_mechanism", "controller_config", "batch_id", "TP", "FP", "TN", "FN", "accuracy", "precision", "recall", "f1_score", "balanced_accuracy"]
    run_diff_keys = ["seed", "attack_scenario", "defense_mechanism", "diff_accuracy", "diff_precision", "diff_recall", "diff_f1_score", "diff_balanced_accuracy", "base_run_id", "c1_run_id"]
    batch_diff_keys = ["seed", "attack_scenario", "defense_mechanism", "batch_id", "diff_accuracy", "diff_precision", "diff_recall", "diff_f1_score", "diff_balanced_accuracy", "base_run_id", "c1_run_id"]
    ctrl_keys = ["run_id", "batch_id", "controller_config", "intensity", "state", "rolling_recall", "multiplier", "clipping_indicator", "zero_denominator_indicator"]
    
    ANALYSIS_DIR.mkdir(exist_ok=True, parents=True)
    
    write_csv(ANALYSIS_DIR / "primary_run_level.csv", primary_run_records, run_keys)
    write_csv(ANALYSIS_DIR / "primary_batch_level.csv", primary_batch_records, batch_keys)
    write_csv(ANALYSIS_DIR / "primary_paired_run_differences.csv", primary_paired_runs, run_diff_keys)
    write_csv(ANALYSIS_DIR / "primary_paired_batch_differences.csv", primary_paired_batches, batch_diff_keys)
    write_csv(ANALYSIS_DIR / "sensitivity_run_level.csv", sens_run_records, run_keys)
    write_csv(ANALYSIS_DIR / "sensitivity_batch_level.csv", sens_batch_records, batch_keys)
    write_csv(ANALYSIS_DIR / "controller_trace_summary.csv", all_controller_records, ctrl_keys)
    
    # Verify caches AFTER
    verify_all_caches(base_provenance)
    
    print("Analysis tables built successfully.")

if __name__ == "__main__":
    main()

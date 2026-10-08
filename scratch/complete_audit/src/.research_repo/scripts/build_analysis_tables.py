#!/usr/bin/env python3
import sys
import os
import json
import csv
import shutil
import hashlib
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.recall_aware_ids.experiment.schemas import RunSummary
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
    
    with open(run_dir / "run_summary.json") as f: summary = json.load(f)
    with open(run_dir / "completion.json") as f: completion = json.load(f)
    with open(run_dir / "confusion.json") as f: confusions = json.load(f)
    with open(run_dir / "config.json") as f: configs = json.load(f)
    with open(run_dir / "scores.json") as f: scores = json.load(f)
        
    check_eq(summary["run_id"], rid, f"{rid}: run_summary.json run_id mismatch")
    check_eq(completion["run_id"], rid, f"{rid}: completion.json run_id mismatch")
    
    check_eq(summary["seed"], row["seed"], f"{rid}: summary seed mismatch")
    check_eq(canonicalize_scenario(summary["attack_scenario"]), canonicalize_scenario(row["attack_scenario"]), f"{rid}: summary attack_scenario mismatch")
    check_eq(canonicalize_defense(summary["defense"]), canonicalize_defense(row["defense_name"]), f"{rid}: summary defense_mechanism mismatch")
    check_eq(summary["config_id"], row["controller_config_id"], f"{rid}: summary controller_config mismatch")
    
    check_eq(len(configs), expected_batches, f"{rid}: config.json len mismatch")
    check_eq(len(confusions), expected_batches, f"{rid}: confusion.json len mismatch")
    check_eq(len(scores), expected_batches, f"{rid}: scores.json len mismatch")
    
    for i in range(expected_batches):
        check_eq(configs[i]["batch_id"], i, f"{rid}: config batch_id {i} mismatch")
        check_eq(configs[i]["run_id"], rid, f"{rid}: config run_id {i} mismatch")
        check_eq(confusions[i]["batch_id"], i, f"{rid}: confusion batch_id {i} mismatch")
        check_eq(confusions[i]["run_id"], rid, f"{rid}: confusion run_id {i} mismatch")
        check_eq(scores[i]["batch_id"], i, f"{rid}: scores batch_id {i} mismatch")
        check_eq(scores[i]["run_id"], rid, f"{rid}: scores run_id {i} mismatch")
        
    try:
        RunSummary(**summary)
    except Exception as e:
        raise ValueError(f"{rid}: Schema validation failed: {e}")
        
    if canonicalize_scenario(row["attack_scenario"]) != "SilentProbing":
        att = summary["total_attempted"]
        succ = summary["total_successful"]
        gasr = summary["global_asr"]
        if att == 0:
            check_eq(gasr, 0.0, f"{rid}: global_asr == 0.0")
        else:
            check_tolerance(gasr, float(succ) / att, f"{rid}: global_asr calculation")
        
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
        acc, prec, rec, f1, bal_acc = calculate_metrics(tp, fp, tn, fn)
        
        check_tolerance(acc, c.get("accuracy"), f"{rid} batch {bid} accuracy")
        check_tolerance(prec, c.get("precision"), f"{rid} batch {bid} precision")
        check_tolerance(rec, c.get("recall"), f"{rid} batch {bid} recall")
        check_tolerance(f1, c.get("f1"), f"{rid} batch {bid} f1_score")
        check_tolerance(bal_acc, c.get("balanced_accuracy"), f"{rid} batch {bid} balanced_accuracy")
        
        run_tp += tp; run_fp += fp; run_tn += tn; run_fn += fn
        
        brec = {
            "run_id": rid,
            "seed": row["seed"],
            "attack_scenario": canonicalize_scenario(row["attack_scenario"]),
            "defense_mechanism": canonicalize_defense(row["defense_name"]),
            "controller_config": row["controller_config_id"],
            "batch_id": bid,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1, "balanced_accuracy": bal_acc
        }
        batch_records.append(brec)
        
        crec = {
            "run_id": rid,
            "batch_id": bid,
            "controller_config": cfg.get("config_id"),
            "intensity": safe_float(cfg.get("intensity")),
            "state": cfg.get("state"),
            "rolling_recall": safe_float(cfg.get("rolling_recall")),
            "multiplier": safe_float(cfg.get("multiplier")),
            "clipping_indicator": 1 if (cfg.get("hit_min_bound") or cfg.get("hit_max_bound")) else 0,
            "zero_denominator_indicator": 1 if cfg.get("zero_denominator") else 0
        }
        controller_records.append(crec)
        
    run_acc, run_prec, run_rec, run_f1, run_bal_acc = calculate_metrics(run_tp, run_fp, run_tn, run_fn, total=500*expected_batches)
    check_tolerance(run_acc, summary.get("accuracy"), f"{rid} summary accuracy")
    check_tolerance(run_prec, summary.get("precision"), f"{rid} summary precision")
    check_tolerance(run_rec, summary.get("recall"), f"{rid} summary recall")
    check_tolerance(run_f1, summary.get("f1"), f"{rid} summary f1_score")
    check_tolerance(run_bal_acc, summary.get("balanced_accuracy"), f"{rid} summary balanced_accuracy")
    
    run_record = {
        "run_id": rid,
        "seed": row["seed"],
        "attack_scenario": canonicalize_scenario(row["attack_scenario"]),
        "defense_mechanism": canonicalize_defense(row["defense_name"]),
        "controller_config": row["controller_config_id"],
        "TP": run_tp, "FP": run_fp, "TN": run_tn, "FN": run_fn,
        "accuracy": run_acc, "precision": run_prec, "recall": run_rec, "f1_score": run_f1, "balanced_accuracy": run_bal_acc
    }
    return run_record, batch_records, controller_records

def exact_alias_resolution(row, target_prov, eval_dir, repo_root):
    rid = row["run_id"]
    run_dir = eval_dir / rid
    
    with open(eval_dir / rid / "alias_pointer.json") as f:
        pointer = json.load(f)
    target = pointer["alias_for_run_id"]
    
    expected_target = f"primary_{row['seed']}_{canonicalize_scenario(row['attack_scenario'])}_{canonicalize_defense(row['defense_name'])}_C1"
    check_eq(target, expected_target, f"{rid} must target exactly its primary C1 equivalent")
    
    is_valid, err = validate_completed_alias(run_dir, row, target_prov, eval_dir)
    if not is_valid:
        raise ValueError(f"Invalid alias {rid}: {err}")
    return target

def base_c1_pairing_and_difference_construction(primary_run_records, primary_batch_records):
    primary_paired_runs = []
    primary_paired_batches = []
    
    base_runs = { (r["seed"], r["attack_scenario"], r["defense_mechanism"]): r for r in primary_run_records if r["controller_config"] == "Base" }
    c1_runs = { (r["seed"], r["attack_scenario"], r["defense_mechanism"]): r for r in primary_run_records if r["controller_config"] == "C1" }
    
    base_batches = { (r["seed"], r["attack_scenario"], r["defense_mechanism"], r["batch_id"]): r for r in primary_batch_records if r["controller_config"] == "Base" }
    c1_batches = { (r["seed"], r["attack_scenario"], r["defense_mechanism"], r["batch_id"]): r for r in primary_batch_records if r["controller_config"] == "C1" }
    
    for k, c1r in c1_runs.items():
        br = base_runs[k]
        primary_paired_runs.append({
            "seed": k[0], "attack_scenario": k[1], "defense_mechanism": k[2],
            "diff_accuracy": c1r["accuracy"] - br["accuracy"],
            "diff_precision": c1r["precision"] - br["precision"],
            "diff_recall": c1r["recall"] - br["recall"],
            "diff_f1_score": c1r["f1_score"] - br["f1_score"],
            "diff_balanced_accuracy": c1r["balanced_accuracy"] - br["balanced_accuracy"],
            "base_run_id": br["run_id"], "c1_run_id": c1r["run_id"]
        })
        
    for k, c1b in c1_batches.items():
        bb = base_batches[k]
        primary_paired_batches.append({
            "seed": k[0], "attack_scenario": k[1], "defense_mechanism": k[2], "batch_id": k[3],
            "diff_accuracy": c1b["accuracy"] - bb["accuracy"],
            "diff_precision": c1b["precision"] - bb["precision"],
            "diff_recall": c1b["recall"] - bb["recall"],
            "diff_f1_score": c1b["f1_score"] - bb["f1_score"],
            "diff_balanced_accuracy": c1b["balanced_accuracy"] - bb["balanced_accuracy"],
            "base_run_id": bb["run_id"], "c1_run_id": c1b["run_id"]
        })
        
    return primary_paired_runs, primary_paired_batches

def verify_official_inventory(inv_file, expected_count=1314, repo_root=REPO_ROOT):
    with open(inv_file) as f:
        inventory = json.load(f)
    inv_hashes = {i["relative_path"]: i["sha256"] for i in inventory}
    
    actual_count = 0
    for rel_path, expected_h in inv_hashes.items():
        actual_path = repo_root / rel_path
        if not actual_path.exists():
            raise FileNotFoundError(f"Inventory artifact missing: {rel_path}")
        if calculate_file_hash(actual_path) != expected_h:
            raise ValueError(f"Inventory artifact hash mismatch: {rel_path}")
        actual_count += 1
    check_eq(actual_count, expected_count, "Inventoried artifacts verified")
    return inv_hashes

def reject_unexpected_artifacts(inv_hashes, eval_dir, repo_root):
    for f in eval_dir.rglob("*"):
        if f.is_file():
            rel = str(f.relative_to(repo_root))
            if rel not in inv_hashes:
                if "quarantine_" in rel or "_quarantined_" in rel:
                    continue
                raise ValueError(f"Unexpected artifact not in evaluation inventory: {rel}")
                
    staging = list(eval_dir.glob(".staging*"))
    check_eq(len(staging), 0, "Staging directories found")
    quarantined = list(eval_dir.glob("*quarantined_*"))
    check_eq(len(quarantined), 1, "Exactly one historical quarantine directory expected")

def write_csv(path, records, keys):
    temp_path = path.with_name(path.name + ".tmp")
    with open(temp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)
    temp_path.replace(path)

def deterministic_table_generation(analysis_dir, primary_run_records, primary_batch_records, primary_paired_runs, primary_paired_batches, sens_run_records, sens_batch_records, all_controller_records):
    run_keys = ["run_id", "seed", "attack_scenario", "defense_mechanism", "controller_config", "TP", "FP", "TN", "FN", "accuracy", "precision", "recall", "f1_score", "balanced_accuracy"]
    batch_keys = ["run_id", "seed", "attack_scenario", "defense_mechanism", "controller_config", "batch_id", "TP", "FP", "TN", "FN", "accuracy", "precision", "recall", "f1_score", "balanced_accuracy"]
    run_diff_keys = ["seed", "attack_scenario", "defense_mechanism", "diff_accuracy", "diff_precision", "diff_recall", "diff_f1_score", "diff_balanced_accuracy", "base_run_id", "c1_run_id"]
    batch_diff_keys = ["seed", "attack_scenario", "defense_mechanism", "batch_id", "diff_accuracy", "diff_precision", "diff_recall", "diff_f1_score", "diff_balanced_accuracy", "base_run_id", "c1_run_id"]
    ctrl_keys = ["run_id", "batch_id", "controller_config", "intensity", "state", "rolling_recall", "multiplier", "clipping_indicator", "zero_denominator_indicator"]
    
    analysis_dir.mkdir(exist_ok=True, parents=True)
    write_csv(analysis_dir / "primary_run_level.csv", primary_run_records, run_keys)
    write_csv(analysis_dir / "primary_batch_level.csv", primary_batch_records, batch_keys)
    write_csv(analysis_dir / "primary_paired_run_differences.csv", primary_paired_runs, run_diff_keys)
    write_csv(analysis_dir / "primary_paired_batch_differences.csv", primary_paired_batches, batch_diff_keys)
    write_csv(analysis_dir / "sensitivity_run_level.csv", sens_run_records, run_keys)
    write_csv(analysis_dir / "sensitivity_batch_level.csv", sens_batch_records, batch_keys)
    write_csv(analysis_dir / "controller_trace_summary.csv", all_controller_records, ctrl_keys)

def main():
    print("Building analysis tables...")
    
    with open(REPO_ROOT / "artifacts" / "reports" / "phase11a_official_output_immutability.log", "w") as f:
        subprocess.run([sys.executable, "scripts/verify_immutability.py", "--before"], stdout=f, stderr=subprocess.STDOUT, check=True)
        
    with open(REPO_ROOT / "artifacts" / "reports" / "phase11a_protected_artifact_hashes.log", "w") as f:
        subprocess.run([sys.executable, "scripts/verify_protected_caches.py", "--before"], stdout=f, stderr=subprocess.STDOUT, check=True)
        
    inv_file = REPO_ROOT / "artifacts" / "reports" / "phase10d_output_inventory.json"
    inv_hashes = verify_official_inventory(inv_file)
    reject_unexpected_artifacts(inv_hashes, EVAL_DIR, REPO_ROOT)
        
    base_provenance = build_production_provenance()
    df, _ = derive_and_validate_matrix(REPO_ROOT / "configs")
        
    unique_runs = df[~df["is_alias"]]
    alias_runs = df[df["is_alias"]]
    
    check_eq(len(unique_runs), 252, "Unique runs")
    check_eq(len(alias_runs), 27, "Alias runs")
    
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
        
        run_prov = dict(base_provenance)
        cache_dir = REPO_ROOT / "artifacts" / "caches" / f"{canonicalize_scenario(row['attack_scenario'])}_{row['seed']}"
        run_prov["cache_manifest_hash"] = calculate_file_hash(cache_dir / "manifest.json")
        
        with open(cache_dir / "manifest.json") as f:
            cache_identity = json.load(f)
            
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
        
        target_prov = dict(base_provenance)
        cache_dir = REPO_ROOT / "artifacts" / "caches" / f"{canonicalize_scenario(row['attack_scenario'])}_{row['seed']}"
        target_prov["cache_manifest_hash"] = calculate_file_hash(cache_dir / "manifest.json")
        
        target = exact_alias_resolution(row, target_prov, EVAL_DIR, REPO_ROOT)
        
        rr = run_records_dict[target].copy()
        rr["run_id"] = rid
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
    
    print("Building paired difference tables...")
    primary_paired_runs, primary_paired_batches = base_c1_pairing_and_difference_construction(primary_run_records, primary_batch_records)
    
    check_eq(len(primary_paired_runs), 45, "Paired run records")
    check_eq(len(primary_paired_batches), 6480, "Paired batch records")
    
    print("Writing analysis tables...")
    deterministic_table_generation(ANALYSIS_DIR, primary_run_records, primary_batch_records, primary_paired_runs, primary_paired_batches, sens_run_records, sens_batch_records, all_controller_records)
    
    with open(REPO_ROOT / "artifacts" / "reports" / "phase11a_official_output_immutability.log", "a") as f:
        f.write("\n")
        subprocess.run([sys.executable, "scripts/verify_immutability.py", "--after"], stdout=f, stderr=subprocess.STDOUT, check=True)
        
    with open(REPO_ROOT / "artifacts" / "reports" / "phase11a_protected_artifact_hashes.log", "a") as f:
        f.write("\n")
        subprocess.run([sys.executable, "scripts/verify_protected_caches.py", "--after"], stdout=f, stderr=subprocess.STDOUT, check=True)

    print("Analysis tables built successfully.")

if __name__ == "__main__":
    main()

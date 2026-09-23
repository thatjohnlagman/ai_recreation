#!/usr/bin/env python3
import sys
import os
import csv
import json
import hashlib
from pathlib import Path
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import math

REPO_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = REPO_ROOT / "artifacts" / "analysis"

def calculate_file_hash(path: Path) -> str:
    with open(path, "rb") as bf:
        return hashlib.sha256(bf.read()).hexdigest()

def verify_table_hashes():
    summary_path = REPO_ROOT / "artifacts" / "reports" / "phase11a_analysis_tables_summary.txt"
    if not summary_path.exists():
        raise FileNotFoundError("Missing phase11a_analysis_tables_summary.txt")
        
    expected_hashes = {}
    with open(summary_path, "r") as f:
        current_table = None
        for line in f:
            line = line.strip()
            if line.startswith("Table: "):
                current_table = line.replace("Table: ", "")
            elif line.startswith("SHA-256: ") and current_table:
                expected_hashes[current_table] = line.replace("SHA-256: ", "")
                current_table = None
                
    for table, expected_hash in expected_hashes.items():
        table_path = ANALYSIS_DIR / table
        if not table_path.exists():
            raise FileNotFoundError(f"Missing {table}")
        actual_hash = calculate_file_hash(table_path)
        if actual_hash != expected_hash:
            raise ValueError(f"Hash mismatch for {table}: expected {expected_hash}, got {actual_hash}. Upstream artifacts have been tampered with!")

def compute_t_test(values):
    n = len(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 0 else 0.0
    df = n - 1
    
    if np.all(values == 0):
        t_stat = 0.0
        p_raw = 1.0
        ci_lower, ci_upper = 0.0, 0.0
        dz = 0.0
    elif np.all(values == values[0]) and values[0] != 0:
        t_stat = float("inf") if values[0] > 0 else float("-inf")
        p_raw = 0.0
        ci_lower, ci_upper = mean, mean
        dz = None
    else:
        res = stats.ttest_1samp(values, 0.0)
        t_stat = float(res.statistic)
        p_raw = float(res.pvalue)
        ci = res.confidence_interval(confidence_level=0.95)
        ci_lower, ci_upper = float(ci.low), float(ci.high)
        dz = float(mean / std) if std != 0 else 0.0
        
    if t_stat == float('inf'):
        t_stat = "Infinity"
    elif t_stat == float('-inf'):
        t_stat = "-Infinity"
        
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "se": se,
        "t": t_stat,
        "df": df,
        "p_raw": p_raw,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "cohen_dz": dz
    }

def run_primary_inference():
    path = ANALYSIS_DIR / "primary_paired_batch_differences.csv"
    data = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
            
    results = []
    p_raws = []
    defenses = ["afp", "randomized_smoothing", "feature_squeezing"]
    metrics = ["diff_precision", "diff_recall", "diff_f1_score"]
    
    for defense in defenses:
        for metric in metrics:
            subset = [float(r[metric]) for r in data if r["defense_mechanism"] == defense]
            if len(subset) != 2160:
                raise ValueError(f"Expected 2160 pairs for {defense} {metric}, got {len(subset)}")
            test_res = compute_t_test(np.array(subset))
            test_res["defense"] = defense
            test_res["metric"] = metric.replace("diff_", "")
            
            p = test_res["p_raw"]
            test_res["raw_decision"] = "significant" if p < 0.05 else "not_significant"
            if test_res["raw_decision"] == "significant":
                test_res["direction"] = "improved" if test_res["mean"] > 0 else "decreased"
            else:
                test_res["direction"] = "no_difference"
                
            results.append(test_res)
            p_raws.append(p)
            
    rej, p_holm, _, _ = multipletests(p_raws, alpha=0.05, method="holm")
    for i, r in enumerate(results):
        r["p_holm"] = float(p_holm[i])
        r["holm_decision"] = "significant" if rej[i] else "not_significant"
        
    return results

def run_supplementary_run_level():
    path = ANALYSIS_DIR / "primary_paired_run_differences.csv"
    data = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
            
    results = []
    p_raws = []
    defenses = ["afp", "randomized_smoothing", "feature_squeezing"]
    metrics = ["diff_precision", "diff_recall", "diff_f1_score"]
    
    for defense in defenses:
        for metric in metrics:
            subset = [float(r[metric]) for r in data if r["defense_mechanism"] == defense]
            if len(subset) != 15:
                raise ValueError(f"Expected 15 pairs for {defense} {metric}, got {len(subset)}")
                
            test_res = compute_t_test(np.array(subset))
            test_res["defense"] = defense
            test_res["metric"] = metric.replace("diff_", "")
            
            p = test_res["p_raw"]
            test_res["raw_decision"] = "significant" if p < 0.05 else "not_significant"
            if test_res["raw_decision"] == "significant":
                test_res["direction"] = "improved" if test_res["mean"] > 0 else "decreased"
            else:
                test_res["direction"] = "no_difference"
                
            subset_arr = np.array(subset)
            if np.all(subset_arr == subset_arr[0]):
                test_res["shapiro_w"] = None
                test_res["shapiro_p"] = None
                test_res["wilcoxon_stat"] = None
                test_res["wilcoxon_p"] = None
            else:
                try:
                    sw_stat, sw_p = stats.shapiro(subset_arr)
                    test_res["shapiro_w"] = float(sw_stat)
                    test_res["shapiro_p"] = float(sw_p)
                except Exception:
                    test_res["shapiro_w"] = None
                    test_res["shapiro_p"] = None
                    
                try:
                    w_stat, w_p = stats.wilcoxon(subset_arr)
                    test_res["wilcoxon_stat"] = float(w_stat)
                    test_res["wilcoxon_p"] = float(w_p)
                except ValueError: 
                    test_res["wilcoxon_stat"] = None
                    test_res["wilcoxon_p"] = None

            results.append(test_res)
            p_raws.append(p)
            
    rej, p_holm, _, _ = multipletests(p_raws, alpha=0.05, method="holm")
    for i, r in enumerate(results):
        r["p_holm"] = float(p_holm[i])
        r["holm_decision"] = "significant" if rej[i] else "not_significant"
        r["is_supplementary"] = True
        
    return results

def compute_descriptive_stats(path, group_cols, metrics):
    data = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
            
    # Grouping
    groups = {}
    for row in data:
        key = tuple(row[k] for k in group_cols)
        if key not in groups:
            groups[key] = {m: [] for m in metrics}
        for m in metrics:
            groups[key][m].append(float(row[m]))
            
    results = []
    for key, values in groups.items():
        res = dict(zip(group_cols, key))
        for m in metrics:
            arr = np.array(values[m])
            res[f"{m}_mean"] = float(np.mean(arr))
            res[f"{m}_std"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            res[f"{m}_min"] = float(np.min(arr))
            res[f"{m}_max"] = float(np.max(arr))
            res[f"{m}_n"] = len(arr)
        results.append(res)
    return results

def write_csv(path, data):
    if not data:
        return
    with open(path, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        writer.writeheader()
        for row in data:
            writer.writerow(row)

def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def main():
    print("Verifying upstream artifact hashes...")
    verify_table_hashes()
    
    print("Running primary inference...")
    primary_results = run_primary_inference()
    write_csv(ANALYSIS_DIR / "phase11b_primary_inference.csv", primary_results)
    write_json(ANALYSIS_DIR / "phase11b_primary_inference.json", primary_results)
    
    print("Running supplementary run-level analysis...")
    supp_results = run_supplementary_run_level()
    write_csv(ANALYSIS_DIR / "phase11b_run_level_supplementary.csv", supp_results)
    write_json(ANALYSIS_DIR / "phase11b_run_level_supplementary.json", supp_results)
    
    print("Computing RQ1/RQ2 descriptive stats...")
    rq1_rq2_results = compute_descriptive_stats(
        ANALYSIS_DIR / "primary_batch_level.csv",
        ["defense_mechanism", "controller_config", "attack_scenario"],
        ["precision", "recall", "f1_score"]
    )
    write_csv(ANALYSIS_DIR / "phase11b_rq1_rq2_descriptive.csv", rq1_rq2_results)
    
    print("Computing RQ4 sensitivity stats...")
    rq4_results = compute_descriptive_stats(
        ANALYSIS_DIR / "sensitivity_batch_level.csv",
        ["defense_mechanism", "controller_config", "attack_scenario"],
        ["precision", "recall", "f1_score"]
    )
    write_csv(ANALYSIS_DIR / "phase11b_rq4_sensitivity_descriptive.csv", rq4_results)
    
    # Assumption diagnostics (from supplementary run level + perhaps primary?)
    # We will output a consolidated assumption diagnostics CSV containing the shapiro and wilcoxon from run_level
    diagnostics = []
    for r in supp_results:
        diagnostics.append({
            "defense": r["defense"],
            "metric": r["metric"],
            "level": "run_level",
            "shapiro_w": r.get("shapiro_w"),
            "shapiro_p": r.get("shapiro_p"),
            "wilcoxon_stat": r.get("wilcoxon_stat"),
            "wilcoxon_p": r.get("wilcoxon_p")
        })
    write_csv(ANALYSIS_DIR / "phase11b_assumption_diagnostics.csv", diagnostics)
    
    print("Writing statistical results report...")
    report_path = REPO_ROOT / "artifacts" / "reports" / "phase11b_statistical_results.md"
    with open(report_path, "w") as f:
        f.write("# Phase 11B Statistical Results\n\n")
        f.write("This report summarizes the locked Phase 11B statistical analysis.\n\n")
        
        f.write("## Important Disclosures\n")
        f.write("- Serial dependence is acknowledged at the batch level. The primary batch-level results are interpreted alongside the supplementary run-level results.\n")
        f.write("- No causal claims are made.\n")
        f.write("- No claims of universal IDS superiority are made.\n")
        f.write("- The methodology was fully prespecified and locked prior to this Phase 11B execution.\n\n")
        
        f.write("## Primary Batch-Level Results (n=2,160 pairs)\n")
        f.write("| Defense | Metric | Mean Diff | Raw p-value | Raw Decision | Holm p-value | Holm Decision |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in primary_results:
            f.write(f"| {r['defense']} | {r['metric']} | {r['mean']:.4f} | {r['p_raw']:.4e} | {r['raw_decision']} | {r['p_holm']:.4e} | {r['holm_decision']} |\n")
            
        f.write("\n## Supplementary Run-Level Results (n=15 pairs)\n")
        f.write("| Defense | Metric | Mean Diff | Raw p-value | Raw Decision | Holm p-value | Holm Decision |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in supp_results:
            f.write(f"| {r['defense']} | {r['metric']} | {r['mean']:.4f} | {r['p_raw']:.4e} | {r['raw_decision']} | {r['p_holm']:.4e} | {r['holm_decision']} |\n")
            
    exec_path = REPO_ROOT / "docs" / "PHASE11B_ANALYSIS_EXECUTION.md"
    with open(exec_path, "w") as f:
        f.write("# Phase 11B Analysis Execution\n\n")
        f.write("The locked analysis was successfully executed, producing strictly bound Phase 11B inferential outcomes.\n")
        f.write("The primary inference is bounded to 9 tests; Holm adjustment is applied independently to both families.\n")
        
    print("Phase 11B execution completed successfully.")

if __name__ == "__main__":
    main()

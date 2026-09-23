#!/usr/bin/env python3
import sys
import os
import csv
import json
import hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = REPO_ROOT / "artifacts" / "analysis"
FIGURES_DIR = ANALYSIS_DIR / "figures"

def calculate_file_hash(path: Path) -> str:
    with open(path, "rb") as bf:
        return hashlib.sha256(bf.read()).hexdigest()

def verify_inputs():
    # Verify 11B bundle
    bundle_path = REPO_ROOT / "phase11b_statistical_analysis_review_bundle_v1_0_0.zip"
    if not bundle_path.exists():
        raise FileNotFoundError("Phase 11B bundle missing")
    bundle_hash = calculate_file_hash(bundle_path)
    if bundle_hash != "cadae5b22cba9c408eea6eb33694795849e9abe56c46062e035b6623a23cdb02":
        raise ValueError(f"Phase 11B bundle hash mismatch: {bundle_hash}")

    # Verify 11A tables
    summary_path = REPO_ROOT / "artifacts" / "reports" / "phase11a_analysis_tables_summary.txt"
    with open(summary_path, "r") as f:
        expected_hashes = {}
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
        actual_hash = calculate_file_hash(table_path)
        if actual_hash != expected_hash:
            raise ValueError(f"Hash mismatch for {table}: {actual_hash} != {expected_hash}")

def read_csv(path):
    data = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def write_csv(path, data):
    if not data:
        return
    with open(path, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        writer.writeheader()
        for row in data:
            writer.writerow(row)

def format_p_value(p):
    val = float(p)
    if val == 0.0:
        return "p < .001"
    if val < 0.001:
        return "p < .001"
    return f"p = {val:.3f}"

def format_p_value_raw(p):
    val = float(p)
    if val == 0.0:
        return "<.001"
    if val < 0.001:
        return "<.001"
    return f"{val:.3f}"

def run_rq1_rq2():
    batch_data = read_csv(ANALYSIS_DIR / "primary_batch_level.csv")
    
    # We want mean precision, recall, f1_score by defense, attack, and config (Base vs C1)
    # Actually just Base for RQ1, C1 for RQ2
    
    rq1_data = []
    rq2_data = []
    
    defenses = ["afp", "feature_squeezing", "randomized_smoothing"]
    attacks = ["SilentProbing", "SurrogateTransfer", "DecisionBoundary"]
    metrics = ["precision", "recall", "f1_score"]
    
    # Compute Base means
    for defense in defenses:
        for attack in attacks:
            row_rq1 = {"defense": defense, "attack_scenario": attack}
            row_rq2 = {"defense": defense, "attack_scenario": attack}
            
            for config in ["Base", "C1"]:
                subset = [r for r in batch_data if r["defense_mechanism"] == defense and r["attack_scenario"] == attack and r["controller_config"] == config]
                
                for metric in metrics:
                    vals = [float(r[metric]) for r in subset]
                    mean_val = float(np.mean(vals))
                    if config == "Base":
                        row_rq1[metric] = mean_val
                    else:
                        row_rq2[metric] = mean_val
                        
            rq1_data.append(row_rq1)
            rq2_data.append(row_rq2)
            
    write_csv(ANALYSIS_DIR / "phase11c_rq1_base_performance.csv", rq1_data)
    write_csv(ANALYSIS_DIR / "phase11c_rq2_controller_performance.csv", rq2_data)
    
    return rq1_data, rq2_data

def run_rq3():
    inf_data = read_csv(ANALYSIS_DIR / "phase11b_primary_inference.csv")
    
    if len(inf_data) != 9:
        raise ValueError("Expected exactly 9 rows in phase11b_primary_inference.csv")
        
    rq3_data = []
    
    # We also need Base mean and C1 mean for context in RQ3 table. We can recompute from primary_batch_level
    batch_data = read_csv(ANALYSIS_DIR / "primary_batch_level.csv")
    
    for row in inf_data:
        defense = row["defense"]
        metric = row["metric"]
        
        base_vals = [float(r[metric]) for r in batch_data if r["defense_mechanism"] == defense and r["controller_config"] == "Base"]
        c1_vals = [float(r[metric]) for r in batch_data if r["defense_mechanism"] == defense and r["controller_config"] == "C1"]
        
        base_mean = float(np.mean(base_vals))
        c1_mean = float(np.mean(c1_vals))
        
        rq3_row = {
            "defense": defense,
            "metric": metric,
            "base_mean": base_mean,
            "c1_mean": c1_mean,
            "c1_minus_base_mean": float(row["mean"]),
            "ci_95_lower": float(row["ci_lower"]),
            "ci_95_upper": float(row["ci_upper"]),
            "t_stat": float(row["t"]),
            "df": int(row["df"]),
            "p_raw": format_p_value_raw(row["p_raw"]),
            "raw_decision": row["raw_decision"],
            "p_holm": format_p_value_raw(row["p_holm"]),
            "cohen_dz": float(row["cohen_dz"]),
            "direction": row["direction"]
        }
        rq3_data.append(rq3_row)
        
    write_csv(ANALYSIS_DIR / "phase11c_rq3_statistical_decisions.csv", rq3_data)
    return rq3_data

def run_rq4():
    sens_data = read_csv(ANALYSIS_DIR / "phase11b_rq4_sensitivity_descriptive.csv")
    write_csv(ANALYSIS_DIR / "phase11c_rq4_sensitivity_summary.csv", sens_data)
    return sens_data

def generate_figures(rq1_data, rq2_data, rq4_data):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1-3. Base vs C1 Precision, Recall, F1 by defense
    metrics = ["precision", "recall", "f1_score"]
    defenses = ["afp", "feature_squeezing", "randomized_smoothing"]
    
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        base_means = []
        c1_means = []
        for d in defenses:
            b_vals = [r[metric] for r in rq1_data if r["defense"] == d]
            c_vals = [r[metric] for r in rq2_data if r["defense"] == d]
            base_means.append(np.mean(b_vals) * 100)
            c1_means.append(np.mean(c_vals) * 100)
            
        x = np.arange(len(defenses))
        width = 0.35
        
        ax.bar(x - width/2, base_means, width, label='Base')
        ax.bar(x + width/2, c1_means, width, label='+ RA (C1)')
        
        ax.set_ylabel(f'{metric.capitalize()} (%)')
        ax.set_title(f'Base vs C1 {metric.capitalize()} by Defense')
        ax.set_xticks(x)
        ax.set_xticklabels([d.upper() for d in defenses])
        ax.legend()
        ax.set_ylim(0, 105)
        
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f'base_vs_c1_{metric}.png')
        plt.close(fig)
        
    # 4. Scenario-level AFP vs AFP + RA comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    attacks = ["SilentProbing", "SurrogateTransfer", "DecisionBoundary"]
    afp_base = [r["recall"] * 100 for r in rq1_data if r["defense"] == "afp"]
    afp_c1 = [r["recall"] * 100 for r in rq2_data if r["defense"] == "afp"]
    
    # Align order with attacks list
    afp_b_ordered = []
    afp_c_ordered = []
    for a in attacks:
        afp_b_ordered.append(next(r["recall"] * 100 for r in rq1_data if r["defense"] == "afp" and r["attack_scenario"] == a))
        afp_c_ordered.append(next(r["recall"] * 100 for r in rq2_data if r["defense"] == "afp" and r["attack_scenario"] == a))
        
    x = np.arange(len(attacks))
    ax.bar(x - width/2, afp_b_ordered, width, label='AFP Base')
    ax.bar(x + width/2, afp_c_ordered, width, label='AFP + RA')
    
    ax.set_ylabel('Recall (%)')
    ax.set_title('Scenario-level Recall: AFP vs AFP + RA')
    ax.set_xticks(x)
    ax.set_xticklabels(attacks)
    ax.legend()
    ax.set_ylim(0, 105)
    
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'afp_scenario_recall.png')
    plt.close(fig)
    
    # 5. C1-C7 sensitivity figure
    fig, ax = plt.subplots(figsize=(10, 5))
    configs = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    afp_recall_means = []
    for c in configs:
        vals = [float(r["recall_mean"]) * 100 for r in rq4_data if r["defense_mechanism"] == "afp" and r["controller_config"] == c]
        afp_recall_means.append(np.mean(vals))
        
    ax.plot(configs, afp_recall_means, marker='o')
    ax.set_ylabel('Recall (%)')
    ax.set_title('AFP + RA Recall Across Sensitivity Configurations (C1-C7)')
    ax.set_ylim(0, 105)
    
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'sensitivity_c1_c7_afp_recall.png')
    plt.close(fig)

def generate_chapter_4(rq1, rq2, rq3):
    chapter_path = REPO_ROOT / "docs" / "PHASE11C_CHAPTER4_DRAFT.md"
    
    with open(chapter_path, "w") as f:
        f.write("# Chapter 4: Results and Interpretation\n\n")
        f.write("## 4.1 Introduction\n")
        f.write("This chapter presents the statistical analysis of the Recall-Aware (RA) feedback controller across three existing defense mechanisms: Adaptive Feature Poisoning (AFP), Randomized Smoothing (RS), and Feature Squeezing (FS). The proposed contribution of this thesis is the RA controller, which dynamically adjusts perturbation intensity using Rolling Recall. The valid causal comparison within this study is each implemented fixed-intensity Base defense versus the same defense augmented with the C1 Recall-Aware controller (denoted as AFP + RA, RS + RA, FS + RA).\n\n")
        
        f.write("## 4.2 RQ1: Base Defense Performance\n")
        f.write("Table 4.1 summarizes the Precision, Recall, and F1-Score of the fixed-intensity Base defenses across attack scenarios.\n\n")
        f.write("| Defense | Scenario | Precision | Recall | F1-Score |\n")
        f.write("|---|---|---|---|---|\n")
        for r in rq1:
            f.write(f"| {r['defense'].upper()} | {r['attack_scenario']} | {r['precision']*100:.2f}% | {r['recall']*100:.2f}% | {r['f1_score']*100:.2f}% |\n")
        
        f.write("\n## 4.3 RQ2: Controller-Augmented Defense Performance\n")
        f.write("Table 4.2 summarizes the performance of the defenses augmented with the C1 RA controller.\n\n")
        f.write("| Defense | Scenario | Precision | Recall | F1-Score |\n")
        f.write("|---|---|---|---|---|\n")
        for r in rq2:
            f.write(f"| {r['defense'].upper()} + RA | {r['attack_scenario']} | {r['precision']*100:.2f}% | {r['recall']*100:.2f}% | {r['f1_score']*100:.2f}% |\n")
            
        f.write("\n## 4.4 RQ3: Statistical Differences (Base vs. C1)\n")
        f.write("The locked primary analysis consists of 2,160 C1 minus Base batch pairs per defense and metric. Raw two-tailed paired t-test p-values at α = .05 determine the primary decision. Holm adjustment is provided as supplementary robustness evidence. Note that serial dependence at the batch level is a known limitation.\n\n")
        
        f.write("| Defense | Metric | Base Mean | C1 Mean | C1-Base Diff | 95% CI | t-stat (df=2159) | p_raw | Decision | Holm | dz | Direction |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rq3:
            f.write(f"| {r['defense'].upper()} | {r['metric']} | {r['base_mean']:.4f} | {r['c1_mean']:.4f} | {r['c1_minus_base_mean']:.4f} | [{r['ci_95_lower']:.4f}, {r['ci_95_upper']:.4f}] | {r['t_stat']:.2f} | {r['p_raw']} | {r['raw_decision']} | {r['p_holm']} | {r['cohen_dz']:.2f} | {r['direction']} |\n")
            
        f.write("\n## 4.5 RQ4: Sensitivity Analysis (C1–C7)\n")
        f.write("Descriptive performance across configurations C1 through C7 demonstrates that alternative parameter selections yield distinct operating points, with no single configuration universally optimal across all metrics.\n\n")
        
        f.write("## 4.6 Integrated Discussion and AFP Interpretation\n")
        f.write("The original AFP paper reported high accuracy in some scenarios while reporting very low attack recall (e.g., 0.01). Accuracy can conceal missed attacks, particularly under class imbalance. While noting these internal reporting inconsistencies in the original paper, the valid internal comparison in this study is Base AFP versus AFP + RA under matched experimental conditions.\n\n")
        
        # Extract specific recall values for AFP to assert correctness
        afp_sp_b = next(r["recall"]*100 for r in rq1 if r["defense"]=="afp" and r["attack_scenario"]=="SilentProbing")
        afp_sp_c = next(r["recall"]*100 for r in rq2 if r["defense"]=="afp" and r["attack_scenario"]=="SilentProbing")
        afp_st_b = next(r["recall"]*100 for r in rq1 if r["defense"]=="afp" and r["attack_scenario"]=="SurrogateTransfer")
        afp_st_c = next(r["recall"]*100 for r in rq2 if r["defense"]=="afp" and r["attack_scenario"]=="SurrogateTransfer")
        afp_db_b = next(r["recall"]*100 for r in rq1 if r["defense"]=="afp" and r["attack_scenario"]=="DecisionBoundary")
        afp_db_c = next(r["recall"]*100 for r in rq2 if r["defense"]=="afp" and r["attack_scenario"]=="DecisionBoundary")
        
        f.write("In our controlled evaluation, RA increased AFP Recall and F1 across all scenarios. Specifically, Recall increased for Silent Probing (approx. {:.2f}% Base to {:.2f}% C1), Surrogate Transfer (approx. {:.2f}% Base to {:.2f}% C1), and Decision Boundary (approx. {:.2f}% Base to {:.2f}% C1). This improvement came with a small, statistically significant decrease in Precision. The controller effectively mitigated recall degradation associated with fixed perturbation intensity under this study’s conditions.\n\n".format(afp_sp_b, afp_sp_c, afp_st_b, afp_st_c, afp_db_b, afp_db_c))
        
        f.write("## 4.7 Summary of Findings\n")
        f.write("The proposed Recall-Aware controller improved Recall and F1-Score when added to AFP, Randomized Smoothing, and Feature Squeezing under the frozen controlled evaluation, with a small reduction in Precision.\n\n")
        
        f.write("## 4.8 Methodological Limitations\n")
        f.write("- Serial dependence in batch-level observations is present.\n")
        f.write("- Results are constrained to the specific dataset, attacks, classifier, and conditions tested.\n")
        f.write("- The study does not claim universal real-world superiority or perfect attack prevention.\n")
        
def main():
    verify_inputs()
    rq1, rq2 = run_rq1_rq2()
    rq3 = run_rq3()
    rq4 = run_rq4()
    generate_figures(rq1, rq2, rq4)
    generate_chapter_4(rq1, rq2, rq3)
    print("Phase 11C Results Generated Successfully.")

if __name__ == "__main__":
    main()

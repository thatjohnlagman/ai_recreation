import re
import csv
import json
from collections import defaultdict

# 1. Base defense performance
base_perf = {}
with open(".research_repo/artifacts/analysis/phase11c_rq1_base_performance.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        d = row['defense']
        if d == 'feature_squeezing': d = 'fs'
        elif d == 'randomized_smoothing': d = 'rs'
        base_perf[(d, row['attack_scenario'])] = {
            'precision': float(row['precision']),
            'recall': float(row['recall']),
            'f1': float(row['f1_score'])
        }

# 2. C1 performance
c1_perf = {}
with open(".research_repo/artifacts/analysis/phase11c_rq2_controller_performance.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        d = row['defense']
        if d == 'feature_squeezing': d = 'fs'
        elif d == 'randomized_smoothing': d = 'rs'
        c1_perf[(d, row['attack_scenario'])] = {
            'precision': float(row['precision']),
            'recall': float(row['recall']),
            'f1': float(row['f1_score'])
        }

# 3. Pooled Base vs C1 summary
pooled_stats = {}
with open(".research_repo/artifacts/analysis/phase11c_rq3_statistical_decisions.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        d = row['defense']
        if d == 'feature_squeezing': d = 'fs'
        elif d == 'randomized_smoothing': d = 'rs'
        if d not in pooled_stats:
            pooled_stats[d] = {}
        pooled_stats[d][row['metric']] = {
            'base': float(row['base_mean']),
            'c1': float(row['c1_mean']),
            'diff': float(row['c1_minus_base_mean'])
        }

# 4. C1-C7 descriptive results
c_n_sums = defaultdict(lambda: defaultdict(lambda: {'precision':0, 'recall':0, 'f1':0}))
c_weighted_sums = defaultdict(lambda: defaultdict(lambda: {'precision':0, 'recall':0, 'f1':0}))
with open(".research_repo/artifacts/analysis/phase11c_rq4_sensitivity_summary.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        d = row['defense_mechanism']
        if d == 'feature_squeezing': d = 'fs'
        elif d == 'randomized_smoothing': d = 'rs'
        c = row['controller_config'] # C1, C2, etc
        
        pn = int(row['precision_n'])
        rn = int(row['recall_n'])
        fn = int(row['f1_score_n'])
        
        c_weighted_sums[d][c]['precision'] += float(row['precision_mean']) * pn
        c_weighted_sums[d][c]['recall'] += float(row['recall_mean']) * rn
        c_weighted_sums[d][c]['f1'] += float(row['f1_score_mean']) * fn
        
        c_n_sums[d][c]['precision'] += pn
        c_n_sums[d][c]['recall'] += rn
        c_n_sums[d][c]['f1'] += fn

c2_c7_pooled = {}
for d in c_weighted_sums:
    if d not in c2_c7_pooled: c2_c7_pooled[d] = {}
    for c in c_weighted_sums[d]:
        c2_c7_pooled[d][c] = {
            'precision': c_weighted_sums[d][c]['precision'] / c_n_sums[d][c]['precision'],
            'recall': c_weighted_sums[d][c]['recall'] / c_n_sums[d][c]['recall'],
            'f1': c_weighted_sums[d][c]['f1'] / c_n_sums[d][c]['f1']
        }

def format_pct(val):
    return f"{val*100:.2f}%"

def format_pp(val):
    return f"{'+' if val>0 else ''}{val*100:.2f} pp"

summary_str = "const STATIC_BENCHMARK_SUMMARY = [\n"
for d in ["afp", "rs", "fs"]:
    if d not in pooled_stats: continue
    s = pooled_stats[d]
    obj = {
        "defense": d.upper(),
        "base_prec": format_pct(s['precision']['base']),
        "base_f1": format_pct(s['f1_score']['base']),
        "base_recall": format_pct(s['recall']['base']),
        "ra_prec": format_pct(s['precision']['c1']),
        "ra_f1": format_pct(s['f1_score']['c1']),
        "ra_recall": format_pct(s['recall']['c1']),
        "delta_prec": format_pp(s['precision']['diff']),
        "delta_f1": format_pp(s['f1_score']['diff']),
        "delta_recall": format_pp(s['recall']['diff']),
        "evasions_prevented": "-",
        "intensity_shift": "-",
        "controller_state": "N/A (Historical)"
    }
    summary_str += "  " + json.dumps(obj) + ",\n"
summary_str += "];"

matrix_str = "const STATIC_C1_C7_MATRIX = [\n"
configs = {
    "C1": "growth_factor=1.05",
    "C2": "window=3",
    "C3": "window=20",
    "C4": "Rmin=0.90",
    "C5": "Rmin=0.97",
    "C6": "growth_factor=1.02",
    "C7": "growth_factor=1.10"
}
for d in ["afp", "rs", "fs"]:
    s = pooled_stats[d]
    
    # Base
    obj_base = {
        "defense": d,
        "mode": "Base",
        "precision_str": format_pct(s['precision']['base']),
        "f1_str": format_pct(s['f1_score']['base']),
        "recall_str": format_pct(s['recall']['base']),
        "research_config": "frozen_base",
        "tp": 0, "fn": 0, "fp": 0, "tn": 0, "fpr_str": "-", "intensity": "-", "state": "N/A"
    }
    matrix_str += "  " + json.dumps(obj_base) + ",\n"

    # C1
    obj_c1 = {
        "defense": d,
        "mode": "C1",
        "precision_str": format_pct(s['precision']['c1']),
        "f1_str": format_pct(s['f1_score']['c1']),
        "recall_str": format_pct(s['recall']['c1']),
        "research_config": configs["C1"],
        "tp": 0, "fn": 0, "fp": 0, "tn": 0, "fpr_str": "-", "intensity": "-", "state": "N/A"
    }
    matrix_str += "  " + json.dumps(obj_c1) + ",\n"
    
    # C2-C7
    for c in ["C2", "C3", "C4", "C5", "C6", "C7"]:
        s2 = c2_c7_pooled.get(d, {}).get(c)
        if not s2: continue
        obj = {
            "defense": d,
            "mode": c,
            "precision_str": format_pct(s2['precision']),
            "f1_str": format_pct(s2['f1']),
            "recall_str": format_pct(s2['recall']),
            "research_config": configs[c],
            "tp": 0, "fn": 0, "fp": 0, "tn": 0, "fpr_str": "-", "intensity": "-", "state": "N/A"
        }
        matrix_str += "  " + json.dumps(obj) + ",\n"
matrix_str += "];"

with open("frontend/results.js", "r") as f:
    content = f.read()

# Replace STATIC_BENCHMARK_SUMMARY
content = re.sub(r'const STATIC_BENCHMARK_SUMMARY = \[.*?\];', summary_str, content, flags=re.DOTALL)

# Replace STATIC_C1_C7_MATRIX
content = re.sub(r'const STATIC_C1_C7_MATRIX = \[.*?\];', matrix_str, content, flags=re.DOTALL)

with open("frontend/results.js", "w") as f:
    f.write(content)

print("Updated frontend/results.js")

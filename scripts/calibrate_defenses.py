import json
import yaml
import hashlib
import time
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recall_aware_ids.defenses.afp import AdaptiveFeaturePoisoning
from recall_aware_ids.defenses.randomized_smoothing import RandomizedSmoothing
from recall_aware_ids.defenses.feature_squeezing import FeatureSqueezing

import resource

def hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def get_peak_ram():
    # ru_maxrss is in bytes on macOS
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

def run_calibration():
    start_time = time.time()
    
    # Files
    rf_path = PROJECT_ROOT / "artifacts" / "models" / "frozen_rf.joblib"
    rf_hash_before = hash_file(rf_path)
    
    x_cal_path = PROJECT_ROOT / "data" / "processed" / "X_calibration.parquet"
    meta_cal_path = PROJECT_ROOT / "data" / "processed" / "metadata_calibration.parquet"
    meta_train_path = PROJECT_ROOT / "data" / "processed" / "metadata_train.parquet"
    meta_eval_path = PROJECT_ROOT / "data" / "processed" / "metadata_eval.parquet"
    
    feature_names_path = PROJECT_ROOT / "artifacts" / "preprocessors" / "feature_names.json"
    feature_mask_path = PROJECT_ROOT / "artifacts" / "preprocessors" / "feature_mask.json"
    bounds_path = PROJECT_ROOT / "artifacts" / "preprocessors" / "training_bounds.parquet"
    profile_path = PROJECT_ROOT / "artifacts" / "preprocessors" / "afp_benign_profile.parquet"
    defenses_yaml_path = PROJECT_ROOT / "configs" / "defenses.yaml"
    
    # 1. Preflight Validation
    X_cal = pd.read_parquet(x_cal_path)
    meta_cal = pd.read_parquet(meta_cal_path)
    
    if len(X_cal) != 31500:
        raise ValueError("X_calibration must be 31500 rows")
    if len(meta_cal) != 31500:
        raise ValueError("metadata_calibration must be 31500 rows")
    if list(X_cal.index) != list(meta_cal.index):
        raise ValueError("Indices must align")
    if X_cal.shape[1] != 78:
        raise ValueError("Must have 78 features")
    if not np.isfinite(X_cal.values).all():
        raise ValueError("Must have zero NaN/Inf")
    
    # Disjoint check
    meta_train = pd.read_parquet(meta_train_path)
    meta_eval = pd.read_parquet(meta_eval_path)
    
    cal_keys = set(zip(meta_cal['_source_file'], meta_cal['_raw_row_idx']))
    train_keys = set(zip(meta_train['_source_file'], meta_train['_raw_row_idx']))
    eval_keys = set(zip(meta_eval['_source_file'], meta_eval['_raw_row_idx']))
    
    if not cal_keys.isdisjoint(train_keys):
        raise ValueError("Calibration not disjoint from Train")
    if not cal_keys.isdisjoint(eval_keys):
        raise ValueError("Calibration not disjoint from Eval")
    
    # Load RF
    rf = joblib.load(rf_path)
    
    # Load configs
    with open(feature_names_path, "r") as f:
        feature_names = json.load(f)
    with open(feature_mask_path, "r") as f:
        feature_mask_data = json.load(f)
        feature_mask = feature_mask_data["feature_mask"]
        
    if len(feature_names) != 78:
        raise ValueError("feature_names must have 78 items")
    if len(feature_mask) != 78:
        raise ValueError("feature_mask must have 78 items")
    if sum(feature_mask) != 63:
        raise ValueError("feature_mask must have exactly 63 eligible features")
    if list(X_cal.columns) != feature_names:
        raise ValueError("X_calibration columns must perfectly match feature_names")
    
    bounds = pd.read_parquet(bounds_path)
    profile = pd.read_parquet(profile_path)
    
    if not np.isfinite(bounds[["train_min", "train_max"]].values).all():
        raise ValueError("Bounds must be finite")
    
    with open(defenses_yaml_path, "r") as f:
        defenses_cfg = yaml.safe_load(f)
        
    # Baseline
    y_true = meta_cal['y_binary'].values
    y_pred_base = rf.predict(X_cal.values)
    
    tp_base = int(np.sum((y_pred_base == 1) & (y_true == 1)))
    fn_base = int(np.sum((y_pred_base == 0) & (y_true == 1)))
    fp_base = int(np.sum((y_pred_base == 1) & (y_true == 0)))
    tn_base = int(np.sum((y_pred_base == 0) & (y_true == 0)))
    recall_base = tp_base / (tp_base + fn_base) if (tp_base + fn_base) > 0 else 0.0
    
    results = {
        "baseline": {
            "TP": tp_base, "FN": fn_base, "FP": fp_base, "TN": tn_base, "recall": recall_base
        },
        "candidates": [],
        "selected": {}
    }
    
    phase8_success = True
    print(f"BASELINE: {results['baseline']}")
    
    X_val = X_cal.values.astype(np.float32)
    
    # Baseline assertion
    if tp_base != 5085 or fn_base != 277 or fp_base != 33 or tn_base != 26105:
        raise ValueError("Baseline confusion matrix mismatch! Stopping calibration.")
        
    # AFP Calibration
    afp = AdaptiveFeaturePoisoning(feature_names, feature_mask, bounds, profile)
    afp_cands = []
    
    for eps in defenses_cfg["afp"]["epsilon_base_grid"]:
        for alpha in defenses_cfg["afp"]["alpha_grid"]:
            X_def, res, mean_eps = afp.defend(X_val, epsilon_base=eps, alpha=alpha, seed=42, attack_scenario="calibration", batch_id=0)
            
            y_pred = rf.predict(X_def)
            tp = int(np.sum((y_pred == 1) & (y_true == 1)))
            fn = int(np.sum((y_pred == 0) & (y_true == 1)))
            fp = int(np.sum((y_pred == 1) & (y_true == 0)))
            tn = int(np.sum((y_pred == 0) & (y_true == 0)))
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            valid = (recall >= 0.80) and (res.final_invalid_fraction <= 0.05)
            
            afp_cands.append({
                "defense": "afp",
                "epsilon_base": eps,
                "alpha": alpha,
                "mean_epsilon_i": mean_eps,
                "TN": tn, "FP": fp, "FN": fn, "TP": tp,
                "recall": recall,
                "proposal_out_of_bounds_count": res.proposal_out_of_bounds_count,
                "proposal_out_of_bounds_fraction": res.proposal_out_of_bounds_fraction,
                "projection_unit_count": res.projection_unit_count,
                "projection_unit_fraction": res.projection_unit_fraction,
                "projected_cell_count": res.projected_cell_count,
                "final_nan_count": res.final_nan_count,
                "final_inf_count": res.final_inf_count,
                "final_bounds_violation_count": res.final_bounds_violation_count,
                "final_invalid_fraction": res.final_invalid_fraction,
                "protected_feature_modification_count": res.protected_feature_modification_count,
                "valid": valid
            })
            
    afp_valid = [c for c in afp_cands if c["valid"]]
    if not afp_valid:
        print("No valid AFP candidate found!")
        afp_selected = None
        phase8_success = False
    else:
        afp_valid.sort(key=lambda x: (x["mean_epsilon_i"], x["epsilon_base"], x["alpha"]), reverse=True)
        afp_selected = afp_valid[0]
        
    results["selected"]["afp"] = afp_selected
    
    # RS Calibration
    rs = RandomizedSmoothing(feature_names, feature_mask, bounds, ensemble_size=11)
    rs_cands = []
    
    rs_grid = defenses_cfg["randomized_smoothing"]["calibration_grid"]
    for sig in rs_grid["sigma"]:
        y_pred, res = rs.predict_ensemble(X_val, sigma=sig, seed=42, attack_scenario="calibration", batch_id=0, predict_func=rf.predict, chunk_size=500)
        
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        valid = (recall >= 0.80) and (res.final_invalid_fraction <= 0.05)
        
        rs_cands.append({
            "defense": "rs",
            "sigma": sig,
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "recall": recall,
            "proposal_out_of_bounds_count": res.proposal_out_of_bounds_count,
            "proposal_out_of_bounds_fraction": res.proposal_out_of_bounds_fraction,
            "projection_unit_count": res.projection_unit_count,
            "projection_unit_fraction": res.projection_unit_fraction,
            "projected_cell_count": res.projected_cell_count,
            "final_nan_count": res.final_nan_count,
            "final_inf_count": res.final_inf_count,
            "final_bounds_violation_count": res.final_bounds_violation_count,
            "final_invalid_fraction": res.final_invalid_fraction,
            "protected_feature_modification_count": res.protected_feature_modification_count,
            "valid": valid
        })
        
    rs_valid = [c for c in rs_cands if c["valid"]]
    if not rs_valid:
        print("No valid RS candidate found!")
        rs_selected = None
        phase8_success = False
    else:
        rs_valid.sort(key=lambda x: x["sigma"], reverse=True)
        rs_selected = rs_valid[0]
        
    results["selected"]["rs"] = rs_selected
    
    # FS Calibration
    fs = FeatureSqueezing(feature_names, feature_mask, bounds)
    fs_cands = []
    
    fs_grid = defenses_cfg["feature_squeezing"]["calibration_grid"]
    for intens in fs_grid["squeezing_intensity"]:
        X_def, res, eff_d = fs.defend(X_val, intensity=intens, seed=42, attack_scenario="calibration", batch_id=0)
        
        y_pred = rf.predict(X_def)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        valid = (recall >= 0.80) and (res.final_invalid_fraction <= 0.05)
        
        fs_cands.append({
            "defense": "fs",
            "squeezing_intensity": intens,
            "effective_d": eff_d,
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "recall": recall,
            "proposal_out_of_bounds_count": res.proposal_out_of_bounds_count,
            "proposal_out_of_bounds_fraction": res.proposal_out_of_bounds_fraction,
            "projection_unit_count": res.projection_unit_count,
            "projection_unit_fraction": res.projection_unit_fraction,
            "projected_cell_count": res.projected_cell_count,
            "final_nan_count": res.final_nan_count,
            "final_inf_count": res.final_inf_count,
            "final_bounds_violation_count": res.final_bounds_violation_count,
            "final_invalid_fraction": res.final_invalid_fraction,
            "protected_feature_modification_count": res.protected_feature_modification_count,
            "valid": valid
        })
        
    fs_valid = [c for c in fs_cands if c["valid"]]
    if not fs_valid:
        print("No valid FS candidate found!")
        fs_selected = None
        phase8_success = False
    else:
        fs_valid.sort(key=lambda x: x["squeezing_intensity"], reverse=True)
        fs_selected = fs_valid[0]
        
    results["selected"]["fs"] = fs_selected
    
    results["candidates"] = afp_cands + rs_cands + fs_cands
    
    # Hashes and Meta
    rf_hash_after = hash_file(rf_path)
    if rf_hash_before != rf_hash_after:
        raise ValueError("RF was mutated!")
    
    results["hashes"] = {
        "frozen_rf": rf_hash_before,
        "x_calibration": hash_file(x_cal_path),
        "defenses_yaml_before": hash_file(defenses_yaml_path),
        "failed_json": "41058cb6c8fabdc1bf484826d071e6bf107735ba36d064d7055e3f3a34f31a4e",
        "failed_md": "99d2131f8fce9e4892479d0711888913f80ac9c9185374adda16d8c0a6f9dbb1"
    }
    
    results["runtime_seconds"] = time.time() - start_time
    results["peak_ram_mb"] = get_peak_ram()
    rf_hash_after = hash_file(rf_path)
    if rf_hash_before != rf_hash_after:
        raise ValueError(f"RF model was mutated during calibration! Before: {rf_hash_before}, After: {rf_hash_after}")
        
    results["date_frozen"] = "REMAINS_UNSET"
    
    results["phase8_success"] = phase8_success
    
    if phase8_success:
        # Update defenses.yaml
        defenses_cfg["afp"]["epsilon_base"] = results["selected"]["afp"]["epsilon_base"]
        defenses_cfg["afp"]["alpha"] = results["selected"]["afp"]["alpha"]
        defenses_cfg["afp"]["intensity_max"] = results["selected"]["afp"]["epsilon_base"]
        
        defenses_cfg["randomized_smoothing"]["sigma"] = results["selected"]["rs"]["sigma"]
        defenses_cfg["randomized_smoothing"]["intensity_max"] = results["selected"]["rs"]["sigma"]
        
        defenses_cfg["feature_squeezing"]["squeezing_intensity"] = results["selected"]["fs"]["squeezing_intensity"]
        defenses_cfg["feature_squeezing"]["intensity_max"] = results["selected"]["fs"]["squeezing_intensity"]
        
        with open(defenses_yaml_path, "w") as f:
            yaml.dump(defenses_cfg, f, sort_keys=False)
            
        results["hashes"]["defenses_yaml_after"] = hash_file(defenses_yaml_path)
    
    with open(PROJECT_ROOT / "artifacts" / "reports" / "defense_calibration.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save markdown report
    md_path = PROJECT_ROOT / "artifacts" / "reports" / "defense_calibration.md"
    with open(md_path, "w") as f:
        f.write("# Phase 8: Defense Calibration Report\n\n")
        if phase8_success:
            f.write("## Selected Parameters\n")
            f.write(f"- **AFP**: epsilon_base={results['selected']['afp']['epsilon_base']}, alpha={results['selected']['afp']['alpha']} (Recall: {results['selected']['afp']['recall']})\n")
            f.write(f"- **RS**: sigma={results['selected']['rs']['sigma']} (Recall: {results['selected']['rs']['recall']})\n")
            f.write(f"- **FS**: squeezing_intensity={results['selected']['fs']['squeezing_intensity']} (Recall: {results['selected']['fs']['recall']})\n\n")
        else:
            f.write("## Calibration Failed\n")
            f.write("No valid candidates found for one or more defenses. Phase 8 explicitly aborted.\n\n")
            
        f.write("## Baseline Metrics\n")
        f.write(f"- Recall: {results['baseline']['recall']}\n\n")
        
        f.write("## Performance Meta\n")
        f.write(f"- Runtime: {results['runtime_seconds']:.2f}s\n")
        f.write(f"- Peak RAM: {results['peak_ram_mb']:.2f} MB\n")
        f.write(f"- `experiment.date_frozen`: {results['date_frozen']}\n")
        
    print("Calibration complete. Results saved to artifacts/reports/defense_calibration.json and artifacts/reports/defense_calibration.md")
    if not phase8_success:
        print("PHASE 8 CALIBRATION FAILED: Constraints not met.")
        sys.exit(1)
        
if __name__ == "__main__":
    run_calibration()

import sys
import json
import time
import resource
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import yaml

PROJECT_ROOT = Path(__file__).parents[1]

# Make sure imports work
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from recall_aware_ids.defenses.afp import AdaptiveFeaturePoisoning
from recall_aware_ids.defenses.randomized_smoothing import RandomizedSmoothing
from recall_aware_ids.defenses.feature_squeezing import FeatureSqueezing
from recall_aware_ids.defenses.base import DefenseResult

def get_peak_ram():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

# Removed _generate_afp_proposal as it is now afp._generate_proposal

def _generate_rs_proposals(rs, X, sigma, seed, attack_scenario, batch_id):
    # Returns a list of 11 full matrices (since this is just diagnostic for clipping vs non-clipping)
    proposals = []
    for m in range(rs.ensemble_size):
        seed_int = rs.get_noise_seed(seed, attack_scenario, "rs", batch_id, ensemble_id=m)
        rng = np.random.RandomState(seed_int)
        noise = rng.normal(0.0, 1.0, size=X.shape).astype(np.float32) * sigma
        noise[:, rs.protected_mask] = 0.0
        proposals.append(X + noise)
    return proposals

def compute_metrics(y_true, y_pred, y_pred_baseline=None):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
    
    tpr = recall
    tnr = tn / (tn + fp) if tn + fp > 0 else 0.0
    balanced_accuracy = (tpr + tnr) / 2.0
    
    res = {
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy
    }
    
    if y_pred_baseline is not None:
        a2b = int(np.sum((y_pred_baseline == 1) & (y_pred == 0)))
        b2a = int(np.sum((y_pred_baseline == 0) & (y_pred == 1)))
        res["flips_a2b"] = a2b
        res["flips_b2a"] = b2a
        
    return res

def run_diagnostics():
    start_time = time.time()
    
    # Validation of loaded artifacts
    x_cal_path = PROJECT_ROOT / "data" / "processed" / "X_calibration.parquet"
    meta_cal_path = PROJECT_ROOT / "data" / "processed" / "metadata_calibration.parquet"
    rf_path = PROJECT_ROOT / "artifacts" / "models" / "frozen_rf.joblib"
    feat_names_path = PROJECT_ROOT / "artifacts" / "preprocessors" / "feature_names.json"
    feat_mask_path = PROJECT_ROOT / "artifacts" / "preprocessors" / "feature_mask.json"
    bounds_path = PROJECT_ROOT / "artifacts" / "preprocessors" / "training_bounds.parquet"
    profile_path = PROJECT_ROOT / "artifacts" / "preprocessors" / "afp_benign_profile.parquet"
    defenses_yaml_path = PROJECT_ROOT / "configs" / "defenses.yaml"
    
    X_cal = pd.read_parquet(x_cal_path)
    meta_cal = pd.read_parquet(meta_cal_path)
    
    if X_cal.shape != (31500, 78):
        raise ValueError(f"X_calibration must be 31500x78, got {X_cal.shape}")
        
    rf = joblib.load(rf_path)
    
    with open(feat_names_path) as f:
        feature_names = json.load(f)
        
    with open(feat_mask_path) as f:
        mask_data = json.load(f)
        feature_mask = mask_data["feature_mask"]
        
    bounds = pd.read_parquet(bounds_path)
    profile = pd.read_parquet(profile_path)
    
    with open(defenses_yaml_path) as f:
        defenses_cfg = yaml.safe_load(f)
        
    # Check exact feature order
    if list(X_cal.columns) != feature_names:
        raise ValueError("X_calibration columns do not match feature_names.")
        
    # Check alignment
    if not (X_cal.index == meta_cal.index).all():
        raise ValueError("X_calibration and metadata_calibration indices are not exactly aligned.")
        
    X_val = X_cal.values.astype(np.float32)
    y_true = meta_cal["y_binary"].values
    
    # 5. Identity Controls
    afp = AdaptiveFeaturePoisoning(feature_names, feature_mask, bounds, profile)
    rs = RandomizedSmoothing(feature_names, feature_mask, bounds, ensemble_size=11)
    
    X_def_id, res_id, _ = afp.defend(X_val, epsilon_base=0.0, alpha=0.0, seed=42, attack_scenario="calibration", batch_id=0)
    
    if not np.array_equal(X_def_id, X_val):
        raise ValueError("AFP eps=0 did not return identical bit-for-bit array.")
        
    if res_id.projected_cell_count != 0:
        raise ValueError("Projection changed cells on identity control.")
        
    y_pred_id = rf.predict(X_def_id)
    base_metrics = compute_metrics(y_true, y_pred_id)
    
    if base_metrics["TN"] != 26105 or base_metrics["FP"] != 33 or base_metrics["FN"] != 277 or base_metrics["TP"] != 5085:
        raise ValueError(f"Baseline mismatch! Got {base_metrics}")
        
    y_pred_rs_id, res_rs_id = rs.predict_ensemble(X_val, sigma=0.0, seed=42, attack_scenario="calibration", batch_id=0, predict_func=rf.predict, chunk_size=500)
    if not np.array_equal(y_pred_rs_id, y_pred_id):
        raise ValueError("RS sigma=0 did not reproduce baseline predictions exactly.")
        
    # 6. Isolate Noise from Clipping
    # A. AFP (eps=0.01, alpha=0.5)
    eps, alpha = 0.01, 0.5
    afp_prop, _ = afp._generate_proposal(X_val, eps, alpha, 42, "calibration", 0)
    afp_clipped, afp_res = afp.project_to_bounds(afp_prop, X_val)
    
    y_unclipped = rf.predict(afp_prop)
    y_clipped = rf.predict(afp_clipped)
    
    afp_unclipped_metrics = compute_metrics(y_true, y_unclipped, y_pred_id)
    afp_clipped_metrics = compute_metrics(y_true, y_clipped, y_pred_id)
    
    # B. RS (sigma=0.01)
    rs_props = _generate_rs_proposals(rs, X_val, 0.01, 42, "calibration", 0)
    rs_unclipped_preds = np.zeros(len(X_val), dtype=int)
    rs_clipped_preds = np.zeros(len(X_val), dtype=int)
    rs_total_prop_oob = 0
    rs_total_proj_cells = 0
    
    for i, prop in enumerate(rs_props):
        p_u = rf.predict(prop)
        clipped_prop, rs_res = rs.project_to_bounds(prop, X_val)
        p_c = rf.predict(clipped_prop)
        rs_unclipped_preds += p_u
        rs_clipped_preds += p_c
        rs_total_prop_oob += rs_res.proposal_out_of_bounds_count
        rs_total_proj_cells += rs_res.projected_cell_count
        
    rs_y_u = (rs_unclipped_preds > 5).astype(int)
    rs_y_c = (rs_clipped_preds > 5).astype(int)
    
    rs_unclipped_metrics = compute_metrics(y_true, rs_y_u, y_pred_id)
    rs_clipped_metrics = compute_metrics(y_true, rs_y_c, y_pred_id)
    
    rs_clip_res = {
        "unclipped": rs_unclipped_metrics,
        "clipped": rs_clipped_metrics,
        "exact_agreement": int(np.sum(rs_y_u == rs_y_c)),
        "proposal_out_of_bounds_count": rs_total_prop_oob,
        "projected_cell_count": rs_total_proj_cells
    }
    
    afp_clip_res = {
        "unclipped": afp_unclipped_metrics,
        "clipped": afp_clipped_metrics,
        "exact_agreement": int(np.sum(y_unclipped == y_clipped)),
        "proposal_out_of_bounds_count": afp_res.proposal_out_of_bounds_count,
        "projected_cell_count": afp_res.projected_cell_count
    }
    
    # 7. Logarithmic Scale Diagnostic
    log_vals = [0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    log_results = []
    
    for val in log_vals:
        for a in [0.25, 0.50, 1.00]:
            X_def, res, _ = afp.defend(X_val, epsilon_base=val, alpha=a, seed=42, attack_scenario="calibration", batch_id=0)
            yp = rf.predict(X_def)
            
            m = compute_metrics(y_true, yp, y_pred_id)
            log_results.append({
                "defense": "afp", "val": val, "alpha": a,
                **m,
                "original_out_of_bounds_count": res.original_out_of_bounds_count,
                "original_eligible_out_of_bounds_count": res.original_eligible_out_of_bounds_count,
                "proposal_out_of_bounds_count": res.proposal_out_of_bounds_count,
                "proposal_out_of_bounds_fraction": res.proposal_out_of_bounds_fraction,
                "projected_cell_count": res.projected_cell_count,
                "final_bounds_violation_count": res.final_bounds_violation_count,
                "final_invalid_fraction": res.final_invalid_fraction
            })
            
        yp_rs, res_rs = rs.predict_ensemble(X_val, sigma=val, seed=42, attack_scenario="calibration", batch_id=0, predict_func=rf.predict, chunk_size=500)
        m_rs = compute_metrics(y_true, yp_rs, y_pred_id)
        log_results.append({
            "defense": "rs", "val": val, "alpha": None,
            **m_rs,
            "original_out_of_bounds_count": res_rs.original_out_of_bounds_count,
            "original_eligible_out_of_bounds_count": res_rs.original_eligible_out_of_bounds_count,
            "proposal_out_of_bounds_count": res_rs.proposal_out_of_bounds_count,
            "proposal_out_of_bounds_fraction": res_rs.proposal_out_of_bounds_fraction,
            "projected_cell_count": res_rs.projected_cell_count,
            "final_bounds_violation_count": res_rs.final_bounds_violation_count,
            "final_invalid_fraction": res_rs.final_invalid_fraction
        })
        
    out = {
        "identity_controls": "PASSED",
        "identity_metrics": {
            "afp_changed_cells": res_id.projected_cell_count,
            "afp_proposal_oob": res_id.proposal_out_of_bounds_count,
            "afp_final_invalid": res_id.final_invalid_fraction
        },
        "isolate_noise_vs_clipping": {
            "afp": afp_clip_res,
            "rs": rs_clip_res
        },
        "logarithmic_results": log_results,
        "runtime_seconds": time.time() - start_time,
        "peak_ram_mb": get_peak_ram()
    }
    
    with open(PROJECT_ROOT / "artifacts" / "reports" / "phase8_failure_diagnostic_corrected.json", "w") as f:
        json.dump(out, f, indent=2)
        
    with open(PROJECT_ROOT / "artifacts" / "reports" / "phase8_failure_diagnostic_corrected.md", "w") as f:
        f.write("# Corrected Phase 8 Failure Diagnostic Report\n\n")
        f.write(f"Identity Controls: {out['identity_controls']}\n\n")
        f.write("## Isolation of Noise vs Clipping (Corrected)\n")
        f.write("When passing the unclipped proposal to the model:\n")
        f.write(f"- AFP unclipped recall: {afp_clip_res['unclipped']['recall']:.4f}\n")
        f.write(f"- AFP clipped recall: {afp_clip_res['clipped']['recall']:.4f}\n")
        f.write(f"- RS unclipped recall: {rs_clip_res['unclipped']['recall']:.4f}\n")
        f.write(f"- RS clipped recall: {rs_clip_res['clipped']['recall']:.4f}\n")
        
        f.write("\n## Logarithmic Diagnostics Summary\n")
        for r in log_results:
            if r['val'] == 0: continue
            lbl = f"{r['defense'].upper()} val={r['val']}"
            if r['alpha'] is not None: lbl += f" alpha={r['alpha']}"
            f.write(f"- {lbl}: Recall={r['recall']:.4f}, PropOOB={r['proposal_out_of_bounds_count']}, ProjCells={r['projected_cell_count']}\n")
            
if __name__ == "__main__":
    run_diagnostics()

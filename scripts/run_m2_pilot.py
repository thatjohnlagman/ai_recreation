import time
import sys
import os
import resource
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Need to load the model, scaler, defenses, attacks
from recall_aware_ids.utils.config import load_yaml
import joblib
from recall_aware_ids.defenses.afp import AdaptiveFeaturePoisoning
from recall_aware_ids.defenses.feature_squeezing import FeatureSqueezing
from recall_aware_ids.defenses.randomized_smoothing import RandomizedSmoothing
from recall_aware_ids.attacks.surrogate_transfer import SurrogateTransferAttack
from recall_aware_ids.attacks.boundary_attack import DecisionBoundaryAttack

ROOT = Path(__file__).parents[1]

def run_pilot():
    print("Running genuine training-derived Pilot...")
    
    start_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    
    # 1. Load actual frozen RF and data
    try:
        x_path = ROOT / "data/processed/X_train.parquet"
        meta_path = ROOT / "data/processed/metadata_train.parquet"
        
        if not x_path.exists() or not meta_path.exists():
            raise FileNotFoundError("X_train.parquet and metadata_train.parquet are required for the pilot")
            
        df_x = pd.read_parquet(x_path).head(5000)
        df_meta = pd.read_parquet(meta_path).head(5000)
        
        X = df_x.values.astype(np.float32)
        y = df_meta['y_binary'].values.astype(int)
        
    except Exception as e:
        print(f"Error loading training data: {e}")
        sys.exit(1)
        
    print(f"Records loaded: {len(X)} from training partition.")
    class_counts = np.bincount(y)
    print(f"Class counts: {class_counts}")
        
    # Load model
    model = joblib.load(ROOT / "artifacts/models/frozen_rf.joblib")
    
    # Load feature artifacts
    with open(ROOT / "artifacts/preprocessors/feature_mask.json") as f:
        mask_data = json.load(f)
        mask = mask_data["feature_mask"]
        fnames = mask_data["feature_columns"]
    
    try:
        real_bounds = pd.read_parquet(ROOT / "artifacts/preprocessors/training_bounds.parquet")
        real_profile = pd.read_parquet(ROOT / "artifacts/preprocessors/afp_benign_profile.parquet")
    except Exception as e:
        print(f"Error loading bounding/profile artifacts: {e}")
        sys.exit(1)
        
    def predict_wrapper(x_in):
        return model.predict(x_in)
        
    def predict_proba_wrapper(x_in):
        return model.predict_proba(x_in)
    
    # Init defenses
    afp = AdaptiveFeaturePoisoning(feature_names=fnames, modifiable_mask=mask, training_bounds=real_bounds, benign_profile=real_profile)
    fs = FeatureSqueezing(feature_names=fnames, modifiable_mask=mask, training_bounds=real_bounds)
    rs = RandomizedSmoothing(feature_names=fnames, modifiable_mask=mask, training_bounds=real_bounds, ensemble_size=11)
    
    print("\n--- Defense Timings ---")
    
    # AFP
    t0 = time.time()
    afp_preds, _, afp_scores = afp.defend(X, epsilon_base=0.0003, alpha=0.5, seed=42, attack_scenario="Pilot", batch_id=0)
    # AFP requires RF inference on transformed data
    _ = model.predict(X)
    _ = model.predict_proba(X)
    t_afp = time.time() - t0
    
    # FS
    t0 = time.time()
    fs_preds, _, fs_scores = fs.defend(X, intensity=2.0, seed=42, attack_scenario="Pilot", batch_id=0)
    _ = model.predict(X)
    _ = model.predict_proba(X)
    t_fs = time.time() - t0
    
    # RS
    t0 = time.time()
    rs_preds, _, rs_scores = rs.predict_ensemble(X, 0.0002, seed=42, attack_scenario="Pilot", batch_id=0, predict_func=predict_wrapper, return_scores=True)
    t_rs = time.time() - t0
    
    print(f"AFP time (5k records + RF infer): {t_afp:.4f}s")
    print(f"FS time (5k records + RF infer):  {t_fs:.4f}s")
    print(f"RS time (5k records + 11 RF inf): {t_rs:.4f}s")
    print(f"RF Prediction Calls for RS: 11 * 5000 = 55000")
    
    print("\n--- Attack Workflows (Training-Derived subsets) ---")
    
    surrogate = SurrogateTransferAttack(feature_names=fnames, modifiable_mask=mask, training_bounds=real_bounds)
    boundary = DecisionBoundaryAttack(feature_names=fnames, modifiable_mask=mask, training_bounds=real_bounds)
    
    from recall_aware_ids.attacks.oracle import BlackBoxOracle
    
    # Surrogate
    query_subset_X = X[:1000]
    target_subset_X = X[1000:1500]
    target_subset_y = y[1000:1500]
    
    t0 = time.time()
    y_pool_oracle = predict_wrapper(query_subset_X)
    surrogate.fit_surrogate(query_subset_X, y_pool_oracle)
    
    surrogate_candidates = []
    for i in range(len(target_subset_X)):
        X_cand, mags = surrogate.generate_candidate(target_subset_X[i])
        surrogate_candidates.append(X_cand)
        
    t_sur = time.time() - t0
    
    print(f"Surrogate Transfer time (1000 queries, 500 attacks): {t_sur:.4f}s")
    
    # Boundary
    # Requires an attack subset and a benign reference pool
    is_attack = y == 1
    is_benign = y == 0
    attack_pool_X = X[is_attack][:100]
    attack_pool_y = y[is_attack][:100]
    benign_pool_X = X[is_benign][:500]
    
    t0 = time.time()
    oracle = BlackBoxOracle(predict_wrapper, max_queries_per_sample=50)
    boundary_results = []
    
    for i in range(len(attack_pool_X)):
        res = boundary.generate(attack_pool_X[i], oracle, sample_id=f"pilot_{i}", true_label=attack_pool_y[i], reference_pool=benign_pool_X)
        boundary_results.append(res)
        
    t_bound = time.time() - t0
    
    queries = oracle.global_query_count
    print(f"Decision Boundary time (100 attacks): {t_bound:.4f}s, Total Queries: {queries}")
    
    end_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mem_base = 1024 * 1024 if sys.platform == "darwin" else 1024
    peak_ram_mb = end_mem / mem_base
    inc_ram_mb = (end_mem - start_mem) / mem_base
    
    print("\n--- Resource Summary ---")
    print(f"Absolute Peak RSS: {peak_ram_mb:.2f} MB")
    print(f"Incremental RSS:   {inc_ram_mb:.2f} MB")
    
    # Cache estimation
    # 78 float32 columns = 312 bytes per row. Plus 10 status columns = ~100 bytes. ~450 bytes per row.
    cache_estimate_kb = (72000 * 450) / 1024
    print(f"Estimated Cache Size (72000 records): {cache_estimate_kb:.2f} KB (~{cache_estimate_kb/1024:.2f} MB)")
    
    print("\nScaling Assumptions:")
    print("- All projections assume linear scaling of RF inference.")
    print("- Surrogate is bounded by tree generation time.")
    print("- Boundary is bounded by sequential target queries.")
    
if __name__ == "__main__":
    run_pilot()

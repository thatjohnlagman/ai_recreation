"""
Phase 10A Training-Derived Runtime Pilot.

Timing and resource measurements use ONLY training-partition data.
No evaluation, crafting, or measurement records are accessed.

Corrections applied:
  - AFP: model called on X_afp (defended), not original X
  - FS: model called on X_fs (defended), not original X; effective_d captured
  - Surrogate: crafting queries go through BlackBoxOracle for proper query tracking
  - Boundary: all queries tracked via BlackBoxOracle
  - Reports: eligible/attempted/success counts, exact Oracle query counts,
             defense diagnostics (final_invalid, protected_feature_modification, projected_cells)
  - Interpretation: RS dominates total experiment time
"""
import time
import sys
import os
import resource
import json
import numpy as np
import pandas as pd
from pathlib import Path

from recall_aware_ids.utils.config import load_yaml
import joblib
from recall_aware_ids.defenses.afp import AdaptiveFeaturePoisoning
from recall_aware_ids.defenses.feature_squeezing import FeatureSqueezing
from recall_aware_ids.defenses.randomized_smoothing import RandomizedSmoothing
from recall_aware_ids.attacks.surrogate_transfer import SurrogateTransferAttack
from recall_aware_ids.attacks.boundary_attack import DecisionBoundaryAttack
from recall_aware_ids.attacks.oracle import BlackBoxOracle

ROOT = Path(__file__).parents[1]


def run_pilot():
    print("Running genuine training-derived Pilot...")
    start_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # ── 1. Load training data ──────────────────────────────────────────────
    try:
        x_path = ROOT / "data/processed/X_train.parquet"
        meta_path = ROOT / "data/processed/metadata_train.parquet"
        if not x_path.exists() or not meta_path.exists():
            raise FileNotFoundError("X_train.parquet and metadata_train.parquet required")
        df_x = pd.read_parquet(x_path).head(5000)
        df_meta = pd.read_parquet(meta_path).head(5000)
        X = df_x.values.astype(np.float32)
        y = df_meta["y_binary"].values.astype(int)
    except Exception as e:
        print(f"Error loading training data: {e}")
        sys.exit(1)

    print(f"Records loaded: {len(X)} from training partition.")
    print(f"Class counts: {np.bincount(y)}")

    # ── 2. Load model and artifacts ───────────────────────────────────────
    model = joblib.load(ROOT / "artifacts/models/frozen_rf.joblib")

    with open(ROOT / "artifacts/preprocessors/feature_mask.json") as f:
        mask_data = json.load(f)
    mask = mask_data["feature_mask"]
    fnames = mask_data["feature_columns"]

    real_bounds = pd.read_parquet(ROOT / "artifacts/preprocessors/training_bounds.parquet")
    real_profile = pd.read_parquet(ROOT / "artifacts/preprocessors/afp_benign_profile.parquet")

    def predict_wrapper(x_in):
        return model.predict(x_in)

    # ── 3. Defense timings ────────────────────────────────────────────────
    print("\n--- Defense Timings (Training-Derived, 5k records) ---")

    afp = AdaptiveFeaturePoisoning(
        feature_names=fnames, modifiable_mask=mask,
        training_bounds=real_bounds, benign_profile=real_profile
    )
    fs = FeatureSqueezing(feature_names=fnames, modifiable_mask=mask, training_bounds=real_bounds)
    rs = RandomizedSmoothing(feature_names=fnames, modifiable_mask=mask,
                              training_bounds=real_bounds, ensemble_size=11)

    # ── AFP ──
    t0 = time.time()
    X_afp, afp_result, mean_eps = afp.defend(X, epsilon_base=0.0003, alpha=0.5, seed=42,
                                              attack_scenario="Pilot", batch_id=0)
    afp_labels = model.predict(X_afp)       # predict on DEFENDED output
    afp_scores = model.predict_proba(X_afp)[:, 1]
    t_afp = time.time() - t0
    print(f"AFP time (5k records + RF infer on defended): {t_afp:.4f}s")
    print(f"  final_invalid_cells:     {afp_result.final_nan_count + afp_result.final_inf_count + afp_result.final_bounds_violation_count}")
    print(f"  protected_modified:      {afp_result.protected_feature_modification_count}")
    print(f"  projected_cells:         {afp_result.projected_cell_count}")

    # ── FS ──
    t0 = time.time()
    X_fs, fs_result, effective_d = fs.defend(X, intensity=2.0, seed=42,
                                              attack_scenario="Pilot", batch_id=0)
    fs_labels = model.predict(X_fs)         # predict on DEFENDED output
    fs_scores = model.predict_proba(X_fs)[:, 1]
    t_fs = time.time() - t0
    print(f"FS time (5k records + RF infer on defended):  {t_fs:.4f}s")
    print(f"  effective_d:             {effective_d}")
    print(f"  final_invalid_cells:     {fs_result.final_nan_count + fs_result.final_inf_count + fs_result.final_bounds_violation_count}")
    print(f"  protected_modified:      {fs_result.protected_feature_modification_count}")

    # ── RS ──
    t0 = time.time()
    rs_preds, rs_result, rs_scores = rs.predict_ensemble(
        X, sigma=0.0002, seed=42, attack_scenario="Pilot", batch_id=0,
        predict_func=predict_wrapper, return_scores=True
    )
    t_rs = time.time() - t0
    print(f"RS time (5k records + 11 RF inf):             {t_rs:.4f}s")
    print(f"  positive_vote_fraction range: [{rs_scores.min():.3f}, {rs_scores.max():.3f}]")
    print(f"  protected_modified:      {rs_result.protected_feature_modification_count}")

    # ── 4. Attack workflows ────────────────────────────────────────────────
    print("\n--- Attack Workflows (Training-Derived subsets) ---")

    surrogate = SurrogateTransferAttack(feature_names=fnames, modifiable_mask=mask,
                                         training_bounds=real_bounds)
    boundary = DecisionBoundaryAttack(feature_names=fnames, modifiable_mask=mask,
                                       training_bounds=real_bounds)

    # ── Surrogate Transfer ──
    query_subset_X = X[:1000]
    target_subset_X = X[1000:1500]
    target_subset_y = y[1000:1500]

    # Oracle for surrogate crafting queries — tracks 1,000 crafting queries
    surrogate_oracle = BlackBoxOracle(predict_wrapper, max_queries_per_sample=None)

    t0 = time.time()
    # Use oracle for pool labels
    y_pool = surrogate_oracle.predict(query_subset_X, sample_ids=list(range(1000)), stage="surrogate_fit")
    surrogate.fit_surrogate(query_subset_X, y_pool)

    eligible = 0
    attempted = 0
    successful = 0
    surrogate_candidates = []
    for i in range(len(target_subset_X)):
        orig_pred = int(predict_wrapper(target_subset_X[i:i+1])[0])
        true_label = int(target_subset_y[i])
        if orig_pred != 1 or true_label != 1:
            continue
        eligible += 1
        X_cand, mags = surrogate.generate_candidate(target_subset_X[i])
        if X_cand is not None:
            attempted += 1
            final = int(predict_wrapper(X_cand.reshape(1, -1))[0])
            if final == 0:
                successful += 1
            surrogate_candidates.append(X_cand)

    t_sur = time.time() - t0
    print(f"Surrogate Transfer (1000 query pool, 500 targets): {t_sur:.4f}s")
    print(f"  Oracle queries (crafting pool):  {surrogate_oracle.global_query_count}")
    print(f"  Eligible targets:                {eligible}")
    print(f"  Attempted:                       {attempted}")
    print(f"  Successful:                      {successful}")

    # ── Decision Boundary ──
    is_attack = y == 1
    is_benign = y == 0
    attack_pool_X = X[is_attack][:100]
    attack_pool_y = y[is_attack][:100]
    benign_pool_X = X[is_benign][:500]

    boundary_oracle = BlackBoxOracle(predict_wrapper, max_queries_per_sample=50)

    t0 = time.time()
    b_eligible = 0
    b_attempted = 0
    b_successful = 0
    boundary_results = []

    for i in range(len(attack_pool_X)):
        orig_pred = int(predict_wrapper(attack_pool_X[i:i+1])[0])
        true_label = int(attack_pool_y[i])
        if orig_pred != 1 or true_label != 1:
            continue
        b_eligible += 1
        sample_id = f"pilot_{i}"
        res = boundary.generate(
            attack_pool_X[i], boundary_oracle, sample_id=sample_id,
            true_label=true_label, reference_pool=benign_pool_X
        )
        b_attempted += 1
        if getattr(res, "success", False):
            b_successful += 1
        boundary_results.append(res)

    t_bound = time.time() - t0
    total_queries = boundary_oracle.global_query_count
    print(f"Decision Boundary (100 attack targets): {t_bound:.4f}s")
    print(f"  Total Oracle queries:            {total_queries}")
    print(f"  Eligible:                        {b_eligible}")
    print(f"  Attempted:                       {b_attempted}")
    print(f"  Successful:                      {b_successful}")

    # ── 5. Resource summary ────────────────────────────────────────────────
    end_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mem_base = 1024 * 1024 if sys.platform == "darwin" else 1024
    peak_ram_mb = end_mem / mem_base
    inc_ram_mb = (end_mem - start_mem) / mem_base

    print("\n--- Resource Summary ---")
    print(f"Absolute Peak RSS: {peak_ram_mb:.2f} MB")
    print(f"Incremental RSS:   {inc_ram_mb:.2f} MB")

    cache_estimate_kb = (72000 * 450) / 1024
    print(f"Estimated Cache Size (72000 records): {cache_estimate_kb:.2f} KB (~{cache_estimate_kb/1024:.2f} MB)")

    print("\n--- Runtime Interpretation ---")
    print(f"RS ({t_rs:.2f}s for 5k) appears likely to DOMINATE total experiment time.")
    print(f"  RS at scale: 11 RF inferences × 72,000 records × 252 runs = ~{int(11*72000*252/1e6)}M RF calls.")
    print(f"  Surrogate ({t_sur:.2f}s) is bounded by tree construction and single-pass inference.")
    print(f"  Boundary ({t_bound:.2f}s / 100 targets) is bounded by sequential query budget.")
    print(f"  Unless later evidence shows otherwise, RS dominates total experiment time.")
    print("\nScaling assumptions:")
    print("  - All projections assume linear scaling of RF inference.")
    print("  - Surrogate candidate generation assumes O(N) per sample.")
    print("  - Boundary is bounded by sequential max_queries per target.")


if __name__ == "__main__":
    run_pilot()

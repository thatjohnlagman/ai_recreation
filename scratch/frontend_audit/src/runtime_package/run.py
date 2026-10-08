import argparse
import sys
import os
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
import yaml

from controller.recall_controller import RecallAwareController
from attacks.silent_probing import SilentProbingAttack
from attacks.surrogate_transfer import SurrogateTransferAttack
from attacks.boundary_attack import DecisionBoundaryAttack
from defenses.afp import AdaptiveFeaturePoisoning
from defenses.randomized_smoothing import RandomizedSmoothing
from defenses.feature_squeezing import FeatureSqueezing

class MockOracle:
    def __init__(self, model):
        self.model = model
        self.max_queries_per_sample = 50
        self.queries = {}

    def predict(self, X, sample_ids=None, stage=None):
        if sample_ids:
            sid = sample_ids[0]
            self.queries[sid] = self.queries.get(sid, 0) + 1
        # Model returns predictions
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model.predict(X)

    def get_query_count(self, sample_id):
        return self.queries.get(sample_id, 0)

class IDS:
    def __init__(self):
        self.model = joblib.load(Path(__file__).parent / "model" / "frozen_rf.joblib")
        with open(Path(__file__).parent / "model" / "feature_names.json", "r") as f:
            self.feature_names = json.load(f)
        with open(Path(__file__).parent / "model" / "feature_mask.json", "r") as f:
            mask_dict = json.load(f)
            # The dictionary contains {"feature_mask_eligible_numerical": [78 booleans]}
            self.feature_mask = list(mask_dict.values())[0]
        self.bounds = pd.read_parquet(Path(__file__).parent / "model" / "training_bounds.parquet")
        
    def predict(self, X):
        if np.ndim(X) == 1:
            X = [X]
        return self.model.predict(X)

def load_data():
    X = pd.read_parquet(Path(__file__).parent / "demo_data" / "X_demo.parquet")
    y = pd.read_parquet(Path(__file__).parent / "demo_data" / "metadata_demo.parquet")
    return X, y

def get_attack(name, feature_names, modifiable_mask, bounds, X_pool=None, y_pool=None, model=None):
    if name == "silent_probing":
        return SilentProbingAttack(feature_names, modifiable_mask, bounds)
    elif name == "surrogate_transfer":
        attack = SurrogateTransferAttack(feature_names, modifiable_mask, bounds)
        if X_pool is not None and model is not None:
            # Fit surrogate on oracle labels
            oracle_labels = model.predict(X_pool)
            attack.fit_surrogate(X_pool, oracle_labels)
        return attack
    elif name == "decision_boundary":
        return DecisionBoundaryAttack(feature_names, modifiable_mask, bounds, max_queries=50, binary_search_steps=10)
    return None

def get_defense(name, feature_names, modifiable_mask, bounds):
    with open(Path(__file__).parent / "configs" / "defenses.yaml", "r") as f:
        d_cfg = yaml.safe_load(f)
    if name == "afp":
        afp_profile = pd.read_parquet(Path(__file__).parent / "model" / "afp_benign_profile.parquet")
        return AdaptiveFeaturePoisoning(feature_names, modifiable_mask, bounds, afp_profile)
    elif name == "rs":
        return RandomizedSmoothing(feature_names, modifiable_mask, bounds)
    elif name == "fs":
        return FeatureSqueezing(feature_names, modifiable_mask, bounds)
    return None

def get_controller(c_type, defense_name):
    if c_type == "base":
        return None
    with open(Path(__file__).parent / "configs" / "controllers.yaml", "r") as f:
        c_cfg = yaml.safe_load(f)
        
    config = c_cfg.get("controller_configurations", {}).get("C1", {})
    if not config:
        config = c_cfg.get("C1", {})
    if not config:
        config = {"id": "C1", "window_size": 5, "Rcritical": 0.85, "Rmin": 0.95, "fast_decay": 0.40, "slow_decay": 0.90, "growth_factor": 1.05}
        
    with open(Path(__file__).parent / "configs" / "defenses.yaml", "r") as f:
        d_cfg = yaml.safe_load(f)
        
    def_key = "afp"
    def_name_long = "afp"
    if defense_name == "rs":
        def_key = "randomized_smoothing"
        def_name_long = "randomized_smoothing"
    elif defense_name == "fs":
        def_key = "feature_squeezing"
        def_name_long = "feature_squeezing"
        
    return RecallAwareController(config, d_cfg[def_key], def_name_long)


def run_simulation(attack_name, defense_name, controller_mode, compare_mode=False):
    ids = IDS()
    X, y = load_data()
    
    attack = get_attack(attack_name, ids.feature_names, ids.feature_mask, ids.bounds, X_pool=X.values, y_pool=y["y_binary"].values, model=ids.model) if attack_name else None
    reference_pool = X[y["y_binary"] == 0].values if attack_name == "decision_boundary" else None
    
    scenarios = []
    if compare_mode:
        scenarios = [
            ("No Defense", None, None),
            (f"Base {defense_name.upper()}", get_defense(defense_name, ids.feature_names, ids.feature_mask, ids.bounds), None),
            (f"Recall-Aware {defense_name.upper()}", get_defense(defense_name, ids.feature_names, ids.feature_mask, ids.bounds), get_controller("recall-aware", defense_name))
        ]
    else:
        d_name_str = defense_name.upper() if defense_name else "NONE"
        scenarios = [(f"{controller_mode.capitalize()} {d_name_str}", get_defense(defense_name, ids.feature_names, ids.feature_mask, ids.bounds) if defense_name else None, get_controller(controller_mode, defense_name) if defense_name else None)]
        
    print("\n" + "="*60)
    print("  RECALL-AWARE IDS LOCAL DEMONSTRATION")
    print(f"  Attack: {attack_name or 'None'} | Defense: {defense_name or 'None'} | Controller: {controller_mode}")
    print("  Note: Standalone local control-path demo; does not replicate Phase 10/11 statistical evaluation.")
    print("="*60)

    oracle = MockOracle(ids.model)
    
    for s_name, d, c in scenarios:
        print(f"\n--- Scenario: {s_name} ---")
        
        detected = 0
        total_attacks = 0
        evasions = 0
        fp = 0
        total_benign = 0
        
        batch_tp = 0
        batch_fn = 0
        batch_id = 0
        
        if c:
            c.reset()
            
        # Obtain intensity for batch_id = 0 once before processing rows
        intensity = None
        if c and d:
            decision = c.get_intensity(batch_id)
            intensity = decision.intensity
        elif d:
            if defense_name == "afp":
                with open(Path(__file__).parent / "configs" / "defenses.yaml", "r") as f:
                    intensity = yaml.safe_load(f)["afp"]["epsilon_base"]
            elif defense_name == "rs":
                with open(Path(__file__).parent / "configs" / "defenses.yaml", "r") as f:
                    intensity = yaml.safe_load(f)["randomized_smoothing"]["sigma"]
            elif defense_name == "fs":
                with open(Path(__file__).parent / "configs" / "defenses.yaml", "r") as f:
                    intensity = yaml.safe_load(f)["feature_squeezing"]["squeezing_intensity"]
        
        for i, (idx, row) in enumerate(X.iterrows()):
            is_attack = (y.iloc[i]["y_binary"] == 1)
            x_val = row.values

            # 1. Attack
            if is_attack and attack:
                try:
                    if attack_name == "silent_probing":
                        res = attack.generate(x_val, oracle, idx, 1)
                        x_val = res.X_adv
                    elif attack_name == "surrogate_transfer":
                        x_cand, _ = attack.generate_candidate(x_val)
                        res = attack.evaluate_transfer(x_cand, x_val, oracle, idx, 1)
                        x_val = res.X_adv
                    elif attack_name == "decision_boundary":
                        res = attack.generate(x_val, oracle, idx, 1, reference_pool)
                        x_val = res.X_adv
                except Exception as e:
                    print(f"[Attack Generation Failed on sample {idx}]: {e}", file=sys.stderr)
                    raise
            
            # 2. Defense and Predict
            pred = 0
            if d:
                # Defend
                if defense_name == "afp":
                    # For AFP: defend(X, epsilon_base, alpha, seed, attack_scenario, batch_id)
                    X_proj, _, _ = d.defend(x_val.reshape(1, -1), intensity, 0.5, 42, "none", batch_id)
                    x_val = X_proj[0]
                    pred = ids.predict(x_val)[0]
                elif defense_name == "rs":
                    # For RS: predict_ensemble(X, sigma, seed, attack_scenario, batch_id, predict_func)
                    preds, _ = d.predict_ensemble(x_val.reshape(1, -1), intensity, 42, "none", batch_id, ids.predict)
                    pred = preds[0]
                elif defense_name == "fs":
                    # For FS: defend(X, intensity, seed, attack_scenario, batch_id)
                    X_proj, _, _ = d.defend(x_val.reshape(1, -1), intensity, seed=42, attack_scenario="none", batch_id=batch_id)
                    x_val = X_proj[0]
                    pred = ids.predict(x_val)[0]
            else:
                pred = ids.predict(x_val)[0]
            
            # 4. Metrics & Controller update
            if is_attack:
                total_attacks += 1
                if pred == 1:
                    detected += 1
                    batch_tp += 1
                else:
                    evasions += 1
                    batch_fn += 1
            else:
                total_benign += 1
                if pred == 1:
                    fp += 1
                    
            if is_attack and c and d:
                if (batch_tp + batch_fn) >= 5:
                    c.submit_observations(batch_id, batch_tp, batch_fn)
                    batch_id += 1
                    batch_tp = 0
                    batch_fn = 0
                    # Request next batch's intensity only after submitting observations
                    decision = c.get_intensity(batch_id)
                    intensity = decision.intensity
                
        if c and d and (batch_tp + batch_fn) > 0:
            c.submit_observations(batch_id, batch_tp, batch_fn)

        dr = (detected / total_attacks * 100) if total_attacks > 0 else 0
        print(f"Total Samples: {len(X)}")
        print(f"Attacks: {total_attacks} | Benign: {total_benign}")
        print(f"Detected Attacks: {detected}")
        print(f"Successful Evasions: {evasions}")
        print(f"Detection Rate (Recall): {dr:.2f}%")
        print(f"False Positives: {fp}")
        if d:
            print(f"Final Perturbation Intensity: {c.current_intensity if c else 'Base'}")
            
    print("\nDemo completed.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recall-Aware IDS Demonstration")
    parser.add_argument("--attack", choices=["silent_probing", "surrogate_transfer", "decision_boundary", "none"], default="silent_probing")
    parser.add_argument("--defense", choices=["afp", "rs", "fs", "none"], default="afp")
    parser.add_argument("--controller", choices=["base", "recall-aware"], default="base")
    parser.add_argument("--compare", action="store_true", help="Run comparison mode (No defense vs Base vs Recall-Aware)")
    
    args = parser.parse_args()
    
    a = None if args.attack == "none" else args.attack
    d = None if args.defense == "none" else args.defense
    
    if args.compare and d:
        run_simulation(a, d, args.controller, compare_mode=True)
    else:
        run_simulation(a, d, args.controller, compare_mode=False)

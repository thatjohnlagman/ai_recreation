import pandas as pd
from typing import List, Dict, Any
import dataclasses
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass(frozen=True)
class RunConfig:
    run_id: str
    seed: int
    attack_scenario: str
    defense_name: str
    controller_config_id: str
    is_alias: bool
    alias_for_run_id: str

def generate_evaluation_matrix(configs_dir: Path) -> pd.DataFrame:
    """
    Generates the deterministic evaluation matrix and validates it
    against the experiment configurations.
    """
    with open(configs_dir / "experiment.yaml") as f:
        exp_yaml = yaml.safe_load(f)
    with open(configs_dir / "attacks.yaml") as f:
        attacks_yaml = yaml.safe_load(f)
    with open(configs_dir / "defenses.yaml") as f:
        defenses_yaml = yaml.safe_load(f)
    with open(configs_dir / "controllers.yaml") as f:
        controllers_yaml = yaml.safe_load(f)

    primary_seeds = exp_yaml["stochastic"]["primary_seeds"]
    sensitivity_seeds = exp_yaml["stochastic"]["sensitivity_seeds"]
    
    attacks = ['Silent Probing', 'Surrogate Transfer', 'Decision Boundary']
    defenses = ['afp', 'feature_squeezing', 'randomized_smoothing']
    
    # Controllers from YAML
    yaml_ctrl_keys = set(controllers_yaml.get("controller_configurations", {}).keys())
    expected_ctrl_keys = {"C1", "C2", "C3", "C4", "C5", "C6", "C7"}
    if yaml_ctrl_keys != expected_ctrl_keys:
        raise ValueError(
            f"controller_configurations must be exactly C1..C7 (no C8 permitted), got: {yaml_ctrl_keys}"
        )
    controllers = list(controllers_yaml["controller_configurations"].keys())
    controllers.append("Base")
    
    primary_controllers = ['Base', 'C1']
    sensitivity_controllers = [c for c in controllers if c != 'Base']

    runs = []
    
    # 1. Primary References (90 runs)
    for seed in primary_seeds:
        for attack in attacks:
            for defense in defenses:
                for ctrl in primary_controllers:
                    run_id = f"primary_{seed}_{attack.replace(' ', '')}_{defense}_{ctrl}"
                    runs.append(RunConfig(
                        run_id=run_id,
                        seed=seed,
                        attack_scenario=attack,
                        defense_name=defense,
                        controller_config_id=ctrl,
                        is_alias=False,
                        alias_for_run_id=""
                    ))
                    
    # 2. Sensitivity References (189 runs)
    for seed in sensitivity_seeds:
        for attack in attacks:
            for defense in defenses:
                for ctrl in sensitivity_controllers:
                    run_id = f"sensitivity_{seed}_{attack.replace(' ', '')}_{defense}_{ctrl}"
                    
                    if ctrl == 'C1':
                        # This is a C1 alias, it points to the primary run
                        alias_target = f"primary_{seed}_{attack.replace(' ', '')}_{defense}_{ctrl}"
                        runs.append(RunConfig(
                            run_id=run_id,
                            seed=seed,
                            attack_scenario=attack,
                            defense_name=defense,
                            controller_config_id=ctrl,
                            is_alias=True,
                            alias_for_run_id=alias_target
                        ))
                    else:
                        runs.append(RunConfig(
                            run_id=run_id,
                            seed=seed,
                            attack_scenario=attack,
                            defense_name=defense,
                            controller_config_id=ctrl,
                            is_alias=False,
                            alias_for_run_id=""
                        ))

    df = pd.DataFrame([dataclasses.asdict(r) for r in runs])
    
    # Validation
    primary_count = len(df[df["run_id"].str.startswith("primary_")])
    sensitivity_count = len(df[df["run_id"].str.startswith("sensitivity_")])
    alias_count = len(df[df["is_alias"]])
    unique_count = len(df[~df["is_alias"]])
    batches = unique_count * 144
    rs_runs = len(df[(~df["is_alias"]) & (df["defense_name"] == "randomized_smoothing")])

    if primary_count != 90: raise ValueError(f"Expected 90 primary references, got {primary_count}")
    if sensitivity_count != 189: raise ValueError(f"Expected 189 sensitivity references, got {sensitivity_count}")
    if len(df) != 279: raise ValueError(f"Expected 279 total rows, got {len(df)}")
    if alias_count != 27: raise ValueError(f"Expected 27 aliases, got {alias_count}")
    if unique_count != 252: raise ValueError(f"Expected 252 unique executions, got {unique_count}")
    if batches != 36288: raise ValueError(f"Expected 36288 batches, got {batches}")
    if rs_runs != 84: raise ValueError(f"Expected 84 unique RS runs, got {rs_runs}")
    
    # Validation directions
    # Verify the generated sets match what's allowed in YAML
    yaml_attack_labels = set()
    for k, v in attacks_yaml.items():
        if isinstance(v, dict) and "label" in v:
            yaml_attack_labels.add(v["label"])
    
    # Check that our hardcoded labels map to yaml
    mapping = {
        'Silent Probing': 'Silent Probing',
        'Surrogate Transfer': 'Surrogate Transferability',
        'Decision Boundary': 'Decision-Boundary Attack'
    }
    
    for r in runs:
        if mapping[r.attack_scenario] not in yaml_attack_labels:
            raise ValueError(f"Invalid attack {r.attack_scenario}")
        if r.defense_name not in defenses_yaml.keys():
            raise ValueError(f"Invalid defense {r.defense_name}")
        if r.controller_config_id not in controllers:
            raise ValueError(f"Invalid controller {r.controller_config_id}")
        
    expected_primary = set()
    for s in primary_seeds:
        for a in attacks:
            for d in defenses:
                for c in primary_controllers:
                    expected_primary.add(f"primary_{s}_{a.replace(' ', '')}_{d}_{c}")
                    
    expected_sens = set()
    for s in sensitivity_seeds:
        for a in attacks:
            for d in defenses:
                for c in sensitivity_controllers:
                    expected_sens.add(f"sensitivity_{s}_{a.replace(' ', '')}_{d}_{c}")
                    
    generated_ids = set(df["run_id"])
    if expected_primary | expected_sens != generated_ids:
        raise ValueError("Bidirectional mapping failed: matrix does not match required combinations exactly")

    return df

def get_unique_executions(matrix: pd.DataFrame) -> pd.DataFrame:
    """Returns only the unique executions that need to be run (i.e. filters out aliases)."""
    return matrix[~matrix['is_alias']].copy()

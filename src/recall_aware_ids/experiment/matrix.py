import pandas as pd
from typing import List, Dict, Any
import dataclasses
from dataclasses import dataclass

@dataclass(frozen=True)
class RunConfig:
    run_id: str
    seed: int
    attack_scenario: str
    defense_name: str
    controller_config_id: str
    is_alias: bool
    alias_for_run_id: str

def generate_evaluation_matrix() -> pd.DataFrame:
    """
    Generates the deterministic evaluation matrix for Phase 10.
    Produces exactly:
    - 90 primary references
    - 189 sensitivity references
    - 27 exact C1 aliases
    - 252 unique executions
    """
    primary_seeds = [42, 43, 44, 45, 46]
    sensitivity_seeds = [42, 43, 44]
    
    attacks = ['Silent Probing', 'Surrogate Transfer', 'Decision Boundary']
    defenses = ['afp', 'feature_squeezing', 'randomized_smoothing']
    
    primary_controllers = ['Base', 'C1']
    sensitivity_controllers = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']

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

    return pd.DataFrame([dataclasses.asdict(r) for r in runs])

def get_unique_executions(matrix: pd.DataFrame) -> pd.DataFrame:
    """Returns only the unique executions that need to be run (i.e. filters out aliases)."""
    return matrix[~matrix['is_alias']].copy()

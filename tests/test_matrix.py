import pytest
from pathlib import Path
from recall_aware_ids.experiment.matrix import generate_evaluation_matrix, get_unique_executions

def test_evaluation_matrix_dimensions():
    matrix = generate_evaluation_matrix(Path(__file__).parents[1] / "configs")
    
    assert len(matrix) == 279
    
    primary_refs = matrix[matrix['run_id'].str.startswith('primary_')]
    sensitivity_refs = matrix[matrix['run_id'].str.startswith('sensitivity_')]
    
    assert len(primary_refs) == 90
    assert len(sensitivity_refs) == 189
    
    # 27 exact C1 aliases
    aliases = matrix[matrix['is_alias']]
    
    # All aliases must be C1
    assert all(aliases['controller_config_id'] == 'C1')
    
    # Alias targets must exist in primary
    alias_targets = aliases['alias_for_run_id'].unique()
    assert all(target in primary_refs['run_id'].values for target in alias_targets)

def test_unique_executions():
    matrix = generate_evaluation_matrix(Path(__file__).parents[1] / "configs")
    unique = get_unique_executions(matrix)
    
    # Exclude aliases
    assert len(unique) == 252
    assert not unique['is_alias'].any()
    
    # RS specific counts
    rs_matrix = matrix[matrix['defense_name'] == 'randomized_smoothing']
    rs_primary = rs_matrix[rs_matrix['run_id'].str.startswith('primary_')]
    assert len(rs_primary) == 30
    rs_sensitivity = rs_matrix[rs_matrix['run_id'].str.startswith('sensitivity_')]
    assert len(rs_sensitivity) == 63
    rs_aliases = rs_matrix[rs_matrix['is_alias']]
    assert len(rs_aliases) == 9
    rs_unique = get_unique_executions(rs_matrix)
    assert len(rs_unique) == 84

def test_matrix_yaml_consistency():
    """generate_evaluation_matrix now inherently checks full bidirectional consistency."""
    matrix = generate_evaluation_matrix(Path(__file__).parents[1] / "configs")
    assert len(matrix) == 279


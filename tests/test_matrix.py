import pytest
from recall_aware_ids.experiment.matrix import generate_evaluation_matrix, get_unique_executions

def test_evaluation_matrix_dimensions():
    matrix = generate_evaluation_matrix()
    
    # 90 primary references
    primary_refs = matrix[matrix['run_id'].str.startswith('primary_')]
    assert len(primary_refs) == 90
    
    # 189 sensitivity references
    sensitivity_refs = matrix[matrix['run_id'].str.startswith('sensitivity_')]
    assert len(sensitivity_refs) == 189
    
    # Total references
    assert len(matrix) == 279
    
    # 27 exact C1 aliases
    aliases = matrix[matrix['is_alias']]
    assert len(aliases) == 27
    
    # All aliases must be C1
    assert all(aliases['controller_config_id'] == 'C1')
    
    # Alias targets must exist in primary
    alias_targets = aliases['alias_for_run_id'].unique()
    assert all(target in primary_refs['run_id'].values for target in alias_targets)
    
    # 252 unique executions
    unique_runs = get_unique_executions(matrix)
    assert len(unique_runs) == 252
    
    # 36,288 unique batch records (252 executions * 144 batches)
    assert len(unique_runs) * 144 == 36288

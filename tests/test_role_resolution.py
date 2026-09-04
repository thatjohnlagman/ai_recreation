import pytest
import pandas as pd
import numpy as np
from recall_aware_ids.experiment.role_resolution import validate_and_resolve_roles

def test_randomized_role_assignments():
    # 90000 eval_positions, scrambled
    positions = np.arange(90000)
    np.random.shuffle(positions)
    
    metadata = pd.DataFrame({'eval_position': positions})
    
    # Randomly select 18000 for crafting, 72000 for measurement
    crafting_pos = positions[:18000]
    measurement_pos = positions[18000:]
    
    roles_df = pd.DataFrame({
        'eval_position': np.concatenate([crafting_pos, measurement_pos]),
        'role': ['crafting'] * 18000 + ['measurement'] * 72000
    })
    
    # Assign batch IDs to measurement_pos randomly but ensuring 500 each
    batches = np.repeat(np.arange(144), 500)
    np.random.shuffle(batches)
    
    batches_df = pd.DataFrame({
        'eval_position': measurement_pos,
        'batch_id': batches
    })
    
    resolved = validate_and_resolve_roles(metadata, roles_df, batches_df)
    
    assert len(resolved) == 72000
    assert list(resolved.columns) == ['measurement_idx', 'eval_position', 'batch_id']
    assert np.array_equal(resolved['measurement_idx'].values, np.arange(72000))
    # Check it is sorted deterministically
    assert np.array_equal(resolved['eval_position'].values, np.sort(resolved['eval_position'].values))

def test_rejections():
    positions = np.arange(90000)
    metadata = pd.DataFrame({'eval_position': positions})
    roles_df = pd.DataFrame({
        'eval_position': positions,
        'role': ['crafting'] * 18000 + ['measurement'] * 72000
    })
    batches_df = pd.DataFrame({
        'eval_position': positions[18000:],
        'batch_id': np.repeat(np.arange(144), 500)
    })
    
    # Unknown role
    bad_roles = roles_df.copy()
    bad_roles.loc[0, 'role'] = 'unknown'
    with pytest.raises(ValueError, match="Unknown role"):
        validate_and_resolve_roles(metadata, bad_roles, batches_df)
        
    # Extra role position
    bad_roles = pd.concat([roles_df, pd.DataFrame({'eval_position': [90000], 'role': ['measurement']})])
    with pytest.raises(ValueError, match="match metadata positions"):
        validate_and_resolve_roles(metadata, bad_roles, batches_df)
        
    # Extra batch position
    bad_batches = pd.concat([batches_df, pd.DataFrame({'eval_position': [90000], 'batch_id': [0]})])
    with pytest.raises(ValueError, match="Extra batch positions found"):
        validate_and_resolve_roles(metadata, roles_df, bad_batches)
        
    # Noninteger batch ID
    bad_batches = batches_df.copy()
    bad_batches['batch_id'] = bad_batches['batch_id'].astype(float)
    with pytest.raises(ValueError, match="batch_id must be an integer type"):
        validate_and_resolve_roles(metadata, roles_df, bad_batches)
        
    # Batch ID out of bounds
    bad_batches = batches_df.copy()
    bad_batches.loc[0, 'batch_id'] = 144
    with pytest.raises(ValueError, match="strictly 0..143"):
        validate_and_resolve_roles(metadata, roles_df, bad_batches)
        
    # Crafting with batch
    bad_batches = pd.concat([batches_df, pd.DataFrame({'eval_position': [0], 'batch_id': [0]})])
    with pytest.raises(ValueError, match="Crafting assignment contains a batch ID"):
        validate_and_resolve_roles(metadata, roles_df, bad_batches)
        
    # Measurement without batch
    bad_batches = batches_df.iloc[:-1]
    with pytest.raises(ValueError, match="Measurement assignment missing a batch ID"):
        validate_and_resolve_roles(metadata, roles_df, bad_batches)

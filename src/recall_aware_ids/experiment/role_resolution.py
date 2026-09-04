import pandas as pd
import numpy as np

def validate_and_resolve_roles(metadata_eval, roles_df, batches_df):
    """
    Explicit role-resolution layer that joins metadata, roles, and batches
    through the `eval_position` identifier.
    Returns a deterministic, measurement-only mapping with measurement_idx = 0..71999.
    """
    if "eval_position" not in metadata_eval.columns:
        raise ValueError("metadata_eval must contain 'eval_position'")
    if "eval_position" not in roles_df.columns:
        raise ValueError("roles_df must contain 'eval_position'")
    if "eval_position" not in batches_df.columns:
        raise ValueError("batches_df must contain 'eval_position'")
        
    if "role" not in roles_df.columns:
        raise ValueError("roles_df must contain 'role'")
        
    if "batch_id" not in batches_df.columns:
        raise ValueError("batches_df must contain 'batch_id'")

    # Ensure no duplicates
    if metadata_eval['eval_position'].duplicated().any():
        raise ValueError("Duplicate eval_position found in metadata")
    if roles_df['eval_position'].duplicated().any():
        raise ValueError("Duplicate eval_position found in roles")
    if batches_df['eval_position'].duplicated().any():
        raise ValueError("Duplicate eval_position found in batches")
        
    # Enforce exactly 90000 positions
    if len(metadata_eval) != 90000:
        raise ValueError(f"Expected exactly 90,000 unique evaluation positions, got {len(metadata_eval)}")
        
    # Verify no missing or extra metadata positions
    meta_positions = set(metadata_eval['eval_position'])
    role_positions = set(roles_df['eval_position'])
    batch_positions = set(batches_df['eval_position'])
    
    if meta_positions != role_positions:
        raise ValueError("Role positions must exactly match metadata positions (90000).")
        
    # Check valid roles
    valid_roles = {"crafting", "measurement"}
    invalid_roles = set(roles_df['role']) - valid_roles
    if invalid_roles:
        raise ValueError(f"Unknown role values found: {invalid_roles}")

    roles_count = roles_df['role'].value_counts()
    crafting_count = roles_count.get("crafting", 0)
    measurement_count = roles_count.get("measurement", 0)
    
    if crafting_count != 18000:
        raise ValueError(f"Expected exactly 18,000 crafting assignments, got {crafting_count}")
    if measurement_count != 72000:
        raise ValueError(f"Expected exactly 72,000 measurement assignments, got {measurement_count}")

    # Enforce batch ID types and domain
    batches_df = batches_df.copy()
    if batches_df['batch_id'].isnull().any():
        raise ValueError("batch_id cannot be null")
        
    if not pd.api.types.is_integer_dtype(batches_df['batch_id']):
        raise ValueError("batch_id must be an integer type")
        
    invalid_batches = set(batches_df['batch_id']) - set(range(144))
    if invalid_batches:
        raise ValueError(f"Batch IDs must be strictly 0..143. Found: {invalid_batches}")

    # Check that batches correspond exactly to measurement role
    merged = roles_df.merge(batches_df, on="eval_position", how="left")
    
    crafting_with_batch = merged[(merged['role'] == 'crafting') & (merged['batch_id'].notna())]
    if not crafting_with_batch.empty:
        raise ValueError("Cross-role identity: Crafting assignment contains a batch ID")
        
    measurement_without_batch = merged[(merged['role'] == 'measurement') & (merged['batch_id'].isna())]
    if not measurement_without_batch.empty:
        raise ValueError("Missing identity: Measurement assignment missing a batch ID")
        
    extra_batches = batch_positions - set(merged[merged['role'] == 'measurement']['eval_position'])
    if extra_batches:
         raise ValueError("Extra batch positions found that do not correspond to measurement role")

    # Enforce exactly 144 batches of 500
    measurement_batches = merged[merged['role'] == 'measurement']
    batch_counts = measurement_batches['batch_id'].value_counts()
    
    if len(batch_counts) != 144:
        raise ValueError(f"Expected exactly 144 unique batch IDs, got {len(batch_counts)}")
        
    if (batch_counts != 500).any():
        raise ValueError("Expected exactly 500 measurement positions per batch")

    # Sort deterministically
    measurement_df = measurement_batches.sort_values(by="eval_position").reset_index(drop=True)
    
    # Assign measurement_idx 0..71999
    measurement_df['measurement_idx'] = np.arange(72000, dtype=int)
    
    # We can also verify that the `metadata_eval` measurement subset aligns if needed,
    # but returning this canonical mapping is sufficient.
    
    # Ensure it only contains the required columns
    return measurement_df[['measurement_idx', 'eval_position', 'batch_id']]

import os
import sys
import tempfile
import json
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path so the script can import local modules
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# We will test the audit_dataset functions directly
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import audit_dataset


def test_extraction_pass():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        
        # Create 2 synthetic CSV files mimicking the structure
        file1 = temp_dir_path / "file1.csv"
        file2 = temp_dir_path / "file2.csv"
        
        # We will inject some invalid rows (e.g. NaN, duplicates)
        data1 = {
            "Label": ["Benign", "Attack", "Benign", "Attack", "Benign", "Attack", "Benign", "Label", "Benign"],
            "feat1": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, "feat1", 9.0], # Row index 2 is NaN, row index 7 is duplicate header
            "feat2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, "feat2", 90.0]
        }
        # Valid benign: index 0, 4, 6, 8 (4 rows)
        # Valid attack: index 1, 3, 5 (3 rows)
        
        data2 = {
            "Label": ["Attack", "Attack", "Benign", "Benign", "Attack"],
            "feat1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feat2": [10.0, 20.0, 30.0, 40.0, 50.0]
        }
        # Valid benign: index 2, 3 (2 rows)
        # Valid attack: index 0, 1, 4 (3 rows)
        
        pd.DataFrame(data1).to_csv(file1, index=False)
        pd.DataFrame(data2).to_csv(file2, index=False)
        
        file_stats = [
            {
                "file": "file1.csv",
                "valid_rows": 7,
                "class_counts": {"Benign": 4, "Attack": 3}
            },
            {
                "file": "file2.csv",
                "valid_rows": 5,
                "class_counts": {"Benign": 2, "Attack": 3}
            }
        ]
        
        allocations = {
            "file1__0": 2, # want 2 benign from file1
            "file1__1": 2, # want 2 attack from file1
            "file2__0": 1, # want 1 benign from file2
            "file2__1": 2  # want 2 attack from file2
        }
        
        # Override CHUNK_SIZE for test to simulate multiple chunks
        original_chunk_size = audit_dataset.CHUNK_SIZE
        audit_dataset.CHUNK_SIZE = 3
        
        try:
            sample_df = audit_dataset.extraction_pass(
                csv_files=[file1, file2],
                allocations=allocations,
                file_stats=file_stats,
                seed=42
            )
            
            # Check exact quota fulfillment
            assert len(sample_df) == sum(allocations.values()), f"Expected {sum(allocations.values())}, got {len(sample_df)}"
            
            # Check binary class constraints
            counts = sample_df.groupby(["_source_file", "y_binary"]).size().to_dict()
            assert counts.get(("file1", 0), 0) == allocations["file1__0"]
            assert counts.get(("file1", 1), 0) == allocations["file1__1"]
            assert counts.get(("file2", 0), 0) == allocations["file2__0"]
            assert counts.get(("file2", 1), 0) == allocations["file2__1"]
            
            # Check no duplicate selected records
            id_cols = ["_source_file", "_raw_row_idx"]
            assert not sample_df.duplicated(subset=id_cols).any(), "Found duplicate (source_file, raw_row_idx)"
            
            # Run again with same seed, check deterministic repetition
            sample_df_2 = audit_dataset.extraction_pass(
                csv_files=[file1, file2],
                allocations=allocations,
                file_stats=file_stats,
                seed=42
            )
            pd.testing.assert_frame_equal(sample_df, sample_df_2)
            
            # Run with different seed, should likely be different (given enough variance, though sample is small)
            sample_df_3 = audit_dataset.extraction_pass(
                csv_files=[file1, file2],
                allocations=allocations,
                file_stats=file_stats,
                seed=43
            )
            # Not strictly guaranteed to be different for very small samples, but highly likely. We just ensure it runs.
            
            print("All extraction tests passed!")
            
        finally:
            audit_dataset.CHUNK_SIZE = original_chunk_size


if __name__ == "__main__":
    test_extraction_pass()

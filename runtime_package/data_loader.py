"""
Data loader and profile manager for the standalone IDS demonstration platform.
Supports two data profiles:
1. 'expanded' (Default):
   - 90,000 evaluation rows (74,679 benign, 15,321 attack)
   - 18,000 crafting rows (used exclusively for surrogate fitting & boundary references)
   - 72,000 measurement rows (used exclusively for simulation targets in 144 batches of 500)
   - Implicit 0-based row index corresponds 1-to-1 with 'eval_position'
2. 'fixture20' (Regression / Fallback Profile):
   - 20 evaluation rows (10 benign, 10 attack)
   - Kept byte-for-byte in demo_data for existing readiness/audit tests
"""
import os
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_DIR = REPO_ROOT / "runtime_package"

class LoadedDataset:
    """Encapsulates the cached dataset and role manifests for a specific data profile."""
    def __init__(self, profile: str, X: pd.DataFrame, metadata: pd.DataFrame, roles_df: Optional[pd.DataFrame] = None, fingerprint: str = ""):
        self.profile = profile
        self.X = X
        self.metadata = metadata
        self.roles_df = roles_df
        self.fingerprint = fingerprint
        self.total_rows = len(X)

        y = metadata["y_binary"].values
        if profile == "expanded" and roles_df is not None:
            c_mask = (roles_df["role"].values == "crafting")
            m_mask = (roles_df["role"].values == "measurement")

            self.crafting_indices = roles_df.loc[c_mask, "eval_position"].values
            self.measurement_indices = roles_df.loc[m_mask, "eval_position"].values

            self.crafting_benign_indices = self.crafting_indices[y[self.crafting_indices] == 0]
            self.crafting_attack_indices = self.crafting_indices[y[self.crafting_indices] == 1]

            self.measurement_benign_indices = self.measurement_indices[y[self.measurement_indices] == 0]
            self.measurement_attack_indices = self.measurement_indices[y[self.measurement_indices] == 1]

            # DDoS specific measurement indices (LOIC-HTTP, HOIC, LOIC-UDP)
            ddos_families = {"DDoS attacks-LOIC-HTTP", "DDOS attack-HOIC", "DDOS attack-LOIC-UDP"}
            fams = metadata["attack_family"].values
            self.measurement_ddos_indices = np.array([
                idx for idx in self.measurement_attack_indices
                if str(fams[idx]) in ddos_families
            ], dtype=int)

            self.crafting_rows = len(self.crafting_indices)
            self.measurement_rows = len(self.measurement_indices)
            self._crafting_set = set(self.crafting_indices)
            self._measurement_set = set(self.measurement_indices)
        else:
            # fixture20 fallback
            all_idx = np.arange(len(X))
            self.crafting_indices = all_idx
            self.measurement_indices = all_idx
            self.crafting_benign_indices = np.where(y == 0)[0]
            self.crafting_attack_indices = np.where(y == 1)[0]
            self.measurement_benign_indices = self.crafting_benign_indices
            self.measurement_attack_indices = self.crafting_attack_indices
            self.measurement_ddos_indices = np.array([
                idx for idx in self.measurement_attack_indices
                if "DDoS" in str(metadata.iloc[idx].get("attack_family", ""))
            ], dtype=int)
            self.crafting_rows = len(all_idx)
            self.measurement_rows = len(all_idx)
            self._crafting_set = set(all_idx)
            self._measurement_set = set(all_idx)

    def is_crafting(self, sample_id: Optional[int]) -> bool:
        """Returns True if sample_id belongs to the crafting role."""
        if sample_id is None:
            return False
        if self.profile == "fixture20":
            return False  # All fixture flows treated normally unless explicitly marked
        return sample_id in self._crafting_set

    def is_measurement(self, sample_id: Optional[int]) -> bool:
        """Returns True if sample_id belongs to the measurement role."""
        if sample_id is None:
            return False
        return sample_id in self._measurement_set


_DATASET_CACHE: Dict[str, LoadedDataset] = {}

def get_file_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_dataset(profile: Optional[str] = None) -> LoadedDataset:
    """
    Loads and caches the specified dataset profile.
    If profile is None, reads environment variable IDS_DATA_PROFILE (defaults to 'expanded').
    """
    if profile is None:
        profile = os.environ.get("IDS_DATA_PROFILE", "expanded").strip().lower()

    if profile not in ["expanded", "fixture20"]:
        raise ValueError(f"Unknown data profile '{profile}'. Supported profiles are 'expanded' and 'fixture20'.")

    if profile in _DATASET_CACHE:
        return _DATASET_CACHE[profile]

    if profile == "expanded":
        exp_dir = RUNTIME_DIR / "expanded_data" / "data"
        x_path = exp_dir / "processed" / "X_eval.parquet"
        meta_path = exp_dir / "processed" / "metadata_eval.parquet"
        roles_path = exp_dir / "manifests" / "evaluation_roles.csv"

        if not x_path.exists() or not meta_path.exists() or not roles_path.exists():
            raise FileNotFoundError(
                f"Expanded dataset files missing in {exp_dir}. "
                f"Ensure ids_expanded_simulation_data.zip was extracted into {RUNTIME_DIR / 'expanded_data'}."
            )

        print(f"[DataLoader] Loading expanded dataset from {x_path.parent}...")
        X = pd.read_parquet(x_path)
        metadata = pd.read_parquet(meta_path)
        roles_df = pd.read_csv(roles_path)
        fingerprint = get_file_sha256(x_path)

        ds = LoadedDataset(profile="expanded", X=X, metadata=metadata, roles_df=roles_df, fingerprint=fingerprint)
        _DATASET_CACHE[profile] = ds
        print(f"[DataLoader] Loaded expanded profile: {ds.total_rows:,} rows ({ds.measurement_rows:,} measurement, {ds.crafting_rows:,} crafting).")
        return ds

    else:  # fixture20
        d_dir = RUNTIME_DIR / "demo_data"
        x_path = d_dir / "X_demo.parquet"
        meta_path = d_dir / "metadata_demo.parquet"

        if not x_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Fixture dataset files missing in {d_dir}.")

        print(f"[DataLoader] Loading fixture20 dataset from {d_dir}...")
        X = pd.read_parquet(x_path)
        metadata = pd.read_parquet(meta_path)
        fingerprint = get_file_sha256(x_path)

        ds = LoadedDataset(profile="fixture20", X=X, metadata=metadata, roles_df=None, fingerprint=fingerprint)
        _DATASET_CACHE[profile] = ds
        print(f"[DataLoader] Loaded fixture20 profile: {ds.total_rows} rows.")
        return ds

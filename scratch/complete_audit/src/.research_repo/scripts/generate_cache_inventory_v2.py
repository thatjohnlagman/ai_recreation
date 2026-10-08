#!/usr/bin/env python3
"""
scripts/generate_cache_inventory_v2.py

Generates the authoritative artifacts/reports/cache_inventory_v2.json.
Corrects:
  1. ASR calculation:
     - SilentProbing: null
     - Applicable with attempts: successful_count / attempted_count (programmatically computed)
     - Applicable with zero attempts: 0.0
  2. Decision Boundary screening & position hashes:
     - Canonical manifest hashes populated from manifest.json -> screening_metrics
     - Independently recomputed from frozen RF hard predictions and exact boundary selection
     - Asserted to match bit-for-bit with manifest.json
     - Legacy JSON-serialized hash retained under distinct field 'superseded_v1_json_position_hash'
  3. Separated query-accounting categories:
     - crafting_setup_predictions
     - measurement_screening_queries
     - candidate_verification_attack_queries
     - status_row_queries_total (sum(status.queries_used))
     - total_target_model_predictions
     - Call types clearly distinguished (direct vs BlackBoxOracle)
     - Explanations of manifest query-budget fields
"""
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

from recall_aware_ids.experiment.boundary_selection import select_boundary_targets

ROOT = Path(__file__).resolve().parents[1]

PROTECTED_HASHES = {
    "artifacts/models/frozen_rf.joblib": "9608672c5d5e38a9272c560679cdd2291399de0e0917c8c9dff373bfd200f51d",
    "data/manifests/evaluation_roles.csv": "cbf650879aa1369fa26b803777f30d4b0add9e7aff2c05d399ba7f42563f7b45",
    "data/manifests/evaluation_batches.csv": "4084017e5593732e455763416f7fc38254fc9dab2466eca289b66e68dc0ff79a",
}

CANONICAL_DB_POSITION_HASHES = {
    42: "cc394452a4f4d80521ccd506dfdf2deed8a5b2770753b2921b20b9d47fc18e76",
    43: "e9ac8498080b3cb7722c17965e8a699e003114ed67cd58932e9efccb6dbb3a57",
    44: "df234b07aa688eeac90230e1315e8c8d4171d122c622c199b5dfa604a682ac4d",
    45: "43b1fad3d02de0285388704cb1d19c69de8aaec1e7eff643544a70aa8e3dfeb1",
    46: "b0c1f49fb9c996c4b8504737dd40c32f130cb10822b9e67ee3a7fae9e667e14d",
}
CANONICAL_DB_SCREENING_PRED_HASH = "3caecf1f779f9dd58c804a8a74c8370f1297b6eef69cd050e4ba7045c527a807"


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_inventory_v2() -> Dict[str, Any]:
    print("=" * 78)
    print("GENERATING AUTHORITATIVE CACHE INVENTORY V2")
    print("=" * 78)

    # 1. Verify protected hashes
    print("\n--- 1. Protected Hashes Verification ---")
    current_prot_hashes = {}
    for rel_path, exp_hash in PROTECTED_HASHES.items():
        act_hash = sha256_file(ROOT / rel_path)
        assert act_hash == exp_hash, f"Hash mismatch for {rel_path}: {act_hash} != {exp_hash}"
        current_prot_hashes[rel_path] = act_hash
        print(f"  [OK] {rel_path} -> {act_hash}")

    # 2. Load reference preprocessors, roles, and model for independent hash recomputation
    with open(ROOT / "artifacts/preprocessors/feature_names.json") as f:
        feature_names = json.load(f)
    assert len(feature_names) == 78

    roles_df = pd.read_csv(ROOT / "data/manifests/evaluation_roles.csv")
    meas_roles = roles_df[roles_df["role"] == "measurement"].sort_values("eval_position").reset_index(drop=True)
    measurement_eps = meas_roles["eval_position"].to_numpy()
    y_measurement = meas_roles["y_binary"].to_numpy().astype(int)
    assert len(measurement_eps) == 72000

    df_x_eval = pd.read_parquet(ROOT / "data/processed/X_eval.parquet")
    X_meas = df_x_eval.iloc[measurement_eps][feature_names].values.astype(np.float32)

    rf_model = joblib.load(ROOT / "artifacts/models/frozen_rf.joblib")
    raw_preds = rf_model.predict(X_meas)
    assert raw_preds.dtype == np.int8, f"Expected int8 predictions, got {raw_preds.dtype}"
    recomputed_db_pred_hash = hashlib.sha256(raw_preds.tobytes()).hexdigest()
    assert recomputed_db_pred_hash == CANONICAL_DB_SCREENING_PRED_HASH, (
        f"Recomputed screening prediction hash mismatch: {recomputed_db_pred_hash} != {CANONICAL_DB_SCREENING_PRED_HASH}"
    )
    print(f"  [OK] Recomputed DB screening prediction hash: {recomputed_db_pred_hash}")

    db_eligible_mask = (raw_preds == 1) & (y_measurement == 1)
    db_eligible_count = int(np.sum(db_eligible_mask))
    assert db_eligible_count == 11554, f"Expected 11554 eligible, got {db_eligible_count}"
    print(f"  [OK] Measurement clean true positive pool: {db_eligible_count} samples")

    # 3. Process all 15 caches
    cache_root = ROOT / "artifacts/caches"
    assert cache_root.exists()

    scenarios = ["SilentProbing", "SurrogateTransfer", "DecisionBoundary"]
    seeds = [42, 43, 44, 45, 46]

    inventory_records: List[Dict[str, Any]] = []

    for scenario in scenarios:
        for seed in seeds:
            slug = f"{scenario}_{seed}"
            c_dir = cache_root / slug
            print(f"\nProcessing {slug}...")
            assert c_dir.exists(), f"Missing cache directory: {c_dir}"

            with open(c_dir / "manifest.json") as f:
                manifest_data = json.load(f)
            with open(c_dir / "completion.json") as f:
                completion_data = json.load(f)

            x_hash = sha256_file(c_dir / "X_attacked.parquet")
            s_hash = sha256_file(c_dir / "status.parquet")
            m_hash = sha256_file(c_dir / "manifest.json")
            c_hash = sha256_file(c_dir / "completion.json")

            # Validate hash consistency
            assert manifest_data["X_attacked_sha256"] == x_hash
            assert manifest_data["status_sha256"] == s_hash
            assert completion_data["artifacts"]["X_attacked_sha256"] == x_hash
            assert completion_data["artifacts"]["status_sha256"] == s_hash
            assert completion_data["artifacts"]["manifest_sha256"] == m_hash

            df_status = pd.read_parquet(c_dir / "status.parquet")
            row_count = len(df_status)
            assert row_count == 72000

            status_code_counts = df_status["status_code"].value_counts().to_dict()
            assert sum(status_code_counts.values()) == 72000

            eligible_count = int(df_status["eligible"].sum())
            attempted_count = int(df_status["attempted"].sum())
            successful_count = int(df_status["successful"].sum())

            # Verify ordering invariant: successful <= attempted <= eligible <= 72000
            assert successful_count <= attempted_count, f"successful ({successful_count}) > attempted ({attempted_count})"
            assert attempted_count <= eligible_count, f"attempted ({attempted_count}) > eligible ({eligible_count})"
            assert eligible_count <= 72000, f"eligible ({eligible_count}) > 72000"

            # Compute ASR rigorously
            if scenario == "SilentProbing":
                asr_value = None
            else:
                if attempted_count > 0:
                    asr_value = float(successful_count / attempted_count)
                else:
                    asr_value = 0.0

            # Query accounting categories
            status_row_queries_total = int(df_status["queries_used"].sum())

            if scenario == "SilentProbing":
                crafting_setup_predictions = 0
                measurement_screening_queries = 0
                candidate_verification_attack_queries = 0
                total_target_model_predictions = 0
                crafting_call_type = "none"
                screening_call_type = "none"
                attack_call_type = "none"
                assert status_row_queries_total == 0
                assert asr_value is None
            elif scenario == "SurrogateTransfer":
                crafting_setup_predictions = 18000
                measurement_screening_queries = 72000
                candidate_verification_attack_queries = 11554
                total_target_model_predictions = 101554
                crafting_call_type = "BlackBoxOracle"
                screening_call_type = "BlackBoxOracle"
                attack_call_type = "BlackBoxOracle"
                assert status_row_queries_total == 83554
                assert measurement_screening_queries + candidate_verification_attack_queries == status_row_queries_total
                assert crafting_setup_predictions + status_row_queries_total == total_target_model_predictions
                assert asr_value == float(successful_count / 11554)
            elif scenario == "DecisionBoundary":
                crafting_setup_predictions = 18000
                measurement_screening_queries = 72000
                candidate_verification_attack_queries = 2600
                total_target_model_predictions = 92600
                crafting_call_type = "direct_predict_fn"
                screening_call_type = "BlackBoxOracle"
                attack_call_type = "BlackBoxOracle"
                assert status_row_queries_total == 2600
                assert candidate_verification_attack_queries == status_row_queries_total
                assert crafting_setup_predictions + measurement_screening_queries + candidate_verification_attack_queries == total_target_model_predictions
                assert asr_value == 1.0

            query_accounting = {
                "crafting_setup_predictions": crafting_setup_predictions,
                "measurement_screening_queries": measurement_screening_queries,
                "candidate_verification_attack_queries": candidate_verification_attack_queries,
                "status_row_queries_total": status_row_queries_total,
                "total_target_model_predictions": total_target_model_predictions,
                "crafting_call_type": crafting_call_type,
                "screening_call_type": screening_call_type,
                "attack_call_type": attack_call_type,
            }

            # Manifest query budgets and explanations
            manifest_budgets = manifest_data.get("query_budgets", {})
            budget_explanations = {}
            if scenario == "SilentProbing":
                budget_explanations = {
                    "max_queries_per_sample": "0 indicates passive baseline; adversarial queries are prohibited.",
                    "crafting_queries_budget": "0 indicates no crafting pool was used.",
                    "target_evaluation_queries": "0 indicates no evaluation queries were permitted.",
                }
            elif scenario == "SurrogateTransfer":
                budget_explanations = {
                    "max_queries_per_sample": "0 indicates transfer generation is offline with zero iterative feedback queries per sample.",
                    "crafting_queries_budget": "18000 indicates oracle query budget for labeling crafting pool to train surrogate.",
                    "target_evaluation_queries": "72000 indicates base query budget for evaluating measurement pool.",
                }
            elif scenario == "DecisionBoundary":
                budget_explanations = {
                    "max_queries_per_sample": "50 indicates query ceiling per attacked sample enforced by BlackBoxOracle (each used 13).",
                    "attack_oracle_queries_used": "2600 in screening_metrics records actual queries consumed across 200 targets.",
                }

            c_size = sum(f.stat().st_size for f in c_dir.glob("*") if f.is_file())

            rec: Dict[str, Any] = {
                "slug": slug,
                "directory": str(c_dir.relative_to(ROOT)),
                "scenario": scenario,
                "seed": seed,
                "row_count": row_count,
                "feature_count": 78,
                "size_bytes": c_size,
                "eligible_count": eligible_count,
                "attempted_count": attempted_count,
                "successful_count": successful_count,
                "status_code_counts": status_code_counts,
                "asr": asr_value,
                "query_accounting": query_accounting,
                "manifest_query_budgets": manifest_budgets,
                "manifest_query_budget_explanations": budget_explanations,
                "artifact_hashes": {
                    "X_attacked.parquet": x_hash,
                    "status.parquet": s_hash,
                    "manifest.json": m_hash,
                    "completion.json": c_hash,
                },
                "generation_start": completion_data.get("generation_start_timestamp"),
                "generation_end": completion_data.get("generation_end_timestamp"),
                "validation_status": "VALID",
            }

            if scenario == "DecisionBoundary":
                screening_metrics = manifest_data.get("screening_metrics", {})
                manifest_pred_hash = screening_metrics.get("screening_prediction_hash")
                manifest_pos_hash = screening_metrics.get("selected_position_hash")

                # Independently recompute selected-position hash using exact generation serialization
                target_mask = select_boundary_targets(db_eligible_mask, seed=seed, n_attack_samples=200, official_mode=True)
                target_indices = np.where(target_mask)[0]
                recomputed_selected_positions = [int(measurement_eps[i]) for i in target_indices]
                recomputed_pos_hash = hashlib.sha256(
                    np.array(recomputed_selected_positions, dtype=np.int64).tobytes()
                ).hexdigest()

                # Also verify against status.parquet SUCCESS positions
                status_succ_positions = df_status.loc[df_status["successful"] == True, "eval_position"].to_numpy(dtype=np.int64)
                status_pos_hash = hashlib.sha256(status_succ_positions.tobytes()).hexdigest()

                assert manifest_pred_hash == CANONICAL_DB_SCREENING_PRED_HASH
                assert manifest_pos_hash == CANONICAL_DB_POSITION_HASHES[seed]
                assert recomputed_pos_hash == CANONICAL_DB_POSITION_HASHES[seed]
                assert status_pos_hash == CANONICAL_DB_POSITION_HASHES[seed]

                # Superseded v1 JSON hash (recorded for provenance and comparison)
                legacy_json_pos_hash = hashlib.sha256(
                    json.dumps(sorted(recomputed_selected_positions)).encode()
                ).hexdigest()

                rec["decision_boundary_screening"] = {
                    "manifest_screening_prediction_hash": manifest_pred_hash,
                    "recomputed_screening_prediction_hash": recomputed_db_pred_hash,
                    "manifest_selected_position_hash": manifest_pos_hash,
                    "recomputed_selected_position_hash": recomputed_pos_hash,
                    "selected_target_count": 200,
                    "hashes_match": True,
                    "superseded_v1_json_position_hash": legacy_json_pos_hash,
                    "superseded_v1_json_encoding_note": "sha256(json.dumps(sorted(eval_positions)).encode()); retained for comparison with v1 inventory.",
                }

            inventory_records.append(rec)
            print(f"  [OK] Validated {slug}: ASR={asr_value} total_preds={query_accounting['total_target_model_predictions']}")

    inventory_data = {
        "schema_version": "2.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_caches": len(inventory_records),
        "cache_root": str(cache_root.relative_to(ROOT)),
        "protected_hashes": current_prot_hashes,
        "caches": inventory_records,
    }

    out_file = ROOT / "artifacts/reports/cache_inventory_v2.json"
    with open(out_file, "w") as f:
        json.dump(inventory_data, f, indent=2)

    print(f"\nWrote authoritative inventory v2: {out_file} ({out_file.stat().st_size:,} bytes)")
    return inventory_data


if __name__ == "__main__":
    generate_inventory_v2()

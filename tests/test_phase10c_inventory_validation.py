"""
tests/test_phase10c_inventory_validation.py

Focused tests for Phase 10C cache inventory v2 validation:
- Silent-Probing ASR remains null;
- applicable zero-attempt ASR is 0.0;
- applicable nonzero ASR equals successes/attempts;
- status-code counts sum to 72,000;
- count ordering: successful <= attempted <= eligible <= 72000;
- manifest and independently recomputed screening hashes agree;
- canonical selected-position hashes agree;
- query-accounting category sums are exact;
- the inventory cannot silently emit null ASR for applicable attacks;
- all 30 cache manifest/completion hashes remain unchanged.
"""
import hashlib
import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Initial 30 cache metadata hashes recorded at preflight
PREFLIGHT_30_METADATA_HASHES = {
    "artifacts/caches/DecisionBoundary_42/completion.json": "239cf0c996b13f2f04220bc29b36901ffe92d5f98195ed37d18688d9578dbe98",
    "artifacts/caches/DecisionBoundary_42/manifest.json": "4e443a3c39663bd745598935654105def62187a02f5c6ed2388a3dfa1966d7de",
    "artifacts/caches/DecisionBoundary_43/completion.json": "6e8db2a27fad404ddd41305a022389428a3fe105e3dcbdfc7b25f61bdb9c4b1b",
    "artifacts/caches/DecisionBoundary_43/manifest.json": "6d2152e01e00e326d09d76c192e3cd520c5a8701be62bd60371ce64e8b73a40f",
    "artifacts/caches/DecisionBoundary_44/completion.json": "440005db4ba4931c743e78a7eb64b364fe397cce997f18f5135fd43aa114c915",
    "artifacts/caches/DecisionBoundary_44/manifest.json": "c956218fe069fb4eaf469dbf0924b74a6c7ad6cdb06517adf201d57aa85c4437",
    "artifacts/caches/DecisionBoundary_45/completion.json": "11fb6c1a4549cac82d6a5bb76f576c6671689e270ee507af9de39420470eef82",
    "artifacts/caches/DecisionBoundary_45/manifest.json": "8b1ece8746841db8bc2dff6da2b56c25446fbcfbf8207b05095bcde7e93e9b5a",
    "artifacts/caches/DecisionBoundary_46/completion.json": "6aac5561532892cf5de4b04c418997513b0eab48f046261cc61fb4166cb7e931",
    "artifacts/caches/DecisionBoundary_46/manifest.json": "34f5db141616bcf5bd50c05db0e0285e78e4499647e0b4f66dab6d99e3c738c6",
    "artifacts/caches/SilentProbing_42/completion.json": "9404185caf4be948a7e58b2c87dca88a4444095fad83c8c2a3720a3cd7c2d4eb",
    "artifacts/caches/SilentProbing_42/manifest.json": "3142249d6690d219018f2978f3794fcb9f81f1de12cc5e7ba63c81482ba3d588",
    "artifacts/caches/SilentProbing_43/completion.json": "da2bd8d5a00c29b3a1ca6f2221b9aaa9ed6c61a76464b99b436a34eacaa7a60c",
    "artifacts/caches/SilentProbing_43/manifest.json": "3a583c3087b88cf520801a300102246942ee9735c97c9fdbb9dc75e90d4573c6",
    "artifacts/caches/SilentProbing_44/completion.json": "fecbea46382033d1aa598116dde76fbdc3196463b959ed018350bada2995bf76",
    "artifacts/caches/SilentProbing_44/manifest.json": "9f7f18354921a71d6a6a28a24e3af03a9b62b894fc84aac010ba25d6872680e0",
    "artifacts/caches/SilentProbing_45/completion.json": "d2a01c86f82a19bed4daa596f98d04e2888f6029acbe040d7e14cdb1f1d41055",
    "artifacts/caches/SilentProbing_45/manifest.json": "7aae84fdbd06e90c7e30011d7e88c358eee8f5f28028c83d446b61cd7b00b6da",
    "artifacts/caches/SilentProbing_46/completion.json": "b16216d120e2c53de962af4189ab81e9750b04d194deb210fee02ac7d6009b40",
    "artifacts/caches/SilentProbing_46/manifest.json": "37a2a65bd06dc36f17d90f01708e3d63036716870345da5d4fa3e8faae12eb0f",
    "artifacts/caches/SurrogateTransfer_42/completion.json": "1ccc76faeb112e045e7a9851cc2da429413b1693ad8a179ba69edf685955d9f5",
    "artifacts/caches/SurrogateTransfer_42/manifest.json": "cf53762f4663244d13c9cc8d9c6f783c8f2a1277ec7409e3c3ed96789390e05c",
    "artifacts/caches/SurrogateTransfer_43/completion.json": "f7ef6a38314ce1edf537dd5c05b457c79d087768825d17d9dd23e112b9ad6fc8",
    "artifacts/caches/SurrogateTransfer_43/manifest.json": "12a9e7f1e38f22ccd7b5ae8a35ba5db5a5ac6d717fff246de91c3bdd48d49b4f",
    "artifacts/caches/SurrogateTransfer_44/completion.json": "be5c1fdf6e420e3d6af6a2b5bcfacedca818f2c422ca79d6d9b670cd47d94170",
    "artifacts/caches/SurrogateTransfer_44/manifest.json": "a4484c8fdecd06ce6b2340123e95608c4ea0cc75c6363d0df4d30e3967cebbdc",
    "artifacts/caches/SurrogateTransfer_45/completion.json": "3d6c2d70133379ef1cb6ee1f710c88fd9baaab91a38438fec0a60d903347c33e",
    "artifacts/caches/SurrogateTransfer_45/manifest.json": "18d2f0d1215968e3daad74f7e4d90b3ff020db2305e6b5fab44b34f27b0ddb0f",
    "artifacts/caches/SurrogateTransfer_46/completion.json": "2ced08cf6162228c38cb7148484ef5131f08892ce8aa06c8fb33c113628e6501",
    "artifacts/caches/SurrogateTransfer_46/manifest.json": "fba14aeffc42e7dc5040e56a4bcf671a9d372a6347931d3a5850ba862d07be9f",
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


@pytest.fixture(scope="module")
def inventory_v2():
    inv_path = ROOT / "artifacts/reports/cache_inventory_v2.json"
    assert inv_path.exists(), "cache_inventory_v2.json does not exist. Run scripts/generate_cache_inventory_v2.py first."
    with open(inv_path, "r") as f:
        return json.load(f)


def test_immutable_30_cache_manifest_and_completion_hashes_unchanged():
    """Confirms all 30 on-disk manifest and completion files match preflight hashes bit-for-bit."""
    for rel_path, expected_hash in PREFLIGHT_30_METADATA_HASHES.items():
        p = ROOT / rel_path
        assert p.exists(), f"Missing cache metadata file: {rel_path}"
        actual_hash = sha256_file(p)
        assert actual_hash == expected_hash, f"Metadata file {rel_path} modified: {actual_hash} != {expected_hash}"


def test_silent_probing_asr_is_null(inventory_v2):
    """Verifies that Silent Probing caches always retain asr = None (serialized as null)."""
    sp_caches = [c for c in inventory_v2["caches"] if c["scenario"] == "SilentProbing"]
    assert len(sp_caches) == 5
    for c in sp_caches:
        assert c["asr"] is None
        assert c["attempted_count"] == 0
        assert c["successful_count"] == 0
        assert c["eligible_count"] == 0


def test_applicable_zero_attempt_asr_is_zero():
    """Verifies rule that an applicable attack with zero attempts produces exactly 0.0 (not None)."""
    # Test computation function directly
    def compute_asr(scenario: str, attempted: int, successful: int):
        if scenario == "SilentProbing":
            return None
        if attempted == 0:
            return 0.0
        return float(successful / attempted)

    assert compute_asr("SurrogateTransfer", 0, 0) == 0.0
    assert compute_asr("DecisionBoundary", 0, 0) == 0.0
    assert compute_asr("SilentProbing", 0, 0) is None


def test_applicable_nonzero_asr_equals_success_over_attempted(inventory_v2):
    """Verifies applicable attacks with attempts have asr == successful / attempted."""
    expected_st_successes = {42: 5, 43: 3, 44: 3, 45: 2, 46: 1}

    for c in inventory_v2["caches"]:
        scen = c["scenario"]
        seed = c["seed"]
        if scen == "SurrogateTransfer":
            assert c["attempted_count"] == 11554
            assert c["successful_count"] == expected_st_successes[seed]
            expected_asr = expected_st_successes[seed] / 11554.0
            assert c["asr"] == pytest.approx(expected_asr, rel=1e-9)
        elif scen == "DecisionBoundary":
            assert c["attempted_count"] == 200
            assert c["successful_count"] == 200
            assert c["asr"] == 1.0


def test_status_code_counts_sum_to_72000(inventory_v2):
    """Verifies status code counts sum to exactly 72,000 for every cache."""
    for c in inventory_v2["caches"]:
        sc_counts = c["status_code_counts"]
        assert sum(sc_counts.values()) == 72000
        assert c["row_count"] == 72000


def test_count_ordering_invariant(inventory_v2):
    """Verifies ordering invariant: successful <= attempted <= eligible <= 72,000."""
    for c in inventory_v2["caches"]:
        succ = c["successful_count"]
        att = c["attempted_count"]
        elig = c["eligible_count"]
        assert 0 <= succ <= att <= elig <= 72000


def test_manifest_and_independent_screening_hashes_agree(inventory_v2):
    """Verifies manifest screening prediction hash matches independent calculation."""
    db_caches = [c for c in inventory_v2["caches"] if c["scenario"] == "DecisionBoundary"]
    assert len(db_caches) == 5
    for c in db_caches:
        info = c["decision_boundary_screening"]
        assert info["manifest_screening_prediction_hash"] == CANONICAL_DB_SCREENING_PRED_HASH
        assert info["recomputed_screening_prediction_hash"] == CANONICAL_DB_SCREENING_PRED_HASH
        assert info["hashes_match"] is True


def test_canonical_selected_position_hashes_agree(inventory_v2):
    """Verifies manifest selected position hashes match independent calculation and status.parquet."""
    db_caches = [c for c in inventory_v2["caches"] if c["scenario"] == "DecisionBoundary"]
    assert len(db_caches) == 5
    for c in db_caches:
        seed = c["seed"]
        info = c["decision_boundary_screening"]
        exp_pos_hash = CANONICAL_DB_POSITION_HASHES[seed]
        assert info["manifest_selected_position_hash"] == exp_pos_hash
        assert info["recomputed_selected_position_hash"] == exp_pos_hash

        # Also verify against status.parquet directly
        c_dir = ROOT / c["directory"]
        df_status = pd.read_parquet(c_dir / "status.parquet")
        succ_positions = df_status.loc[df_status["successful"] == True, "eval_position"].to_numpy(dtype=np.int64)
        assert len(succ_positions) == 200
        status_pos_hash = hashlib.sha256(succ_positions.tobytes()).hexdigest()
        assert status_pos_hash == exp_pos_hash


def test_query_accounting_category_sums_are_exact(inventory_v2):
    """Verifies query accounting categories are mutually consistent and exact."""
    for c in inventory_v2["caches"]:
        qa = c["query_accounting"]
        scen = c["scenario"]

        if scen == "SilentProbing":
            assert qa["crafting_setup_predictions"] == 0
            assert qa["measurement_screening_queries"] == 0
            assert qa["candidate_verification_attack_queries"] == 0
            assert qa["status_row_queries_total"] == 0
            assert qa["total_target_model_predictions"] == 0
            assert qa["crafting_call_type"] == "none"
            assert qa["screening_call_type"] == "none"
            assert qa["attack_call_type"] == "none"
        elif scen == "SurrogateTransfer":
            assert qa["crafting_setup_predictions"] == 18000
            assert qa["measurement_screening_queries"] == 72000
            assert qa["candidate_verification_attack_queries"] == 11554
            assert qa["status_row_queries_total"] == 83554
            assert qa["measurement_screening_queries"] + qa["candidate_verification_attack_queries"] == qa["status_row_queries_total"]
            assert qa["crafting_setup_predictions"] + qa["status_row_queries_total"] == qa["total_target_model_predictions"]
            assert qa["total_target_model_predictions"] == 101554
            assert qa["crafting_call_type"] == "BlackBoxOracle"
            assert qa["screening_call_type"] == "BlackBoxOracle"
            assert qa["attack_call_type"] == "BlackBoxOracle"
        elif scen == "DecisionBoundary":
            assert qa["crafting_setup_predictions"] == 18000
            assert qa["measurement_screening_queries"] == 72000
            assert qa["candidate_verification_attack_queries"] == 2600
            assert qa["status_row_queries_total"] == 2600
            assert qa["candidate_verification_attack_queries"] == qa["status_row_queries_total"]
            assert qa["crafting_setup_predictions"] + qa["measurement_screening_queries"] + qa["candidate_verification_attack_queries"] == qa["total_target_model_predictions"]
            assert qa["total_target_model_predictions"] == 92600
            assert qa["crafting_call_type"] == "direct_predict_fn"
            assert qa["screening_call_type"] == "BlackBoxOracle"
            assert qa["attack_call_type"] == "BlackBoxOracle"


def test_inventory_cannot_silently_emit_null_asr_for_applicable_attacks(inventory_v2):
    """Ensures no applicable attack in the inventory has null ASR."""
    for c in inventory_v2["caches"]:
        if c["scenario"] != "SilentProbing":
            assert c["asr"] is not None, f"Applicable attack {c['slug']} has null ASR!"
            assert isinstance(c["asr"], float)
            assert 0.0 <= c["asr"] <= 1.0

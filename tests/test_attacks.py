import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recall_aware_ids.attacks.oracle import BlackBoxOracle, BudgetExhaustedFailure, BatchBudgetFailure, DuplicateSampleIDsFailure
from recall_aware_ids.attacks.base import BaseAttack, AttackResult
from recall_aware_ids.attacks.silent_probing import SilentProbingAttack
from recall_aware_ids.attacks.surrogate_transfer import SurrogateTransferAttack
from recall_aware_ids.attacks.boundary_attack import DecisionBoundaryAttack

FEATURE_NAMES = [f"feat_{i}" for i in range(78)]
MODIFIABLE_MASK = [True] * 70 + [False] * 8

TRAIN_BOUNDS = pd.DataFrame(
    {"min": [-100.0] * 78, "max": [100.0] * 78},
    index=FEATURE_NAMES
)

def _dummy_predict(X):
    if not isinstance(X, np.ndarray): X = np.array(X)
    return (np.sum(X, axis=1) > 10.0).astype(int)

# --- Oracle Tests ---

def test_oracle_duplicate_id_rejection():
    called = False
    def spy_predict(X):
        nonlocal called
        called = True
        return np.ones(X.shape[0])
        
    oracle = BlackBoxOracle(spy_predict, max_queries_per_sample=10)
    X = np.zeros((3, 78))
    
    # "id1" string duplicated
    res = oracle.predict(X, sample_ids=["id1", "id2", "id1"])
    assert isinstance(res, DuplicateSampleIDsFailure)
    assert "id1" in res.duplicates
    
    # tuple id duplicated
    res_tuple = oracle.predict(X, sample_ids=[(1, 2), (1, 3), (1, 2)])
    assert isinstance(res_tuple, DuplicateSampleIDsFailure)
    assert (1, 2) in res_tuple.duplicates
    
    assert oracle.global_query_count == 0
    assert oracle.get_query_count("id1") == 0
    assert not called

def test_oracle_batch_atomic_rejection():
    called = 0
    def spy_predict(X):
        nonlocal called
        called += 1
        return np.ones(X.shape[0])
        
    oracle = BlackBoxOracle(spy_predict, max_queries_per_sample=2)
    X = np.zeros((2, 78))
    
    oracle.predict(X, sample_ids=["id1", "id2"], stage="s1")
    oracle.predict(X, sample_ids=["id1", "id2"], stage="s2")
    assert called == 2
    
    X3 = np.zeros((3, 78))
    # Batch should be rejected entirely because id1 is out of budget
    res = oracle.predict(X3, sample_ids=["id1", "id3", "id4"], stage="s3")
    assert isinstance(res, BatchBudgetFailure)
    assert "id1" in res.exhausted_ids
    
    # No queries added, model not called
    assert called == 2
    assert oracle.global_query_count == 4
    assert oracle.get_query_count("id3") == 0

def test_oracle_history_and_counts():
    oracle = BlackBoxOracle(_dummy_predict, max_queries_per_sample=10)
    oracle.predict(np.zeros((1, 78)), sample_ids=["id1"], stage="eligibility")
    oracle.predict(np.zeros((1, 78)), sample_ids=["id1"], stage="endpoint")
    
    assert oracle.global_query_count == 2
    assert oracle.get_query_count("id1") == 2
    assert oracle.per_sample_history["id1"] == ["eligibility", "endpoint"]

# --- BaseAttack Tests ---

def test_base_attack_feature_order_and_bounds_validation():
    bad_bounds = TRAIN_BOUNDS.copy()
    bad_bounds.index = [f"feat_{i}" for i in range(77, -1, -1)]
    with pytest.raises(ValueError, match="training_bounds index does not perfectly match feature_names order"):
        BaseAttack(FEATURE_NAMES, MODIFIABLE_MASK, bad_bounds)

def test_base_attack_nonfinite_rejection():
    base = BaseAttack(FEATURE_NAMES, MODIFIABLE_MASK, TRAIN_BOUNDS)
    X_orig = np.zeros(78, dtype=np.float32)
    X_adv = np.zeros(78, dtype=np.float32)
    X_adv[0] = np.nan
    with pytest.raises(ValueError, match="Non-finite values encountered in input"):
        base.project_and_clip(X_adv, X_orig)

def test_base_attack_projection_immutability_all_protected():
    base = BaseAttack(FEATURE_NAMES, MODIFIABLE_MASK, TRAIN_BOUNDS)
    X_orig = np.zeros(78, dtype=np.float32)
    X_orig[70:] = np.arange(10, 18, dtype=np.float32)
    
    X_adv = np.ones(78, dtype=np.float32) * 200.0
    
    X_proj = base.project_and_clip(X_adv, X_orig)
    
    # Input immutability
    assert X_adv[0] == 200.0 
    assert X_adv[75] == 200.0
    
    assert X_proj.dtype == np.float32
    assert np.isfinite(X_proj).all()
    assert X_proj.shape == (78,)
    
    # Modifiable clipped to inclusive bounds (100.0)
    assert X_proj[0] == 100.0
    
    # Every protected feature perfectly restored
    assert np.array_equal(X_proj[70:], np.arange(10, 18, dtype=np.float32))

def test_base_attack_float64_magnitude_regression():
    base = BaseAttack(FEATURE_NAMES, MODIFIABLE_MASK, TRAIN_BOUNDS)
    
    X_orig = np.zeros(78, dtype=np.float32)
    X_adv = np.zeros(78, dtype=np.float32)
    
    X_adv[0] = np.float32(1e10)
    X_orig[0] = np.float32(1e4)
    
    mags = base.calculate_magnitudes(X_adv, X_orig)
    
    # float64 subtraction
    diff_64 = np.float64(np.float32(1e10)) - np.float64(np.float32(1e4))
    assert np.isclose(mags["l1"], np.abs(diff_64))
    
    # float32 subtraction loses precision
    diff_32 = np.float32(1e10) - np.float32(1e4)
    assert diff_64 != diff_32

# --- Surrogate Transfer Tests ---

def test_surrogate_nextafter_directions():
    att = SurrogateTransferAttack(FEATURE_NAMES, MODIFIABLE_MASK, TRAIN_BOUNDS)
    
    # 4 cases
    # 1. inclusive branch where t32 already satisfies threshold
    path1 = [(0, "<=", 50.0)]
    att.generate_candidate = lambda x: (None, None)
    intervals1 = att._resolve_intervals(path1)
    assert intervals1[0][1] == np.float32(50.0)
    
    # 2. inclusive branch where rounded t32 > threshold, requiring nextafter(-inf)
    # We want a t_f64 such that np.float32(t_f64) > t_f64.
    # 0.1 didn't work. Let's use an explicit float64 that is slightly below a float32 exact value.
    t_f32 = np.float32(50.000004)
    t_f64 = np.float64(50.000003) 
    # Force the surrogate dummy model to have t_f64 as threshold, but when cast to float32 it rounds up (let's just mock it)
    # The method _resolve_intervals takes path. path has threshold as float64.
    # If path has 50.000003, t32 becomes 50.000004. Then 50.000004 <= 50.000003 is False.
    # Therefore it will use nextafter(-inf).
    path2 = [(0, "<=", t_f64)]
    intervals2 = att._resolve_intervals(path2)
    assert intervals2[0][1] == np.nextafter(np.float32(t_f64), -np.inf, dtype=np.float32)
    
    # 3. strict branch where t32 > threshold already
    path3 = [(0, ">", t_f64)]
    intervals3 = att._resolve_intervals(path3)
    # Since t_f32 > t_f64, it already satisfies ">", no nextafter needed
    assert intervals3[0][0] == t_f32
    
    # 4. strict branch where t32 <= threshold, requiring nextafter(+inf)
    path4 = [(0, ">", 50.0)]
    intervals4 = att._resolve_intervals(path4)
    # 50.0 == 50.0, so it's not strictly >. We need nextafter(+inf)
    assert intervals4[0][0] == np.nextafter(np.float32(50.0), np.inf, dtype=np.float32)

def test_surrogate_contradictory_constraints_and_bounds_rejection():
    oracle = BlackBoxOracle(lambda x: np.ones(len(x)), max_queries_per_sample=10)
    att = SurrogateTransferAttack(FEATURE_NAMES, MODIFIABLE_MASK, TRAIN_BOUNDS)
    
    class MockTree:
        children_left = [1, -1, -1]
        children_right = [2, -1, -1]
        feature = [0, -2, -2]
        threshold = [50.0, -2.0, -2.0]
        value = [[[0, 1]], [[1, 0]], [[1, 0]]] 
    
    class MockSurrogate:
        tree_ = MockTree()
        classes_ = np.array([0, 1])
        def predict(self, X): return np.zeros(X.shape[0])
        
    att.surrogate = MockSurrogate()
    att.benign_class_idx = 0
    
    att._extract_benign_paths = lambda: [
        (1, [(0, "<=", 10.0), (0, ">", 20.0)]), 
        (2, [(75, "<=", -10.0)]) 
    ]
    
    X_orig = np.zeros(78, dtype=np.float32)
    X_orig[75] = 0.0 
    
    X_cand, mags = att.generate_candidate(X_orig)
    assert X_cand is None
    assert mags == "NO_FEASIBLE_CANDIDATE"
    
def test_surrogate_lexicographic_tie_breaking():
    att = SurrogateTransferAttack(FEATURE_NAMES, MODIFIABLE_MASK, TRAIN_BOUNDS)
    
    class MockSurrogate:
        tree_ = None
        classes_ = np.array([0, 1])
        def predict(self, X): return np.zeros(X.shape[0])
        
    att.surrogate = MockSurrogate()
    att.benign_class_idx = 0
    
    att._extract_benign_paths = lambda: [
        (1, [(0, ">", 10.0)]), # L0=1, L2=10, Linf=10
        (2, [(1, ">", 5.0)]),  # L0=1, L2=5, Linf=5 -> SHOULD WIN
        (3, [(0, ">", 5.0), (1, ">", 5.0)]), # L0=2
    ]
    
    X_orig = np.zeros(78, dtype=np.float32)
    X_cand, mags = att.generate_candidate(X_orig)
    
    assert mags["l0"] == 1
    assert mags["linf"] == np.nextafter(np.float32(5.0), np.inf, dtype=np.float32)

def test_surrogate_same_seed_deterministic():
    X_pool = np.vstack([np.zeros((50, 78)), np.ones((50, 78))])
    y_pool = np.array([0]*50 + [1]*50)
    
    att1 = SurrogateTransferAttack(FEATURE_NAMES, MODIFIABLE_MASK, TRAIN_BOUNDS, effective_seed=123)
    att1.fit_surrogate(X_pool, y_pool)
    path1 = att1._extract_benign_paths()
    
    att2 = SurrogateTransferAttack(FEATURE_NAMES, MODIFIABLE_MASK, TRAIN_BOUNDS, effective_seed=123)
    att2.fit_surrogate(X_pool, y_pool)
    path2 = att2._extract_benign_paths()
    
    assert path1 == path2
    
def test_surrogate_full_paths_target_rejection():
    # true target-rejection transfer behavior
    oracle = BlackBoxOracle(lambda X: (X[:, 0] > 50).astype(int), max_queries_per_sample=10)
    att = SurrogateTransferAttack(FEATURE_NAMES, MODIFIABLE_MASK, TRAIN_BOUNDS)
    
    # Fit so that surrogate thinks feat_1 <= 50 is benign
    X_pool_benign = np.zeros((50, 78), dtype=np.float32)
    X_pool_attack = np.zeros((50, 78), dtype=np.float32)
    X_pool_attack[:, 1] = 100.0 # Surrogate will use feat_1
    X_pool = np.vstack([X_pool_benign, X_pool_attack])
    y_pool = (X_pool[:, 1] > 50).astype(int)
    att.fit_surrogate(X_pool, y_pool)
    
    # But target model uses feat_0 > 50.
    X_attack = np.zeros(78, dtype=np.float32)
    X_attack[0] = 100.0 # attack according to target model
    X_attack[1] = 100.0 # attack according to surrogate
    
    # Transfer fails because candidate will only change feat_1
    res = att.evaluate_transfer(None, X_attack, oracle, "id_tr", true_label=1)
    
    # The true evaluation flow:
    X_cand, mags = att.generate_candidate(X_attack)
    assert X_cand[1] <= 50.0
    assert X_cand[0] == 100.0 # Target sees this as attack
    
    res = att.evaluate_transfer(X_cand, X_attack, oracle, "id_tr2", true_label=1)
    assert res.status_code == "TARGET_REJECTION"

# --- Boundary Attack Tests ---

def test_boundary_mismatch_and_unbounded_oracle():
    oracle = BlackBoxOracle(_dummy_predict) # unbounded
    att = DecisionBoundaryAttack(FEATURE_NAMES, MODIFIABLE_MASK, TRAIN_BOUNDS, max_queries=50)
    
    with pytest.raises(ValueError, match="mismatches attack limit"):
        att.generate(np.zeros(78), oracle, "id", 1, [])
        
    oracle2 = BlackBoxOracle(_dummy_predict, max_queries_per_sample=10)
    with pytest.raises(ValueError, match="mismatches attack limit"):
        att.generate(np.zeros(78), oracle2, "id", 1, [])

def test_boundary_exhaustion_before_39():
    oracle = BlackBoxOracle(_dummy_predict, max_queries_per_sample=50)
    att = DecisionBoundaryAttack(FEATURE_NAMES, MODIFIABLE_MASK, TRAIN_BOUNDS, max_queries=50)
    
    X_attack = np.ones(78, dtype=np.float32) * 100.0
    refs = [np.ones(78, dtype=np.float32) * 100.0 for _ in range(5)]
    
    res = att.generate(X_attack, oracle, "id1", 1, refs)
    assert res.status_code == "NO_FEASIBLE_CANDIDATE"
    assert res.query_count == 6

def test_boundary_39_query_reservation():
    oracle = BlackBoxOracle(_dummy_predict, max_queries_per_sample=50)
    att = DecisionBoundaryAttack(FEATURE_NAMES, MODIFIABLE_MASK, TRAIN_BOUNDS, max_queries=50)
    
    X_attack = np.ones(78, dtype=np.float32) * 100.0
    refs = [np.ones(78, dtype=np.float32) * 100.0 for _ in range(40)]
    
    res = att.generate(X_attack, oracle, "id1", 1, refs)
    assert res.status_code == "INSUFFICIENT_BUDGET_FOR_FULL_SEARCH"
    assert res.query_count == 39

def test_boundary_successful_m_plus_12():
    # zero-initialized attack sample with one modifiable feature set to 100
    # failed references with same attack value
    # valid all-zero benign reference
    # protected features equal between attack sample and benign reference.
    oracle = BlackBoxOracle(lambda X: (X[:, 0] > 50).astype(int), max_queries_per_sample=50)
    att = DecisionBoundaryAttack(FEATURE_NAMES, MODIFIABLE_MASK, TRAIN_BOUNDS, max_queries=50, binary_search_steps=10)
    
    X_attack = np.zeros(78, dtype=np.float32)
    X_attack[0] = 100.0 
    # Protected features are all zero, which matches the benign ref.
    
    bad_ref = np.zeros(78, dtype=np.float32)
    bad_ref[0] = 100.0
    
    good_ref = np.zeros(78, dtype=np.float32)
    
    refs = [bad_ref] * 37 + [good_ref]
    
    res = att.generate(X_attack, oracle, "id1", 1, refs)
    assert res.status_code == "SUCCESS"
    assert res.query_count == 50
    
    # 38 endpoint screenings, 10 midpoint queries, 1 final verification
    history = oracle.per_sample_history["id1"]
    assert history.count("endpoint_screening") == 38
    for i in range(10):
        assert f"binary_search_step_{i}" in history
    assert history[-1] == "verification"
    
    # final prediction benign
    assert oracle.predict(res.X_adv.reshape(1, -1), ["id_check"], "check")[0] == 0
    
    # all protected features unchanged
    assert np.array_equal(res.X_adv[70:], X_attack[70:])
    
    # final finite float32
    assert res.X_adv.dtype == np.float32
    assert np.isfinite(res.X_adv).all()

def test_negative_properties():
    # explain predict_proba and estimators_ scan matches as intentional negative assertions
    oracle = BlackBoxOracle(_dummy_predict, 10)
    assert not hasattr(oracle, "predict_proba")
    assert not hasattr(oracle, "estimators_")

if __name__ == "__main__":
    pytest.main(["-q", __file__])

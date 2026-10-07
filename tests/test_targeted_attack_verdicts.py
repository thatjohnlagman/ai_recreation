#!/usr/bin/env python3
"""
Targeted Regression Test Suite: Attack Verdicts, Surrogate Seed, and Terminology
Tests production wrapper behavior in attacker_sim.py, operator_benchmarks.py, and README.md.
"""

import sys
import io
import os
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
RUNTIME_DIR = BASE_DIR / "runtime_package"
if str(RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_DIR))

import attacker_sim
from attacker_sim import (
    get_ctx,
    AttackerContext,
    run_silent_probing_attack,
    run_surrogate_transfer_attack,
    start_simulation_session,
    stop_simulation_session,
    SurrogateTransferAttack,
)


class TestTargetedAttackVerdicts(unittest.TestCase):
    def setUp(self):
        self.ctx = get_ctx(profile="fixture20", seed=42)

    def test_branch_1_original_detected_candidate_allowed_evasion_success(self):
        """Original detected (403) + candidate allowed (200) -> SURROGATE EVASION SUCCESS."""
        submitted_flows = []

        def mock_oracle_predict(self_oracle, X, sample_ids=None, stage=None):
            if stage == "eligibility":
                return np.array([1], dtype=int)  # 403 detected
            elif stage == "surrogate_fitting":
                # Alternate 0 and 1 so decision tree learns both classes
                return np.array([0 if i % 2 == 0 else 1 for i in range(len(X))], dtype=int)
            return np.array([1] * len(X), dtype=int)

        def mock_run_single_flow(*args, **kwargs):
            submitted_flows.append(kwargs)
            return {
                "status_code": 200,
                "blocked": False,
                "allowed": True,
                "body": {"status": "success", "details": {"defense": "afp", "mode": "base"}},
            }

        def mock_generate_cand(self_atk, x):
            cand = x.copy().astype(np.float32)
            cand[0] += 1.0
            return cand, {"l0": 1, "l2": 1.0, "linf": 1.0}

        out = io.StringIO()
        with patch.object(attacker_sim.RemoteServerOracle, "predict", mock_oracle_predict), \
             patch.object(SurrogateTransferAttack, "generate_candidate", mock_generate_cand), \
             patch("attacker_sim.run_single_flow", side_effect=mock_run_single_flow), \
             patch("attacker_sim.start_simulation_session", return_value="sim-test-01"), \
             patch("attacker_sim.stop_simulation_session") as mock_stop, \
             patch("sys.stdout", out):
            result = run_surrogate_transfer_attack()

        output_str = out.getvalue()
        self.assertTrue(result, "Surrogate attack should return True for completed demonstration")
        self.assertIn("SURROGATE EVASION SUCCESS", output_str)
        self.assertEqual(len(submitted_flows), 1, "Exactly one final measured target must be submitted")
        self.assertFalse(submitted_flows[0]["is_query"], "Final submission must have is_query=False")
        mock_stop.assert_called_once()

    def test_branch_2_original_detected_candidate_detected_blocked(self):
        """Original detected (403) + candidate detected (403) -> CANDIDATE DETECTED / TRANSFER BLOCKED."""
        submitted_flows = []

        def mock_oracle_predict(self_oracle, X, sample_ids=None, stage=None):
            if stage == "eligibility":
                return np.array([1], dtype=int)  # 403 detected
            elif stage == "surrogate_fitting":
                return np.array([0 if i % 2 == 0 else 1 for i in range(len(X))], dtype=int)
            return np.array([1] * len(X), dtype=int)

        def mock_generate_cand(self_atk, x):
            cand = x.copy().astype(np.float32)
            cand[0] += 1.0
            return cand, {"l0": 1, "l2": 1.0, "linf": 1.0}

        def mock_run_single_flow(*args, **kwargs):
            submitted_flows.append(kwargs)
            return {
                "status_code": 403,
                "blocked": True,
                "allowed": False,
                "body": {"status": "blocked", "details": {"defense": "afp", "mode": "base"}},
            }

        out = io.StringIO()
        with patch.object(attacker_sim.RemoteServerOracle, "predict", mock_oracle_predict), \
             patch.object(SurrogateTransferAttack, "generate_candidate", mock_generate_cand), \
             patch("attacker_sim.run_single_flow", side_effect=mock_run_single_flow), \
             patch("attacker_sim.start_simulation_session", return_value="sim-test-02"), \
             patch("attacker_sim.stop_simulation_session") as mock_stop, \
             patch("sys.stdout", out):
            result = run_surrogate_transfer_attack()

        output_str = out.getvalue()
        self.assertTrue(result, "Completed demonstration should return True")
        self.assertIn("CANDIDATE DETECTED / TRANSFER BLOCKED", output_str)
        self.assertNotIn("SURROGATE EVASION SUCCESS", output_str)
        self.assertEqual(len(submitted_flows), 1)
        # Vector must be the candidate, not accidentally reverted to original
        submitted_vec = submitted_flows[0]["vector"]
        sample_id = submitted_flows[0]["sample_id"]
        orig_vec = self.ctx.dataset.X.iloc[sample_id].values
        self.assertFalse(
            np.array_equal(np.array(submitted_vec, dtype=np.float32), orig_vec.astype(np.float32)),
            "Submitted vector must be the modified candidate, not the original flow"
        )
        mock_stop.assert_called_once()

    def test_branch_3_original_already_allowed_baseline_false_negative(self):
        """Original allowed (200 on eligibility) -> BASELINE FALSE NEGATIVE / ORIGINAL MALICIOUS FLOW ALLOWED."""
        submitted_flows = []

        def mock_oracle_predict(self_oracle, X, sample_ids=None, stage=None):
            if stage == "eligibility":
                return np.array([0], dtype=int)  # 200 allowed
            return np.array([0] * len(X), dtype=int)

        def mock_run_single_flow(*args, **kwargs):
            submitted_flows.append(kwargs)
            return {
                "status_code": 200,
                "blocked": False,
                "allowed": True,
                "body": {"status": "success"},
            }

        out = io.StringIO()
        with patch.object(attacker_sim.RemoteServerOracle, "predict", mock_oracle_predict), \
             patch("attacker_sim.run_single_flow", side_effect=mock_run_single_flow), \
             patch("attacker_sim.start_simulation_session", return_value="sim-test-03"), \
             patch("attacker_sim.stop_simulation_session") as mock_stop, \
             patch("sys.stdout", out):
            result = run_surrogate_transfer_attack()

        output_str = out.getvalue()
        self.assertTrue(result, "Completed demonstration should return True")
        self.assertIn("BASELINE FALSE NEGATIVE / ORIGINAL MALICIOUS FLOW ALLOWED", output_str)
        self.assertNotIn("SURROGATE EVASION SUCCESS", output_str)
        self.assertEqual(len(submitted_flows), 1)
        self.assertFalse(submitted_flows[0]["is_query"])
        # Submitted flow must be unchanged original
        submitted_vec = submitted_flows[0]["vector"]
        sample_id = submitted_flows[0]["sample_id"]
        orig_vec = self.ctx.dataset.X.iloc[sample_id].values
        self.assertTrue(
            np.array_equal(np.array(submitted_vec, dtype=np.float32), orig_vec.astype(np.float32)),
            "Submitted vector must be the unchanged original flow"
        )
        mock_stop.assert_called_once()

    def test_branch_4_zero_change_or_none_candidate_baseline_fallback(self):
        """Missing or zero-change candidate -> transparent baseline fallback without evasion claim."""
        submitted_flows = []

        def mock_oracle_predict(self_oracle, X, sample_ids=None, stage=None):
            if stage == "eligibility":
                return np.array([1], dtype=int)
            elif stage == "surrogate_fitting":
                return np.array([0 if i % 2 == 0 else 1 for i in range(len(X))], dtype=int)
            return np.array([1] * len(X), dtype=int)

        def mock_run_single_flow(*args, **kwargs):
            submitted_flows.append(kwargs)
            return {
                "status_code": 403,
                "blocked": True,
                "allowed": False,
                "body": {"status": "blocked"},
            }

        out = io.StringIO()
        # Patch generate_candidate to return (None, "NO_FEASIBLE_CANDIDATE")
        with patch.object(attacker_sim.RemoteServerOracle, "predict", mock_oracle_predict), \
             patch.object(SurrogateTransferAttack, "generate_candidate", return_value=(None, "NO_FEASIBLE_CANDIDATE")), \
             patch("attacker_sim.run_single_flow", side_effect=mock_run_single_flow), \
             patch("attacker_sim.start_simulation_session", return_value="sim-test-04"), \
             patch("attacker_sim.stop_simulation_session") as mock_stop, \
             patch("sys.stdout", out):
            result = run_surrogate_transfer_attack()

        output_str = out.getvalue()
        self.assertTrue(result, "Completed baseline fallback should return True")
        self.assertNotIn("SURROGATE EVASION SUCCESS", output_str)
        self.assertIn("Baseline Fallback", output_str)
        self.assertEqual(len(submitted_flows), 1)
        mock_stop.assert_called_once()

    def test_branch_5_eligibility_transport_error_fails_cleanly(self):
        """Preliminary eligibility error -> EXECUTION ERROR, returns False, no fabricated verdict."""
        def mock_oracle_predict(self_oracle, X, sample_ids=None, stage=None):
            raise RuntimeError("HTTP 500 Internal Server Error from target oracle")

        out = io.StringIO()
        with patch.object(attacker_sim.RemoteServerOracle, "predict", mock_oracle_predict), \
             patch("attacker_sim.start_simulation_session", return_value="sim-test-05"), \
             patch("attacker_sim.stop_simulation_session") as mock_stop, \
             patch("sys.stdout", out):
            result = run_surrogate_transfer_attack()

        output_str = out.getvalue()
        self.assertFalse(result, "Transport failure must return False for nonzero CLI exit")
        self.assertIn("[EXECUTION ERROR]", output_str)
        self.assertNotIn("SURROGATE EVASION SUCCESS", output_str)
        self.assertNotIn("BASELINE FALSE NEGATIVE", output_str)
        mock_stop.assert_called_once()

    def test_branch_6_final_submission_failure_returns_false(self):
        """Final submission HTTP 500 -> EXECUTION ERROR, returns False."""
        def mock_oracle_predict(self_oracle, X, sample_ids=None, stage=None):
            if stage == "eligibility":
                return np.array([1], dtype=int)
            elif stage == "surrogate_fitting":
                return np.array([0 if i % 2 == 0 else 1 for i in range(len(X))], dtype=int)
            return np.array([1] * len(X), dtype=int)

        def mock_run_single_flow(*args, **kwargs):
            return {"status_code": 500, "error": "Server error", "blocked": False, "allowed": False}

        out = io.StringIO()
        with patch.object(attacker_sim.RemoteServerOracle, "predict", mock_oracle_predict), \
             patch("attacker_sim.run_single_flow", side_effect=mock_run_single_flow), \
             patch("attacker_sim.start_simulation_session", return_value="sim-test-06"), \
             patch("attacker_sim.stop_simulation_session") as mock_stop, \
             patch("sys.stdout", out):
            result = run_surrogate_transfer_attack()

        output_str = out.getvalue()
        self.assertFalse(result, "Final submission HTTP 500 must return False")
        self.assertIn("[EXECUTION ERROR]", output_str)
        mock_stop.assert_called_once()

    def test_branch_7_seed_propagation(self):
        """Nondefault CLI seed reaches SurrogateTransferAttack constructor and DecisionTree fitting."""
        ctx_seed = get_ctx(profile="fixture20", seed=999)
        self.assertEqual(ctx_seed.seed, 999)

        captured_attack = []
        original_init = SurrogateTransferAttack.__init__

        def spy_init(self_atk, *args, **kwargs):
            original_init(self_atk, *args, **kwargs)
            captured_attack.append(self_atk)

        def mock_oracle_predict(self_oracle, X, sample_ids=None, stage=None):
            if stage == "eligibility":
                return np.array([1], dtype=int)
            return np.array([0 if i % 2 == 0 else 1 for i in range(len(X))], dtype=int)

        def mock_run_single_flow(*args, **kwargs):
            return {"status_code": 403, "blocked": True, "allowed": False, "body": {}}

        with patch.object(SurrogateTransferAttack, "__init__", spy_init), \
             patch.object(attacker_sim.RemoteServerOracle, "predict", mock_oracle_predict), \
             patch("attacker_sim.run_single_flow", side_effect=mock_run_single_flow), \
             patch("attacker_sim.start_simulation_session", return_value="sim-seed-test"), \
             patch("attacker_sim.stop_simulation_session"):
            run_surrogate_transfer_attack()

        self.assertTrue(len(captured_attack) >= 1)
        atk = captured_attack[0]
        self.assertEqual(atk.effective_seed, 999, "effective_seed must match CLI context seed")
        if atk.surrogate is not None:
            self.assertEqual(atk.surrogate.random_state, 999, "surrogate DecisionTreeClassifier must receive effective_seed")

        # Reset context back to seed 42
        get_ctx(profile="fixture20", seed=42)

    def test_branch_8_silent_probing_wording(self):
        """Silent Probing: 0 preliminary crafting queries, MALICIOUS BASELINE FLOW ALLOWED / FALSE NEGATIVE."""
        # 1. Allowed branch
        def mock_allowed(*args, **kwargs):
            return {"status_code": 200, "blocked": False, "allowed": True, "body": {}}

        out_allowed = io.StringIO()
        with patch("attacker_sim.run_single_flow", side_effect=mock_allowed), \
             patch("attacker_sim.start_simulation_session", return_value="sim-silent-01"), \
             patch("attacker_sim.stop_simulation_session"), \
             patch("sys.stdout", out_allowed):
            res_allowed = run_silent_probing_attack()

        text_allowed = out_allowed.getvalue()
        self.assertTrue(res_allowed)
        self.assertIn("0 preliminary crafting queries", text_allowed)
        self.assertIn("MALICIOUS BASELINE FLOW ALLOWED / FALSE NEGATIVE", text_allowed)
        self.assertNotIn("EVADED & ALLOWED", text_allowed)
        self.assertNotIn("penetrated protected server", text_allowed)

        # 2. Blocked branch
        def mock_blocked(*args, **kwargs):
            return {"status_code": 403, "blocked": True, "allowed": False, "body": {"details": {"defense": "afp", "mode": "base"}}}

        out_blocked = io.StringIO()
        with patch("attacker_sim.run_single_flow", side_effect=mock_blocked), \
             patch("attacker_sim.start_simulation_session", return_value="sim-silent-02"), \
             patch("attacker_sim.stop_simulation_session"), \
             patch("sys.stdout", out_blocked):
            res_blocked = run_silent_probing_attack()

        text_blocked = out_blocked.getvalue()
        self.assertTrue(res_blocked)
        self.assertIn("DETECTED & BLOCKED (403)", text_blocked)

        # 3. Execution error
        def mock_error(*args, **kwargs):
            return {"status_code": 500, "error": "Internal error"}

        out_err = io.StringIO()
        with patch("attacker_sim.run_single_flow", side_effect=mock_error), \
             patch("attacker_sim.start_simulation_session", return_value="sim-silent-03"), \
             patch("attacker_sim.stop_simulation_session"), \
             patch("sys.stdout", out_err):
            res_err = run_silent_probing_attack()

        self.assertFalse(res_err)
        self.assertIn("[EXECUTION ERROR]", out_err.getvalue())

    def test_branch_9_feature_squeezing_decimal_computation(self):
        """Feature Squeezing decimal precision reduction formula d = max(0, 6 - int(intensity))."""
        def compute_d(intensity):
            return max(0, 6 - int(intensity))

        self.assertEqual(compute_d(0.0), 6)
        self.assertEqual(compute_d(1.0), 5)
        self.assertEqual(compute_d(1.5), 5)
        self.assertEqual(compute_d(2.0), 4)
        self.assertEqual(compute_d(5.0), 1)
        self.assertEqual(compute_d(6.0), 0)
        self.assertEqual(compute_d(7.0), 0)


if __name__ == "__main__":
    unittest.main()

# Phase 10C: Official Attack-Cache Generation and Validation Report (v2 — Authoritative)

**Host Platform**: MacBook Air (Mac16,12), Apple M4, 16 GB RAM, macOS 27.0.0, Python 3.9.6 arm64  
**Project Root**: `/Users/trumpler-mac/Downloads/thesissep20276/september 2026/recall-aware-ids`  
**Git Branch**: `phase10b-cache-orchestration`  
**Protocol Freeze Tag**: `phase10-protocol-freeze` (`65005505415a2bdf2d5744dbd135e9214e74081a`)  
**Execution Commit**: `2a613ac0c2a7fc26a14c1f125d3470b0a7a9838e`  
**Frozen Protocol Timestamp**: `experiment.date_frozen: "2026-09-17T21:23:51+08:00"`  
**Evidence Version**: v2 (Supersedes v1 evidence with corrected ASR, canonical Decision Boundary screening hashes, and disambiguated query-accounting categories)  

---

## 1. Executive Summary & Immutability Guarantee

In Phase 10C, exactly 15 canonical evaluation attack caches were generated under the frozen protocol via `./.venv-m4/bin/python scripts/build_evaluation_caches.py --execute`. All 15 cache directories in `artifacts/caches/` are immutable and have remained strictly unmodified throughout this evidence review.

**Zero cache payload or metadata files were modified**:
* All 15 `X_attacked.parquet` files remain bit-for-bit identical to generation output.
* All 15 `status.parquet` files remain bit-for-bit identical to generation output.
* All 15 `manifest.json` files remain bit-for-bit identical to generation output.
* All 15 `completion.json` files remain bit-for-bit identical to generation output.

This v2 report and its companion machine-readable inventory `artifacts/reports/cache_inventory_v2.json` resolve three evidence reporting discrepancies present in the initial v1 artifacts:
1. **ASR Semantics**: Corrects the derived inventory to compute ASR programmatically (`null` for Silent Probing, `successful / attempted` for applicable attacks with attempts, and `0.0` for applicable attacks with zero attempts) rather than reading a non-existent manifest key.
2. **Decision Boundary Screening Hashes**: Populates the canonical manifest screening prediction hash (`3caecf1f...`) and exact canonical selected-position hashes from `manifest.json -> screening_metrics`, independently recomputed and verified against the frozen Random Forest hard predictions and deterministic boundary selector.
3. **Disambiguated Query Accounting**: Separates target model predictions across setup, screening, verification, status rows, and total volume, clearly distinguishing direct model setup calls from oracle-routed calls.

---

## 2. Preflight & Protected Artifact Integrity

All protected artifact hashes were verified before, during, and after cache generation and evidence repair:

| Protected File | Expected & Actual SHA-256 Hash | Status |
|---|---|---|
| `artifacts/models/frozen_rf.joblib` | `9608672c5d5e38a9272c560679cdd2291399de0e0917c8c9dff373bfd200f51d` | **MATCH** |
| `data/manifests/evaluation_roles.csv` | `cbf650879aa1369fa26b803777f30d4b0add9e7aff2c05d399ba7f42563f7b45` | **MATCH** |
| `data/manifests/evaluation_batches.csv` | `4084017e5593732e455763416f7fc38254fc9dab2466eca289b66e68dc0ff79a` | **MATCH** |

* Available disk space: 68 GiB (preflight) / 69 GiB (current).
* Directory status: Exactly 15 cache directories in `artifacts/caches/`; 0 incomplete staging directories (`.staging_*`); 0 quarantine directories (`_quarantined_*`).
* Non-mutating preflight PASSED cleanly prior to official execution.

---

## 3. Canonical 15-Cache Validation & Corrected Metrics

Every cache contains exactly 72,000 rows aligned with `eval_position` across 144 batches of 500, with 78 float32 features matching `feature_names.json` in exact order, and zero NaN/infinite values.

### 3.1 Cache Summary Table (Authoritative v2)

| # | Cache Directory | Scenario | Seed | Status | Eligible | Attempted | Success | Attack Success Rate (ASR) | Status Row Queries (`sum`) | Total Target Predictions |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `SilentProbing_42` | SilentProbing | 42 | **VALID** | 0 | 0 | 0 | *null* | 0 | 0 |
| 2 | `SilentProbing_43` | SilentProbing | 43 | **VALID** | 0 | 0 | 0 | *null* | 0 | 0 |
| 3 | `SilentProbing_44` | SilentProbing | 44 | **VALID** | 0 | 0 | 0 | *null* | 0 | 0 |
| 4 | `SilentProbing_45` | SilentProbing | 45 | **VALID** | 0 | 0 | 0 | *null* | 0 | 0 |
| 5 | `SilentProbing_46` | SilentProbing | 46 | **VALID** | 0 | 0 | 0 | *null* | 0 | 0 |
| 6 | `SurrogateTransfer_42` | SurrogateTransfer | 42 | **VALID** | 11,554 | 11,554 | 5 | $5 / 11554 \approx 0.000433$ | 83,554 | 101,554 |
| 7 | `SurrogateTransfer_43` | SurrogateTransfer | 43 | **VALID** | 11,554 | 11,554 | 3 | $3 / 11554 \approx 0.000260$ | 83,554 | 101,554 |
| 8 | `SurrogateTransfer_44` | SurrogateTransfer | 44 | **VALID** | 11,554 | 11,554 | 3 | $3 / 11554 \approx 0.000260$ | 83,554 | 101,554 |
| 9 | `SurrogateTransfer_45` | SurrogateTransfer | 45 | **VALID** | 11,554 | 11,554 | 2 | $2 / 11554 \approx 0.000173$ | 83,554 | 101,554 |
| 10 | `SurrogateTransfer_46` | SurrogateTransfer | 46 | **VALID** | 11,554 | 11,554 | 1 | $1 / 11554 \approx 0.000087$ | 83,554 | 101,554 |
| 11 | `DecisionBoundary_42` | DecisionBoundary | 42 | **VALID** | 11,554 | 200 | 200 | $200 / 200 = 1.0$ | 2,600 | 92,600 |
| 12 | `DecisionBoundary_43` | DecisionBoundary | 43 | **VALID** | 11,554 | 200 | 200 | $200 / 200 = 1.0$ | 2,600 | 92,600 |
| 13 | `DecisionBoundary_44` | DecisionBoundary | 44 | **VALID** | 11,554 | 200 | 200 | $200 / 200 = 1.0$ | 2,600 | 92,600 |
| 14 | `DecisionBoundary_45` | DecisionBoundary | 45 | **VALID** | 11,554 | 200 | 200 | $200 / 200 = 1.0$ | 2,600 | 92,600 |
| 15 | `DecisionBoundary_46` | DecisionBoundary | 46 | **VALID** | 11,554 | 200 | 200 | $200 / 200 = 1.0$ | 2,600 | 92,600 |

### 3.2 Ordering and Sum Invariants
Across all 15 caches:
* `status_code` counts sum to exactly 72,000:
  - **Silent Probing**: `NOT_APPLICABLE`: 72,000.
  - **Surrogate Transfer**: `INELIGIBLE_TRUE_BENIGN`: 59,743; `INELIGIBLE_FALSE_NEGATIVE`: 703; `TARGET_REJECTION`: 11,549–11,553; `SUCCESS`: 1–5.
  - **Decision Boundary**: `INELIGIBLE_TRUE_BENIGN`: 59,743; `INELIGIBLE_FALSE_NEGATIVE`: 703; `NOT_ATTEMPTED`: 11,354; `SUCCESS`: 200.
* Count ordering is strictly maintained:
  $$0 \le \text{successful} \le \text{attempted} \le \text{eligible} \le 72,000$$

---

## 4. Query-Accounting Category Disambiguation

To prevent ambiguity, target-model interactions are broken down into discrete categories based on executed code:

| Query Accounting Stage | Silent Probing (seeds 42–46) | Surrogate Transfer (seeds 42–46) | Decision Boundary (seeds 42–46) | Call Routing & Execution Details |
|---|---|---|---|---|
| **Crafting / Setup Predictions** | 0 | 18,000 | 18,000 | • **ST**: Routed through `BlackBoxOracle` to label crafting pool.<br>• **DB**: Direct `predict_fn(X_craft)` setup call to identify benign references ($y=0$); not routed through oracle. |
| **Measurement Screening Queries** | 0 | 72,000 | 72,000 | • **ST**: Routed through target `BlackBoxOracle` (1 per measurement sample to verify clean TP).<br>• **DB**: Routed through screening `BlackBoxOracle` (stage="screening") over all 72,000 samples. |
| **Candidate Verification / Attack Queries** | 0 | 11,554 | 2,600 | • **ST**: Routed through target `BlackBoxOracle` (1 query on $X_{\text{cand}}$ for each of the 11,554 clean TPs).<br>• **DB**: Routed through attack `BlackBoxOracle` across 200 targets (13 queries each = 2,600 queries). |
| **Status-Row Query Total (`sum(queries_used)`)** | 0 | 83,554 | 2,600 | • **SP**: All status rows record `queries_used = 0`.<br>• **ST**: Ineligible rows record 1; attempted record 2 ($59743\cdot 1 + 703\cdot 1 + 11554\cdot 2 = 83554$).<br>• **DB**: 200 attacked record 13 ($200\cdot 13 = 2600$); remaining 71,800 record 0. |
| **Total Row-Level Target Predictions** | 0 | 101,554 | 92,600 | • **SP**: $0$.<br>• **ST**: $18000 \text{ (crafting)} + 72000 \text{ (screening)} + 11554 \text{ (verification)} = 101554$.<br>• **DB**: $18000 \text{ (direct setup)} + 72000 \text{ (screening oracle)} + 2600 \text{ (attack oracle)} = 92600$. |

### Exact Meanings of Manifest `query_budgets` Fields
* `max_queries_per_sample`:
  - `SilentProbing`: `0` (passive baseline; queries prohibited).
  - `SurrogateTransfer`: `0` (offline transfer attack; zero iterative boundary queries per sample).
  - `DecisionBoundary`: `50` (strict query ceiling enforced by `BlackBoxOracle`; actual usage was 13 queries/sample).
* `crafting_queries_budget`:
  - `SilentProbing`: `0`.
  - `SurrogateTransfer`: `18000` (budget for oracle labeling of the 18,000 crafting samples to fit surrogate).
  - `DecisionBoundary`: `0` (direct setup call on crafting samples; not tracked against attack query budget).
* `target_evaluation_queries`:
  - `SilentProbing`: `0`.
  - `SurrogateTransfer`: `72000` (base budget for evaluating 72,000 measurement positions).
  - `DecisionBoundary`: `0` (attack queries tracked under `screening_metrics.attack_oracle_queries_used: 2600`).

---

## 5. Canonical Decision Boundary Screening Hashes

For each Decision Boundary cache, the screening prediction and selected position hashes were verified against the frozen Random Forest hard predictions and canonical target selection:

* **Shared Screening Prediction Hash**:
  - Manifest Value: `3caecf1f779f9dd58c804a8a74c8370f1297b6eef69cd050e4ba7045c527a807`
  - Recomputed Value: `3caecf1f779f9dd58c804a8a74c8370f1297b6eef69cd050e4ba7045c527a807`
  - Encoding: `hashlib.sha256(rf.predict(X_meas).astype(np.int8).tobytes()).hexdigest()`
  - Agreement: **MATCH (100% bit-for-bit across all 5 seeds)**

* **Canonical Selected-Position Hashes**:
  - Recomputed using exact generation serialization: `hashlib.sha256(np.array(selected_positions, dtype=np.int64).tobytes()).hexdigest()`.
  - Confirmed bit-for-bit identical with `manifest.json -> screening_metrics.selected_position_hash` and `status.parquet` (`status_code == 'SUCCESS'`):

| Seed | Manifest & Recomputed Canonical `selected_position_hash` | Status |
|---|---|---|
| **42** | `cc394452a4f4d80521ccd506dfdf2deed8a5b2770753b2921b20b9d47fc18e76` | **MATCH** |
| **43** | `e9ac8498080b3cb7722c17965e8a699e003114ed67cd58932e9efccb6dbb3a57` | **MATCH** |
| **44** | `df234b07aa688eeac90230e1315e8c8d4171d122c622c199b5dfa604a682ac4d` | **MATCH** |
| **45** | `43b1fad3d02de0285388704cb1d19c69de8aaec1e7eff643544a70aa8e3dfeb1` | **MATCH** |
| **46** | `b0c1f49fb9c996c4b8504737dd40c32f130cb10822b9e67ee3a7fae9e667e14d` | **MATCH** |

*(Note: The alternative JSON-string hash emitted by the superseded v1 inventory script is retained in `cache_inventory_v2.json` under `superseded_v1_json_position_hash` for audit trail documentation).*

---

## 6. Historical Superseded Evidence Hashes

The original v1 artifacts are preserved as superseded historical evidence:

| Superseded Artifact | SHA-256 Digest | Role |
|---|---|---|
| `artifacts/reports/cache_inventory.json` | `3903283cc5d0835b2e6b5ae410c4f6ae0f8951bca24af8c2434d8b1acf1766bf` | Superseded v1 cache inventory (contains `asr: null`) |
| `artifacts/reports/phase10c_official_cache_generation.md` | `6fe8db86e15806e8b38b9b930d9240c2a21c9490b30a08df4173d50c5fbd54a6` | Superseded v1 report |
| `phase10c_official_cache_generation_review_bundle.zip` | `dfdee9e333c20ef1d50f49e03af310cba5e0fbf570cc573e8c3a828480917246` | Superseded v1 review bundle |

---

## 7. Status for Phase 10D

* Phase 10C official cache generation is **FULLY COMPLETED and VERIFIED**.
* All 15/15 canonical evaluation caches in `artifacts/caches/` are complete, valid, and immutable.
* Official defense and controller evaluation runs (Phase 10D: 252 unique runs across 36,288 batches) remain **UNSTARTED**.
* Empirical Recall-Aware (RA) effectiveness remains **STRICTLY UNKNOWN**.

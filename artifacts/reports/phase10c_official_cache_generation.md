# Phase 10C: Official Attack-Cache Generation and Validation Report

## Executive Summary
Phase 10C successfully executed official evaluation attack-cache generation for the Recall-Aware IDS thesis evaluation protocol on Apple Silicon (`MacBook Air Mac16,12`, Apple M4, 16 GB RAM, macOS 27.0.0, Python 3.9.6 arm64). Official execution was authorized via the reviewed entry point (`./.venv-m4/bin/python scripts/build_evaluation_caches.py --execute`) at commit `2a613ac0c2a7fc26a14c1f125d3470b0a7a9838e` on branch `phase10b-cache-orchestration`.

All 15 canonical evaluation caches were generated sequentially, validated atomically in staging directories beneath the cache root, and published without error. Following generation, all 15 caches underwent comprehensive independent post-execution validation using `validate_completed_cache()` and `ConcreteAttackCacheProvider`.

**Key Milestones & Guarantees:**
1. **15 / 15 Caches Built and Validated**: Exactly 15 canonical caches published into `artifacts/caches`.
2. **Atomic Publication & Zero Leftovers**: Zero incomplete staging directories (`.staging_*`), zero quarantine directories (`_quarantined_*`).
3. **Exact Row Count & Alignment**: Exactly 72,000 measurement rows per cache, strictly aligned with `evaluation_batches.csv` positions (`eval_position`).
4. **Exact Feature Schema**: Exactly 78 floating-point feature columns in exact frozen order matching `feature_names.json`, with zero NaNs and zero infinite values.
5. **Exact Hash Consistency**: Manifest, completion, attacked-feature, and status SHA-256 hashes agree across all 15 caches.
6. **Protected Artifacts Untouched**: The frozen RF model, evaluation roles manifest, and evaluation batches manifest hashes are 100% identical before and after execution.
7. **Protocol Boundaries Maintained**: Official defense/controller evaluation experiments remain **unstarted**. Empirical Recall-Aware (RA) effectiveness remains **strictly unknown**.

---

## 1. System & Execution Preflight State

### 1.1 Git Commitment & Ancestry
- **Current Branch**: `phase10b-cache-orchestration`
- **Execution HEAD Commit**: `2a613ac0c2a7fc26a14c1f125d3470b0a7a9838e`
- **Protocol Freeze Tag**: `phase10-protocol-freeze` (`65005505415a2bdf2d5744dbd135e9214e74081a`)
- **Git Ancestry**: Verified (`phase10-protocol-freeze` is a direct ancestor of execution commit `2a613ac0c2a7fc26a14c1f125d3470b0a7a9838e`).
- **Working Tree Cleanliness**: Tracked working tree was 100% clean prior to execution.
- **Protocol Freeze Timestamp**: `experiment.date_frozen = "2026-09-17T21:23:51+08:00"`.

### 1.2 Resource Metrics
- **Available Disk Space at Start**: 68 GiB
- **Total Cache Disk Usage**: 122 MB across all 15 caches (~8.1–8.2 MB per cache)
- **Total Execution Runtime**: ~1 hour 48 minutes across all 15 caches

### 1.3 Authoritative Protected Hashes Verification
All three protected artifacts were verified before and after cache generation:

| Artifact | Relative Path | Authoritative Frozen SHA-256 | Post-Execution SHA-256 | Status |
|---|---|---|---|---|
| **Frozen RF Model** | `artifacts/models/frozen_rf.joblib` | `9608672c5d5e38a9272c560679cdd2291399de0e0917c8c9dff373bfd200f51d` | `9608672c5d5e38a9272c560679cdd2291399de0e0917c8c9dff373bfd200f51d` | **MATCH** |
| **Evaluation Roles** | `data/manifests/evaluation_roles.csv` | `cbf650879aa1369fa26b803777f30d4b0add9e7aff2c05d399ba7f42563f7b45` | `cbf650879aa1369fa26b803777f30d4b0add9e7aff2c05d399ba7f42563f7b45` | **MATCH** |
| **Evaluation Batches** | `data/manifests/evaluation_batches.csv` | `4084017e5593732e455763416f7fc38254fc9dab2466eca289b66e68dc0ff79a` | `4084017e5593732e455763416f7fc38254fc9dab2466eca289b66e68dc0ff79a` | **MATCH** |

---

## 2. Official Cache Inventory & Generation Summary

| # | Scenario | Seed | Canonical Identifier | Rows | Eligible | Attempted | Success | Queries | ASR | Validation |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `SilentProbing` | 42 | `SilentProbing_42` | 72,000 | 0 | 0 | 0 | 0 | `null` | **VALID** |
| 2 | `SilentProbing` | 43 | `SilentProbing_43` | 72,000 | 0 | 0 | 0 | 0 | `null` | **VALID** |
| 3 | `SilentProbing` | 44 | `SilentProbing_44` | 72,000 | 0 | 0 | 0 | 0 | `null` | **VALID** |
| 4 | `SilentProbing` | 45 | `SilentProbing_45` | 72,000 | 0 | 0 | 0 | 0 | `null` | **VALID** |
| 5 | `SilentProbing` | 46 | `SilentProbing_46` | 72,000 | 0 | 0 | 0 | 0 | `null` | **VALID** |
| 6 | `SurrogateTransfer` | 42 | `SurrogateTransfer_42` | 72,000 | 11,554 | 11,554 | 5 | 83,554 | 0.000433 | **VALID** |
| 7 | `SurrogateTransfer` | 43 | `SurrogateTransfer_43` | 72,000 | 11,554 | 11,554 | 3 | 83,554 | 0.000260 | **VALID** |
| 8 | `SurrogateTransfer` | 44 | `SurrogateTransfer_44` | 72,000 | 11,554 | 11,554 | 3 | 83,554 | 0.000260 | **VALID** |
| 9 | `SurrogateTransfer` | 45 | `SurrogateTransfer_45` | 72,000 | 11,554 | 11,554 | 2 | 83,554 | 0.000173 | **VALID** |
| 10 | `SurrogateTransfer` | 46 | `SurrogateTransfer_46` | 72,000 | 11,554 | 11,554 | 1 | 83,554 | 0.000087 | **VALID** |
| 11 | `DecisionBoundary` | 42 | `DecisionBoundary_42` | 72,000 | 11,554 | 200 | 200 | 2,600 | 1.000000 | **VALID** |
| 12 | `DecisionBoundary` | 43 | `DecisionBoundary_43` | 72,000 | 11,554 | 200 | 200 | 2,600 | 1.000000 | **VALID** |
| 13 | `DecisionBoundary` | 44 | `DecisionBoundary_44` | 72,000 | 11,554 | 200 | 200 | 2,600 | 1.000000 | **VALID** |
| 14 | `DecisionBoundary` | 45 | `DecisionBoundary_45` | 72,000 | 11,554 | 200 | 200 | 2,600 | 1.000000 | **VALID** |
| 15 | `DecisionBoundary` | 46 | `DecisionBoundary_46` | 72,000 | 11,554 | 200 | 200 | 2,600 | 1.000000 | **VALID** |

---

## 3. Scenario-Specific Verification

### 3.1 Silent Probing (`SilentProbing`, Seeds 42–46)
- **Identity Transformation**: `X_attacked.parquet` was verified byte-for-byte against the original unperturbed evaluation measurement features (`X_meas_orig`). Zero feature values were altered.
- **Zero Attack Activity**:
  - Eligible attacks: 0
  - Attempted attacks: 0
  - Successful attacks: 0
  - Oracle queries: 0
- **Status Code Distribution**: Exactly 72,000 rows with `status_code == "NOT_APPLICABLE"` across all 5 seeds.
- **ASR Semantics**: Explicitly serialized as `null` (`None` in Python), complying strictly with the protocol rule that ASR is not applicable to silent reconnaissance.

### 3.2 Surrogate Transfer (`SurrogateTransfer`, Seeds 42–46)
- **Crafting Partition Isolation**: Surrogate Decision Tree was fitted strictly on the 18,000 crafting records with labels determined exclusively by target model oracle hard predictions (`predict_fn(X_craft)`).
- **Measurement Pool Coverage**: Full 72,000 measurement rows processed per seed.
- **Clean True Positive Eligibility**: Across all seeds, exactly 11,554 measurement samples were identified as clean true positives (`y_meas == 1 & orig_pred == 1`).
- **Transfer Evaluation Queries**: Each candidate submitted to the black-box oracle used exactly 1 query for ineligible screening and 1 additional verification query for eligible attempts (total: 83,554 queries per seed).
- **Status Code Breakdown**:
  - `INELIGIBLE_TRUE_BENIGN`: 59,743
  - `INELIGIBLE_FALSE_NEGATIVE`: 703
  - `TARGET_REJECTION`: 11,549 (seed 42), 11,551 (seeds 43, 44), 11,552 (seed 45), 11,553 (seed 46)
  - `SUCCESS`: 5 (seed 42), 3 (seeds 43, 44), 2 (seed 45), 1 (seed 46)

### 3.3 Decision Boundary (`DecisionBoundary`, Seeds 42–46)
- **Global Target Selection**: From the 11,554 eligible clean TPs, exactly **200 targets** were globally selected per seed using the deterministic seed-governed selection rule.
- **Target Attempt Count**: Exactly 200 evasion attempts executed per seed.
- **Target Query Ceiling**: Each of the 200 attacked targets executed exactly 13 queries (1 eligibility, 1 initial endpoint, 10 binary search steps, 1 verification query), strictly complying with the 50-query budget ceiling.
- **Target Success**: All 200 attempted targets achieved successful boundary crossing in the unconstrained baseline condition (ASR = 1.000000 on attempted targets).
- **Distinct Target Selection**: Selected positions hash verified to be distinct across all seeds:
  - Seed 42: `5877af1e602534b7016835ee94600443c87552c3e2f17a5039b9569428a58191`
  - Seed 43: `ce0d247acc05ce3e9109b0800edeed3a4aa026b48a5969507727179be1bc01f1`
  - Seed 44: `397f35406f9fba1872c933ed931124dc8bf21258cc8f2c4df663c98dbce6352b`
  - Seed 45: `ce29e5c73e3663039b3c4cf9cf2998bec1a846c628af3010bd8f9a1403b98610`
  - Seed 46: `d5e1c136bfd67a3e7ebe7fc89fb1a5ec77f6584106029ef494f0f275ffc38704`
- **Status Code Breakdown**:
  - `INELIGIBLE_TRUE_BENIGN`: 59,743
  - `NOT_ATTEMPTED`: 11,354
  - `INELIGIBLE_FALSE_NEGATIVE`: 703
  - `SUCCESS`: 200

---

## 4. Independent Post-Execution Validation Outcome

Following generation, the independent validator script verified:
1. `validate_completed_cache()` passed on all 15 cache directories using independently derived expected cache identities.
2. `ConcreteAttackCacheProvider` initialized with `official_mode=True, expected_row_count=72000` and passed all structural, schema, and alignment checks for all 15 directories.
3. Attacked-feature Parquet digests (`X_attacked_sha256`) and status Parquet digests (`status_sha256`) match between on-disk files, `manifest.json`, and `completion.json`.
4. Machine-readable inventory JSON saved at `artifacts/reports/cache_inventory.json` (17,621 bytes).

---

## 5. Protocol Boundaries and Next Steps

- **Official Cache Generation**: **COMPLETE** (15 / 15 caches ready).
- **Official Defense & Controller Experiments**: **UNSTARTED**.
- **Empirical Recall-Aware Effectiveness**: **STRICTLY UNKNOWN**.
- The pipeline is fully prepared for authorized execution of the 252-run evaluation matrix in the subsequent phase.

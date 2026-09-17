# Phase 10C: Evidence Correction & Immutability Comparison Report

**Host Platform**: MacBook Air (Mac16,12), Apple M4, 16 GB RAM, macOS 27.0.0, Python 3.9.6 arm64  
**Project Root**: `/Users/trumpler-mac/Downloads/thesissep20276/september 2026/recall-aware-ids`  
**Git Branch**: `phase10b-cache-orchestration`  
**Base Protocol Freeze**: `phase10-protocol-freeze` (`65005505415a2bdf2d5744dbd135e9214e74081a`)  
**Execution Commit**: `2a613ac0c2a7fc26a14c1f125d3470b0a7a9838e`  

---

## 1. Proof of Absolute Cache Immutability

Throughout this evidence repair, all 15 official cache directories remained strictly read-only and immutable. Zero bytes of attack features, attack status records, manifest metadata, or completion markers were modified, regenerated, or reordered.

Below is the byte-for-byte SHA-256 comparison across all 30 cache metadata files (`manifest.json` and `completion.json`) recorded before this repair and re-verified after the repair:

| Cache Directory | Metadata File | Pre-Repair SHA-256 | Post-Repair SHA-256 | Status |
|---|---|---|---|---|
| `DecisionBoundary_42` | `manifest.json` | `4e443a3c39663bd745598935654105def62187a02f5c6ed2388a3dfa1966d7de` | `4e443a3c39663bd745598935654105def62187a02f5c6ed2388a3dfa1966d7de` | **UNCHANGED** |
| `DecisionBoundary_42` | `completion.json` | `239cf0c996b13f2f04220bc29b36901ffe92d5f98195ed37d18688d9578dbe98` | `239cf0c996b13f2f04220bc29b36901ffe92d5f98195ed37d18688d9578dbe98` | **UNCHANGED** |
| `DecisionBoundary_43` | `manifest.json` | `6d2152e01e00e326d09d76c192e3cd520c5a8701be62bd60371ce64e8b73a40f` | `6d2152e01e00e326d09d76c192e3cd520c5a8701be62bd60371ce64e8b73a40f` | **UNCHANGED** |
| `DecisionBoundary_43` | `completion.json` | `6e8db2a27fad404ddd41305a022389428a3fe105e3dcbdfc7b25f61bdb9c4b1b` | `6e8db2a27fad404ddd41305a022389428a3fe105e3dcbdfc7b25f61bdb9c4b1b` | **UNCHANGED** |
| `DecisionBoundary_44` | `manifest.json` | `c956218fe069fb4eaf469dbf0924b74a6c7ad6cdb06517adf201d57aa85c4437` | `c956218fe069fb4eaf469dbf0924b74a6c7ad6cdb06517adf201d57aa85c4437` | **UNCHANGED** |
| `DecisionBoundary_44` | `completion.json` | `440005db4ba4931c743e78a7eb64b364fe397cce997f18f5135fd43aa114c915` | `440005db4ba4931c743e78a7eb64b364fe397cce997f18f5135fd43aa114c915` | **UNCHANGED** |
| `DecisionBoundary_45` | `manifest.json` | `8b1ece8746841db8bc2dff6da2b56c25446fbcfbf8207b05095bcde7e93e9b5a` | `8b1ece8746841db8bc2dff6da2b56c25446fbcfbf8207b05095bcde7e93e9b5a` | **UNCHANGED** |
| `DecisionBoundary_45` | `completion.json` | `11fb6c1a4549cac82d6a5bb76f576c6671689e270ee507af9de39420470eef82` | `11fb6c1a4549cac82d6a5bb76f576c6671689e270ee507af9de39420470eef82` | **UNCHANGED** |
| `DecisionBoundary_46` | `manifest.json` | `34f5db141616bcf5bd50c05db0e0285e78e4499647e0b4f66dab6d99e3c738c6` | `34f5db141616bcf5bd50c05db0e0285e78e4499647e0b4f66dab6d99e3c738c6` | **UNCHANGED** |
| `DecisionBoundary_46` | `completion.json` | `6aac5561532892cf5de4b04c418997513b0eab48f046261cc61fb4166cb7e931` | `6aac5561532892cf5de4b04c418997513b0eab48f046261cc61fb4166cb7e931` | **UNCHANGED** |
| `SilentProbing_42` | `manifest.json` | `3142249d6690d219018f2978f3794fcb9f81f1de12cc5e7ba63c81482ba3d588` | `3142249d6690d219018f2978f3794fcb9f81f1de12cc5e7ba63c81482ba3d588` | **UNCHANGED** |
| `SilentProbing_42` | `completion.json` | `9404185caf4be948a7e58b2c87dca88a4444095fad83c8c2a3720a3cd7c2d4eb` | `9404185caf4be948a7e58b2c87dca88a4444095fad83c8c2a3720a3cd7c2d4eb` | **UNCHANGED** |
| `SilentProbing_43` | `manifest.json` | `3a583c3087b88cf520801a300102246942ee9735c97c9fdbb9dc75e90d4573c6` | `3a583c3087b88cf520801a300102246942ee9735c97c9fdbb9dc75e90d4573c6` | **UNCHANGED** |
| `SilentProbing_43` | `completion.json` | `da2bd8d5a00c29b3a1ca6f2221b9aaa9ed6c61a76464b99b436a34eacaa7a60c` | `da2bd8d5a00c29b3a1ca6f2221b9aaa9ed6c61a76464b99b436a34eacaa7a60c` | **UNCHANGED** |
| `SilentProbing_44` | `manifest.json` | `9f7f18354921a71d6a6a28a24e3af03a9b62b894fc84aac010ba25d6872680e0` | `9f7f18354921a71d6a6a28a24e3af03a9b62b894fc84aac010ba25d6872680e0` | **UNCHANGED** |
| `SilentProbing_44` | `completion.json` | `fecbea46382033d1aa598116dde76fbdc3196463b959ed018350bada2995bf76` | `fecbea46382033d1aa598116dde76fbdc3196463b959ed018350bada2995bf76` | **UNCHANGED** |
| `SilentProbing_45` | `manifest.json` | `7aae84fdbd06e90c7e30011d7e88c358eee8f5f28028c83d446b61cd7b00b6da` | `7aae84fdbd06e90c7e30011d7e88c358eee8f5f28028c83d446b61cd7b00b6da` | **UNCHANGED** |
| `SilentProbing_45` | `completion.json` | `d2a01c86f82a19bed4daa596f98d04e2888f6029acbe040d7e14cdb1f1d41055` | `d2a01c86f82a19bed4daa596f98d04e2888f6029acbe040d7e14cdb1f1d41055` | **UNCHANGED** |
| `SilentProbing_46` | `manifest.json` | `37a2a65bd06dc36f17d90f01708e3d63036716870345da5d4fa3e8faae12eb0f` | `37a2a65bd06dc36f17d90f01708e3d63036716870345da5d4fa3e8faae12eb0f` | **UNCHANGED** |
| `SilentProbing_46` | `completion.json` | `b16216d120e2c53de962af4189ab81e9750b04d194deb210fee02ac7d6009b40` | `b16216d120e2c53de962af4189ab81e9750b04d194deb210fee02ac7d6009b40` | **UNCHANGED** |
| `SurrogateTransfer_42` | `manifest.json` | `cf53762f4663244d13c9cc8d9c6f783c8f2a1277ec7409e3c3ed96789390e05c` | `cf53762f4663244d13c9cc8d9c6f783c8f2a1277ec7409e3c3ed96789390e05c` | **UNCHANGED** |
| `SurrogateTransfer_42` | `completion.json` | `1ccc76faeb112e045e7a9851cc2da429413b1693ad8a179ba69edf685955d9f5` | `1ccc76faeb112e045e7a9851cc2da429413b1693ad8a179ba69edf685955d9f5` | **UNCHANGED** |
| `SurrogateTransfer_43` | `manifest.json` | `12a9e7f1e38f22ccd7b5ae8a35ba5db5a5ac6d717fff246de91c3bdd48d49b4f` | `12a9e7f1e38f22ccd7b5ae8a35ba5db5a5ac6d717fff246de91c3bdd48d49b4f` | **UNCHANGED** |
| `SurrogateTransfer_43` | `completion.json` | `f7ef6a38314ce1edf537dd5c05b457c79d087768825d17d9dd23e112b9ad6fc8` | `f7ef6a38314ce1edf537dd5c05b457c79d087768825d17d9dd23e112b9ad6fc8` | **UNCHANGED** |
| `SurrogateTransfer_44` | `manifest.json` | `a4484c8fdecd06ce6b2340123e95608c4ea0cc75c6363d0df4d30e3967cebbdc` | `a4484c8fdecd06ce6b2340123e95608c4ea0cc75c6363d0df4d30e3967cebbdc` | **UNCHANGED** |
| `SurrogateTransfer_44` | `completion.json` | `be5c1fdf6e420e3d6af6a2b5bcfacedca818f2c422ca79d6d9b670cd47d94170` | `be5c1fdf6e420e3d6af6a2b5bcfacedca818f2c422ca79d6d9b670cd47d94170` | **UNCHANGED** |
| `SurrogateTransfer_45` | `manifest.json` | `18d2f0d1215968e3daad74f7e4d90b3ff020db2305e6b5fab44b34f27b0ddb0f` | `18d2f0d1215968e3daad74f7e4d90b3ff020db2305e6b5fab44b34f27b0ddb0f` | **UNCHANGED** |
| `SurrogateTransfer_45` | `completion.json` | `3d6c2d70133379ef1cb6ee1f710c88fd9baaab91a38438fec0a60d903347c33e` | `3d6c2d70133379ef1cb6ee1f710c88fd9baaab91a38438fec0a60d903347c33e` | **UNCHANGED** |
| `SurrogateTransfer_46` | `manifest.json` | `fba14aeffc42e7dc5040e56a4bcf671a9d372a6347931d3a5850ba862d07be9f` | `fba14aeffc42e7dc5040e56a4bcf671a9d372a6347931d3a5850ba862d07be9f` | **UNCHANGED** |
| `SurrogateTransfer_46` | `completion.json` | `2ced08cf6162228c38cb7148484ef5131f08892ce8aa06c8fb33c113628e6501` | `2ced08cf6162228c38cb7148484ef5131f08892ce8aa06c8fb33c113628e6501` | **UNCHANGED** |

Likewise, all 15 `X_attacked.parquet` and all 15 `status.parquet` files remain bit-for-bit identical to their generation hashes.

---

## 2. Field-by-Field Correction Comparison

The table below contrasts the fields in superseded `artifacts/reports/cache_inventory.json` (v1) with authoritative `artifacts/reports/cache_inventory_v2.json` (v2):

| Topic / Field | Superseded Evidence (`cache_inventory.json`, v1) | Authoritative Evidence (`cache_inventory_v2.json`, v2) | Technical Rationale & Exact Semantics |
|---|---|---|---|
| **Inventory Schema Version** | Not present | `"schema_version": "2.0"` | Explicitly declares the authoritative v2 schema. |
| **Attack Success Rate (`asr`) — Silent Probing** | `null` | `null` | Unchanged; Silent Probing is passive and non-attacking; ASR is not applicable. |
| **Attack Success Rate (`asr`) — Surrogate Transfer** | `null` (all 5 seeds) | • Seed 42: $5/11554 \approx 0.000433$<br>• Seed 43: $3/11554 \approx 0.000260$<br>• Seed 44: $3/11554 \approx 0.000260$<br>• Seed 45: $2/11554 \approx 0.000173$<br>• Seed 46: $1/11554 \approx 0.000087$ | `manifest.json` does not store an `"asr"` key; v1 derived script incorrectly queried `manifest_data.get("asr")` returning `None`. v2 programmatically computes $\text{ASR} = \text{successful} / \text{attempted}$. |
| **Attack Success Rate (`asr`) — Decision Boundary** | `null` (all 5 seeds) | `1.0` ($200 / 200$, all 5 seeds) | v2 programmatically computes $\text{ASR} = 200 / 200 = 1.0$. |
| **DB Screening Prediction Hash** | `"screening_prediction_hash": null` | `"manifest_screening_prediction_hash": "3caecf1f...",`<br>`"recomputed_screening_prediction_hash": "3caecf1f..."` | In v1, the script checked `manifest_data.get("screening_prediction_hash")`, omitting the nested `screening_metrics` object. v2 reads the manifest value and independently verifies it against frozen RF hard predictions over measurement rows. |
| **DB Selected Position Hash** | `"selected_position_hash": "5877af1e..."` (derived from `json.dumps(sorted(positions))`) | `"manifest_selected_position_hash": "cc394452..." (seed 42),`<br>`"recomputed_selected_position_hash": "cc394452..." (seed 42)` | v1 computed an ad-hoc JSON serialization hash. v2 populates the canonical manifest hash (`manifest.json -> screening_metrics.selected_position_hash`) and independently verifies it using the exact generation serialization (`int64.tobytes()`). |
| **Legacy JSON Position Hash** | Labeled as canonical `"selected_position_hash"` | Retained under distinct field `"superseded_v1_json_position_hash"` with documented encoding note | Never mislabeled as the manifest hash; preserved solely for cross-inventory provenance. |
| **Query Accounting** | Single ambiguous field `"total_queries": 83554` (ST), `2600` (DB), `0` (SP) | Structured object `query_accounting`: <br>• `crafting_setup_predictions`<br>• `measurement_screening_queries`<br>• `candidate_verification_attack_queries`<br>• `status_row_queries_total`<br>• `total_target_model_predictions`<br>• Call types (`direct_predict_fn` vs `BlackBoxOracle`) | Disambiguates row-level target model predictions from status-row query sums. Clearly documents that DB crafting setup (18,000 predictions) was a direct `predict_fn` call, not an oracle-routed query. |
| **Manifest Query Budget Fields & Explanations** | Not present | Nested `manifest_query_budgets` with explicit `manifest_query_budget_explanations` | Explains exact meanings of `max_queries_per_sample`, `crafting_queries_budget`, and `target_evaluation_queries` according to executed source. |

---

## 3. Superseded vs Authoritative Artifact Hashes

| Artifact | Version | SHA-256 Digest | Status |
|---|---|---|---|
| `artifacts/reports/cache_inventory.json` | v1 | `3903283cc5d0835b2e6b5ae410c4f6ae0f8951bca24af8c2434d8b1acf1766bf` | Superseded |
| `artifacts/reports/cache_inventory_v2.json` | **v2** | *(see current filesystem / review bundle ledger)* | **Authoritative** |
| `artifacts/reports/phase10c_official_cache_generation.md` | v1 | `6fe8db86e15806e8b38b9b930d9240c2a21c9490b30a08df4173d50c5fbd54a6` | Superseded |
| `artifacts/reports/phase10c_official_cache_generation_v2.md` | **v2** | *(see current filesystem / review bundle ledger)* | **Authoritative** |
| `phase10c_official_cache_generation_review_bundle.zip` | v1 | `dfdee9e333c20ef1d50f49e03af310cba5e0fbf570cc573e8c3a828480917246` | Superseded |
| `phase10c_official_cache_generation_review_bundle_v2.zip` | **v2** | *(see final verification section)* | **Authoritative** |

---

## 4. Integrity and Readiness Conclusion

* Every canonical cache in `artifacts/caches/` has been proven immutable.
* All 10 focused tests in `tests/test_phase10c_inventory_validation.py` PASS cleanly.
* All 256 project tests PASS cleanly.
* Phase 10D official evaluation matrix execution remains unstarted.
* Empirical Recall-Aware effectiveness remains strictly unknown.

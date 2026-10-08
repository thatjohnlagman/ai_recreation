# Phase 10A Integration Semantics and Technical Specifications

This document defines the strict integration semantics, execution contracts, and architectural rules enforced across Phase 10A components for the Recall-Aware IDS thesis project.

---

## 1. Evaluation Roles, Batch Structure, and Data Partitioning

### 1.1 Partition Sizes and Roles
The 90,000 evaluation rows (`data/processed/X_eval.parquet`) are partitioned into two strictly disjoint pools according to `data/manifests/evaluation_roles.csv`:
- **Crafting Pool**: Exactly 18,000 records (`role == "crafting"`). Reserved exclusively for surrogate model training and calibration references. Never evaluated in measurement batches.
- **Measurement Pool**: Exactly 72,000 records (`role == "measurement"`). Formatted into 144 sequential, immutable evaluation batches.
- **Batch Composition**: Defined in `data/manifests/evaluation_batches.csv`. Exactly 144 batches of 500 records each:
  - 17 batches contain 414 benign and 86 attack records.
  - 127 batches contain 415 benign and 85 attack records.
  - Total measurement records: $17 \times 500 + 127 \times 500 = 72,000$.

### 1.2 Identity Joins
- Resolving sample roles and batch assignments must always proceed via explicit inner join on `(_source_file, _raw_row_idx)` or deterministic `eval_position`.
- **Prohibited**: Inferring roles via arithmetic boundary thresholds (e.g. `eval_position >= 18000`).
- Missing, duplicated, or misaligned identities trigger immediate `ValueError`.

---

## 2. Attack Realization, Cache Provider, and Cache Builder Contracts

### 2.1 Attack Scenarios and Mathematical Rules
1. **Silent Probing**:
   - Identity attack scenario representing clean baseline traffic.
   - Attacked features are identical to input measurement features: $X_{\text{attacked}} = X_{\text{orig}}$.
   - Zero attack queries are made.
   - Status schema requirements: `eligible = False`, `attempted = False`, `successful = False`, `status_code = "NOT_APPLICABLE"`, `queries_used = 0`, distortion metrics ($L_0, L_1, L_2, L_\infty = 0.0$).
   - Attack Success Rate (ASR) is strictly `null` (not applicable).

2. **Surrogate Transfer**:
   - A surrogate Decision Tree is fitted strictly and exclusively on the 18,000 crafting records.
   - Adversarial perturbation generation uses the surrogate model only.
   - The target Random Forest internal weights, tree splits, and class probabilities are never exposed to the attack generation logic.
   - Target model verification is executed strictly through the `BlackBoxOracle` interface.
   - Target queries for screening eligibility and final candidate verification are tracked and counted separately.

3. **Decision Boundary**:
   - Hard-label oracle access only via `BlackBoxOracle` (maximum 50 queries per attempted target).
   - Exactly 10 deterministic midpoint bisection steps after identifying a valid counter-class endpoint.
   - Target selection: Exactly 200 deterministic targets globally across the entire 72,000 measurement pool for each `(scenario, seed)`.
   - Selection is global, not 200 per batch.
   - Clean screening queries (to confirm true positives) are explicitly distinguished from adversarial attack budget queries.
   - Boundary ASR denominator is exactly 200 attempted targets globally.

### 2.2 Attack Cache Provider (`ConcreteAttackCacheProvider`)
The provider validates the cache directory prior to serving batch data:
- Rejects missing, unparseable, or tampered `manifest.json`.
- Enforces strict 64-character lowercase hex string formatting for all provenance and file hashes. Rejects placeholder strings (`"0"*64`, `"a"*64`, `"dummy"`, `"todo"`).
- Requires non-empty dictionary structures for `attack_script_hashes`, `attack_parameters`, and `query_budgets`.
- Validates that artifact SHA-256 digests (`X_attacked.parquet`, `status.parquet`) match manifest declarations.
- Enforces exact row count (72,000 in official mode; configurable in test harnesses).
- Enforces strictly sorted, monotonically increasing `eval_position` values with zero duplicates and zero nulls.
- Enforces exact 78 feature column names and column ordering. Feature columns must be floating-point dtypes (no objects or strings).
- Enforces strict boolean types for `eligible`, `attempted`, and `successful` status columns (unconditional rejection of integer coercion).
- Enforces strict non-negative integer dtypes for `queries_used` (rejecting boolean types).
- Enforces status code vocabulary (`SUCCESS`, `TARGET_REJECTION`, `SURROGATE_REJECTION`, `BUDGET_EXCEEDED`, `NOT_APPLICABLE`).

### 2.3 Attack Cache Builder (`AttackCacheBuilder`)
- Generates canonical realizations for `(scenario, seed)` combinations.
- Writes to a `.tmp` staging directory before atomically moving to final publication path.
- Quarantines incomplete or failed generation attempts with timestamped identifiers rather than silently deleting them.
- Re-reads and validates generated artifacts through `ConcreteAttackCacheProvider` before publication.

---

## 3. Defense Adapter Semantics and Common Randomness

### 3.1 Feature Protection and Defense Scope
- **Total Features**: 78.
- **Defense-Eligible Features**: 63 features identified in `artifacts/preprocessors/feature_mask.json`.
- **Protected Features**: 15 critical protocol and identity features (`Dst Port`, `Protocol`, packet flags, timestamp features) that must never be altered by any defense.
- Defenses must guarantee that all 15 protected features remain bit-for-bit identical to raw input.

### 3.2 Individual Defense Implementations
1. **Adaptive Feature Poisoning (AFP)**:
   - Evaluates multiplicative noise formulation on eligible features:
     $$x'_j = x_j \cdot (1 + \epsilon \cdot \text{sign}(x_j - \mu_j))$$
     where $\mu_j$ is the benign centroid from `artifacts/preprocessors/afp_benign_profile.parquet`.
   - Output is projected into valid training bounds (`artifacts/preprocessors/training_bounds.parquet`).
   - Classifier inference is performed on defended array $X_{\text{afp}}$, never the original unperturbed array.
   - Classification score is model output probability: `predict_proba(X_afp)[:, 1]`.

2. **Feature Squeezing (FS)**:
   - Squeezing intensity $I \in [0.0, 2.0]$. Effective bit-depth: $d_{\text{eff}} = \max(0, 6 - \lfloor I \rfloor)$.
   - Bit-depth reduction applied to 63 modifiable features.
   - Classifier inference is performed on defended array $X_{\text{fs}}$.
   - Classification score is model output probability: `predict_proba(X_fs)[:, 1]`.

3. **Randomized Smoothing (RS)**:
   - Evaluates an ensemble of 11 Gaussian-perturbed variants ($\sigma = 0.0002$) for each input sample.
   - Perturbation applies only to the 63 eligible features; 15 protected features remain unperturbed.
   - Hard classification prediction is majority vote ($\ge 6$ positive votes).
   - Score for PR-AUC calculation is exact vote fraction: $\text{score} = \frac{\text{positive\_votes}}{11}$.

### 3.3 Common Randomness
- When Base policy (fixed intensity) and Recall-Aware policy operate at the identical intensity value for a batch, identical pseudo-random seeds guarantee identical defense realizations.

---

## 4. Recall-Aware Controller and Policy Mechanics

### 4.1 Timing Contract for Batch $t$
1. **Pre-Prediction**: The controller yields intensity $\theta_t$ based on history up to batch $t-1$. No ground-truth labels from batch $t$ are observed.
2. **Defend & Predict**: Batch $t$ is perturbed by the active defense at intensity $\theta_t$ and evaluated by the frozen RF classifier.
3. **Post-Prediction Feedback**: True positives ($TP_t$) and false negatives ($FN_t$) are computed by comparing predictions with ground truth.
4. **State Transition**: Observations $(TP_t, FN_t)$ are submitted to the controller to update internal rolling recall and determine intensity $\theta_{t+1}$ for batch $t+1$.

### 4.2 Rolling Recall Computation
- Rolling recall over sliding window $W$:
  $$R_t = \frac{\sum_{i=t-W+1}^t TP_i}{\sum_{i=t-W+1}^t TP_i + \sum_{i=t-W+1}^t FN_i}$$
- **Prohibited**: Averaging per-batch recall values $\frac{1}{W} \sum \text{Recall}_i$.

### 4.3 State Transitions
- **Red State**: $R_t < R_{\text{critical}} \implies \theta_{t+1} = \max(\theta_{\min}, \theta_t \cdot \text{fast\_decay})$.
- **Yellow State**: $R_{\text{critical}} \le R_t < R_{\min} \implies \theta_{t+1} = \max(\theta_{\min}, \theta_t \cdot \text{slow\_decay})$.
- **Green State**: $R_t \ge R_{\min} \implies \theta_{t+1} = \min(\theta_{\max}, \theta_t \cdot \text{growth\_factor})$.

### 4.4 Authorized Configurations
Only configurations C1 through C7 are authorized:
- C1: $W=5, R_{\text{crit}}=0.85, R_{\min}=0.95, \text{fast}=0.40, \text{slow}=0.90, \text{growth}=1.05$
- C2: $W=3, R_{\text{crit}}=0.85, R_{\min}=0.95, \text{fast}=0.40, \text{slow}=0.90, \text{growth}=1.05$
- C3: $W=20, R_{\text{crit}}=0.85, R_{\min}=0.95, \text{fast}=0.40, \text{slow}=0.90, \text{growth}=1.05$
- C4: $W=5, R_{\text{crit}}=0.80, R_{\min}=0.90, \text{fast}=0.40, \text{slow}=0.90, \text{growth}=1.05$
- C5: $W=5, R_{\text{crit}}=0.90, R_{\min}=0.97, \text{fast}=0.40, \text{slow}=0.90, \text{growth}=1.05$
- C6: $W=5, R_{\text{crit}}=0.85, R_{\min}=0.95, \text{fast}=0.60, \text{slow}=0.95, \text{growth}=1.02$
- C7: $W=5, R_{\text{crit}}=0.85, R_{\min}=0.95, \text{fast}=0.25, \text{slow}=0.80, \text{growth}=1.10$
- **C8 is strictly prohibited** and explicitly rejected by configuration parsers.

---

## 5. Experiment Matrix and Execution Aliasing

The experimental design comprises:
- **Primary References**: 3 attacks $\times$ 3 defenses $\times$ 2 policies (Base, RA-C1) $\times$ 5 seeds (42–46) = 90 logical runs.
- **Sensitivity References**: 3 attacks $\times$ 3 defenses $\times$ 7 controllers (C1–C7) $\times$ 3 seeds (42–44) = 189 logical runs.
- **Total Logical References**: 279.
- **Aliased References**: The 27 C1 sensitivity runs for seeds 42–44 alias the primary C1 runs.
- **Required Unique Executions**: Exactly 252 unique runs ($252 \times 144 = 36,288$ batch records).
- **RS Executions**: Exactly 84 unique runs.

---

## 6. Run Execution, Metrics, and Serialization

1. **Global Run-Level PR-AUC**:
   - Computed over all 72,000 aligned prediction scores collected sequentially across the entire run.
   - Batch-level PR-AUC values are never averaged to form the run metric.
2. **Confusion Matrix Invariants**:
   - Each batch confusion matrix ($TP + FP + TN + FN$) must sum to exactly 500.
   - Each run confusion matrix must sum to exactly 72,000.
3. **Atomic Execution**:
   - Intermediate outputs write to temporary files.
   - On run completion, output is re-opened, verified against schemas, and `completion.json` is written strictly last.
   - Failed or interrupted runs are quarantined with timestamped paths; recovery restarts execution at batch 0.

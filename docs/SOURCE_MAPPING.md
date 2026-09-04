# Source Mapping

| Component | Approved Thesis Definition | Prior-Paper Guidance | Implemented Decision | Exact Reproduction? |
|---|---|---|---|---|
| Dataset | CSE-CIC-IDS2018 (Chapter 1 & 3) | CIC-IDS2018 used in Springer paper | 10 CSV files from official dataset, read-only path configured in YAML | N/A |
| Binary classification | 0=benign, 1=attack (Chapter 1 & 3) | Binary classification used in both prior papers | Implemented exactly | Yes |
| 70/30 stratified split | Chapter 3, Appendix 1 §Pre-experimentation | Both prior papers use similar splits | Stratified split on binary label, seed=42 | Yes |
| RF classifier | Fixed Random Forest (Chapter 1, 3) | 200 estimators, depth 20, Gini (Springer) | RF(n_estimators=200, max_depth=20, criterion='gini', n_jobs=2, random_state=42) | No — n_jobs=2 for M2/8GB |
| AFP equation | **Multiplicative**: ε_i = ε_base × (1 + α × δ_i) (Chapter 3) | **Additive**: ε_i = ε_base + α × δ_i (AFP source paper) | Thesis multiplicative form implemented as `StudyDefinedAFP` | **No — thesis form used, not source paper** |
| AFP noise | Bounded noise (Chapter 3) | Bounded uniform used in source AFP | Uniform(−ε_i, +ε_i) — operationally equivalent and documented | No — operational choice |
| AFP benign profile | mu and sigma from benign training records (Chapter 3) | Similar in source paper | Benign-only training partition; sigma floor = 1e-6 | Approximately yes |
| AFP label | "Adaptive Feature Poisoning" | AFP / Behavior-Aware Defense | "Adaptive Feature Poisoning" / code: `StudyDefinedAFP` | N/A |
| Randomized Smoothing equation | x' = x + η; η ~ N(0,σ²); ŷ = mode(f(x'_1),...,f(x'_m)) (Chapter 3) | Cohen et al. 2019 | Implemented exactly; m=11 per handoff (odd) | Approximately yes |
| RS certification | Not claimed (Chapter 3 scope limitation) | Cohen 2019 provides certification | Monte Carlo implementation only — no certified robustness claimed | No |
| Feature Squeezing equation | d = max(0, 6 − int(intensity)); x' = round(x, d) (Chapter 3) | Xu et al. 2018 bit-depth reduction | Implemented exactly from thesis formula | Approximately yes |
| Controller formula | rolling_recall = ΣTP / (ΣTP + ΣFN) (Chapter 3) | Not in prior papers | Implemented exactly from thesis pseudocode | Yes |
| Controller states | Red/Critical, Yellow/Warning, Green/Healthy (Chapter 3) | Not in prior papers | Implemented exactly | Yes |
| Controller update | fast_decay, slow_decay, growth_factor with clip (Chapter 3) | Not in prior papers | Implemented exactly | Yes |
| Controller timing | Batch t intensity chosen before batch t labels seen (Chapter 3) | Not in prior papers | Enforced: label update applies to t+1 only | Yes |
| C1–C7 configurations | Table in Appendix 1 (Experiment Paper) | Not in prior papers | All 7 configurations implemented in controllers.yaml exactly | Yes |
| Silent probing scenario | Sequential original evaluation batches (Chapter 3 pseudocode) | Springer: random-walk + CPD + side-channel | Thesis definition: original batches submitted sequentially — NOT Springer pipeline | **No — study-defined only** |
| Surrogate transferability | Decision Tree surrogate (Chapter 3) | Springer: transferability via surrogate | DT surrogate trained on RF hard labels; tree-path attack; no RF internals accessed | Approximately yes |
| Decision-boundary attack | Binary search interpolation toward benign (Chapter 3 pseudocode) | Springer: boundary search | Binary search along attack-to-benign line; hard labels only; fixed query budget | Approximately yes |
| Metrics | Precision, Recall, F1-Score (attack=positive) (Chapter 1, 3) | Same in both papers | Computed exactly from TP/FP/TN/FN counts | Yes |
| Statistical test | Two-tailed paired t-test, α=0.05 (Chapter 3) | Not specified | scipy.stats.ttest_rel; pairing key: (seed, attack, defense, batch_id) | Yes |
| RQ4 sensitivity | Descriptive, C1–C7 (Appendix 1) | Not in prior papers | Descriptive only — C1 is not replaced retroactively | Yes |
| Working sample | Not specified in thesis (hardware constraint) | Not relevant | 300,000 records stratified from full dataset per handoff | N/A |

## Differences from Prior Papers

### AFP Equation Difference
- **Source AFP paper (Ennaji et al., arXiv 2512.13501):** ε_i = ε_base + α × δ_i (additive form)
- **Thesis Chapter 3:** ε_i = ε_base × (1 + α × δ_i) (multiplicative form)
- **Decision:** Thesis form implemented. Difference recorded here and in IMPLEMENTATION_DECISIONS.md.

### Silent Probing Scope Difference
- **Springer paper:** Full random-walk + change-point detection + causality analysis + side-channel
- **Thesis Chapter 3:** Sequential original evaluation batches as the primary condition
- **Decision:** Thesis definition implemented as primary scenario. Springer pipeline is not reproduced.

### RS Ensemble Size
- **Prompt default:** m=51
- **Handoff hardware constraint:** m=11 (validated on training calibration data)
- **Decision:** m=11 used per handoff for M2/8GB hardware.

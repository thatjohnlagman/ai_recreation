# AUDIT FINDINGS

1. **Static vs Live Confusion (Severity: Medium)**
   - **Evidence**: `frontend/results.js` defaults to `STATIC_C1_C7_MATRIX` when `useLiveBenchmark` is false. These static values are the true averaged thesis results.
   - **Consequence**: An observer seeing "Live Benchmark: OFF" might think the data is purely hypothetical or from an inactive local run, when it is actually the verified research data. 
   - **Action**: Add a clear label or tooltip in the UI explaining that "OFF" displays the fully compiled `.research_repo` research aggregates.

2. **Ground-Truth Leakage in Demo (Severity: Low/Expected)**
   - **Evidence**: `attacker_sim.py` transmits true adversarial labels to `server.py` to simulate controller metric feedback.
   - **Consequence**: In operational reality, controllers must estimate recall/precision via oracle sampling. The demo seamlessly simulates this for visual representation.
   - **Action**: Acknowledge this simulation architecture in the presentation.

3. **Missing Config Mappings (Severity: Low)**
   - **Evidence**: `results.js` C1-C7 config names ("Short Win", "Aggress.", etc.) are hardcoded and loosely mapped to `STATIC_C1_C7_MATRIX` scraped from the repo.
   - **Consequence**: Slight mismatch in naming conventions.
   - **Action**: Ensure presentation narrative bridges the C1-C7 numeric identifiers with their descriptive strategies.

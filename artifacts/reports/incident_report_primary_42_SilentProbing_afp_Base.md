# Incident Record: `primary_42_SilentProbing_afp_Base`

**Date:** 2026-09-23
**Run ID:** `primary_42_SilentProbing_afp_Base`

## 1. Original Interruption
During the parallel evaluation execution of the 252 unique runs, the host machine went to sleep/lock state, leading to a deliberate, graceful interruption (via `SIGINT/Ctrl+C`) of the active evaluation workers to safely preserve system state. At the exact moment of interruption, 234 out of 252 unique runs had securely completed and published their final outputs. 

However, one specific staging directory belonging to the actively processing run—`primary_42_SilentProbing_afp_Base`—was interrupted mid-execution. 

## 2. Temporary False Quarantine
As part of the subsequent orchestration diagnostics, we identified a provenance bug causing `ValueError` rejections during alias publication. While iteratively repairing `scripts/run_evaluation.py` to fix this provenance mismatch, an interim bug fix was temporarily tested that attempted to dynamically inject `manifest_sha256` into the core dictionary returned by `validate_cache_against_inventory`.

When the repair was briefly tested against the filesystem, this injection caused the validation logic to reject `primary_42_SilentProbing_afp_Base` (and attempt to quarantine it) because its existing `run_summary.json` correctly did not contain the artificially injected hash. The provenance injection bug was diagnosed and reverted, establishing the canonical rule that `manifest_sha256` must be calculated strictly from the file system and placed *only* into `target_run_provenance` during execution and alias validation.

## 3. Final Clean Recomputation
Because `primary_42_SilentProbing_afp_Base` was actively interrupted during the original sleep event, its staging directory was inherently incomplete. The `quarantined` directory suffix was manually removed to restore the directory to its target path. 

However, the manually restored target still contained `quarantine_reason.json`. Upon running the final, unfiltered `./.venv-m4/bin/python scripts/run_evaluation.py --execute` pass, the canonical final pass rejected that restored directory as invalid due to unexpected files.

The runner renamed it to:
`primary_42_SilentProbing_afp_Base_quarantined_20260923070229061127`

The runner then restarted the run from batch 0 in a new staging directory.
- All 144 batches and 72,000 records were regenerated.
- Controller/policy state was reset.
- No partial output was resumed or copied.
- The new result was atomically published.
- One historical quarantine directory remains as incident evidence.

## 4. Current Directory Status
- **active staging directories:** 0
- **unresolved/incomplete final runs:** 0
- **historical quarantine directories:** 1

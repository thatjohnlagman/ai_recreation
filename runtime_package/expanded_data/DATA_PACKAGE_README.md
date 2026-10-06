# DATA_PACKAGE_README

## Export Scope
This data package provides an exact read-only export of the 90,000-row Phase 10 frozen evaluation dataset for the Recall-Aware IDS simulator. The files are provided byte-for-byte to retain positional alignment, existing preprocessing states, and full provenance context. No changes were made to the original files or methodology.

## Snapshot details
- **Repository Root:** `/Users/trumpler-mac/Downloads/thesissep20276/september 2026/recall-aware-ids`
- **Branch:** `phase10b-cache-orchestration`
- **Commit:** `19ee50fc34bf4e2303ddfaf5340cd9c59d4c4b9f`
- **Working Tree State:** Contains untracked demo package and some uncommitted phase 11 evidence/hash edits. State was unaltered by this extraction.

## Dataset Overview
- **Total Rows:** 90,000
- **Features:** 78
- **Role Split:** 18,000 crafting | 72,000 measurement
- **Class Split:** 74,679 benign | 15,321 attack

## Receiving Tool Integration Instructions
Please follow these requirements when integrating:
- **Fixture Fallback:** Preserve the existing 20-flow fixture for regression tests and fallback rehearsals.
- **Alignment:** Load matching expanded data/metadata and role manifests on both backend and attacker. A sample ID must resolve to the same original evaluation row on both sides. Distinguish expanded IDs from the old fixture IDs.
- **Roles:** Use measurement rows as simulation targets and crafting rows for surrogate queries/benign boundary references. Keep their identities separate.
- **Provenance:** Retain dataset provenance and labels, and record the original evaluation position when a candidate is transformed.
- **Preprocessing:** Feed the already-scaled features directly to the current model/defense pipeline. **DO NOT** re-scale the features.
- **Sampling:** Shuffle without replacement for varied sessions, with a recorded seed and explicit handling of pool exhaustion.
- **Surrogate Fitting:** Fit surrogates using a bounded, recorded selection from the crafting pool; loading 90,000 rows does not require sending all rows over HTTP for each action.
- **Metrics Separation:** Compare defenses on identical target sequences with fresh or separate metrics. Preserve the distinction between crafting/query telemetry and measured target results.
- **Batch Cadence:** Keep the interactive five-labeled-attack-decision controller cadence disclosed. This data export does not change the controller or reproduce the formal 500-flow research batching automatically.
- **Predictions:** Source observed recall/FPR from predictions. A larger pool does not guarantee errors or different defense verdicts.

## Missing or Unverified Items
See `OMISSIONS.md` for a comprehensive list of what was excluded (e.g., the massive RF joblib, raw partitions, and evaluation caches). Validation of the model bounds passed qualitatively by confirming presence of the correct manifest, but out-of-bounds metrics were not explicitly derived because out-of-bounds on evaluation data is an expected norm.

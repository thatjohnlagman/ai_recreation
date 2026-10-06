# Validation Report

| Check | Expected | Observed | Status |
| --- | --- | --- | --- |
| Features | 90,000 rows x 78 numeric cols | 90,000 rows x 78 numeric cols (all finite) | **PASS** |
| Names Align | Match `feature_names.json` | Exact match | **PASS** |
| Metadata | 90,000 aligned rows, 0 nulls | 90,000 aligned rows, 0 nulls on expected columns | **PASS** |
| Row Identity | Mappable `eval_position` | Implicit 0-based row index maps exactly 1-to-1 to CSV manifests | **PASS** |
| Roles | 18,000 crafting, 72,000 measurement | 18,000 crafting, 72,000 measurement (no overlap) | **PASS** |
| Batch Size | 72,000 measurement in batches | 72,000 across 144 batches (500 rows each) | **PASS** |
| Index Bases | Document actual bases | Batch ID base is `1`, within-batch base is `0` | **PASS** |
| Distribution | 74,679 benign, 15,321 attack | 74,679 benign, 15,321 attack (crafting split matches target counts) | **PASS** |
| Preprocessing | Scaled with training scaler | Confirmed from reports, features are normalized. Do not rescale. | **PASS** |
| Model Hash | `9608672c...` | `9608672c5d5e38a9272c560679cdd2291399de0e0917c8c9dff373bfd200f51d` | **PASS** |

### Additional Notes
All read-only assertions passed flawlessly. No source files were modified, repaired, or altered during the execution of the validation script. All counts precisely matched the expected quantities mapped in the original thesis documents.

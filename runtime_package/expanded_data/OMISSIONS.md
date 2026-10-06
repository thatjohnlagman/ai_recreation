# Omissions

The following files and items were INTENTIONALLY EXCLUDED from this export payload to preserve portability and respect the simulator data contract requirements:

- **Raw Datasets:** Massive multi-gigabyte raw datasets (such as raw PCAPs and CSVs from the CIC-IDS2018 corpus).
- **Training/Calibration Feature Partitions:** Excluded. Target pool targets only the 90,000 evaluation rows.
- **Massive Evaluation Caches:** Excluded.
- **Virtual Environments:** Excluded (`.venv` folders).
- **Git History:** Excluded (`.git` folders).
- **Secrets / Config Credentials:** Excluded.
- **RF Joblib Model:** The duplicate `frozen_rf.joblib` large file (52MB) was explicitly excluded because the exact same RF weights already exist in the target tool package, and the exact model checksum has been independently verified.
- **Unrelated Research Files:** All code, attack logic, tests, and irrelevant thesis PDF files are excluded.

with open("tests/test_phase11a_analysis_readiness.py", "r") as f:
    content = f.read()

dummy_cache = '{\n            "attacks_yaml_hash": "a" * 64,\n            "controllers_yaml_hash": "a" * 64,\n            "defenses_yaml_hash": "a" * 64,\n            "evaluation_batches_hash": "a" * 64,\n            "evaluation_roles_hash": "a" * 64,\n            "experiment_yaml_hash": "a" * 64,\n            "feature_mask_hash": "a" * 64,\n            "feature_names_hash": "a" * 64,\n            "frozen_rf_hash": "a" * 64,\n            "scaler_hash": "a" * 64,\n            "training_bounds_hash": "a" * 64\n        }'

content = content.replace('"cache_identity": {},', f'"cache_identity": {dummy_cache},')

with open("tests/test_phase11a_analysis_readiness.py", "w") as f:
    f.write(content)

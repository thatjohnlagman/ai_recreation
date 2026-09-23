with open("tests/test_phase11a_analysis_readiness.py", "r") as f:
    content = f.read()

content = content.replace(
    '    with patch("scripts.build_analysis_tables.EVAL_DIR", tmp_path):\n        rr, brs, crs = process_run(row)',
    '    row = {"run_id": "test_run", "attack_scenario": "FeatureSqueezing", "seed": 42, "defense_name": "Base", "controller_config_id": "Base"}\n    with patch("scripts.build_analysis_tables.EVAL_DIR", tmp_path):\n        rr, brs, crs = process_run(row)'
)
content = content.replace(
    '    with patch("scripts.build_analysis_tables.EVAL_DIR", tmp_path):\n        with pytest.raises(ValueError, match="Batch count"):',
    '    row = {"run_id": "test_run", "attack_scenario": "FeatureSqueezing", "seed": 42, "defense_name": "Base", "controller_config_id": "Base"}\n    with patch("scripts.build_analysis_tables.EVAL_DIR", tmp_path):\n        with pytest.raises(ValueError, match="Batch count"):'
)
content = content.replace(
    '    with patch("scripts.build_analysis_tables.EVAL_DIR", tmp_path):\n        with pytest.raises(ValueError, match="out of order"):',
    '    row = {"run_id": "test_run", "attack_scenario": "FeatureSqueezing", "seed": 42, "defense_name": "Base", "controller_config_id": "Base"}\n    with patch("scripts.build_analysis_tables.EVAL_DIR", tmp_path):\n        with pytest.raises(ValueError, match="out of order"):'
)
content = content.replace(
    '    with patch("scripts.build_analysis_tables.EVAL_DIR", tmp_path):\n        with pytest.raises(ValueError, match="completion.json run_id mismatch"):',
    '    row = {"run_id": "test_run", "attack_scenario": "FeatureSqueezing", "seed": 42, "defense_name": "Base", "controller_config_id": "Base"}\n    with patch("scripts.build_analysis_tables.EVAL_DIR", tmp_path):\n        with pytest.raises(ValueError, match="completion.json run_id mismatch"):'
)
content = content.replace(
    '    with patch("scripts.build_analysis_tables.EVAL_DIR", tmp_path):\n        with pytest.raises(ValueError, match="Batch ID mismatch"):',
    '    row = {"run_id": "test_run", "attack_scenario": "FeatureSqueezing", "seed": 42, "defense_name": "Base", "controller_config_id": "Base"}\n    with patch("scripts.build_analysis_tables.EVAL_DIR", tmp_path):\n        with pytest.raises(ValueError, match="Batch ID mismatch"):'
)

with open("tests/test_phase11a_analysis_readiness.py", "w") as f:
    f.write(content)

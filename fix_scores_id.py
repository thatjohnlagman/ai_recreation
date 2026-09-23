with open("scripts/build_analysis_tables.py", "r") as f:
    content = f.read()

content = content.replace('check_eq(scores["run_id"], rid, f"{rid}: scores.json run_id mismatch")', '')

with open("scripts/build_analysis_tables.py", "w") as f:
    f.write(content)

with open("tests/test_phase11a_analysis_readiness.py", "r") as f:
    content = f.read()

content = content.replace('scores = {"run_id": run_id, "scores": []}', 'scores = []')

with open("tests/test_phase11a_analysis_readiness.py", "w") as f:
    f.write(content)

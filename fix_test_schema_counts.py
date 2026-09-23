with open("tests/test_phase11a_analysis_readiness.py", "r") as f:
    content = f.read()

content = content.replace('"status_code_counts": {},', '"status_code_counts": {"200": 72000},')
content = content.replace('"total_queries": 1000,', '"total_queries": 72000,')

with open("tests/test_phase11a_analysis_readiness.py", "w") as f:
    f.write(content)

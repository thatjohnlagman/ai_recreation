with open("tests/test_phase11a_analysis_readiness.py", "r") as f:
    content = f.read()

dummy_lx = '{"min": 0.0, "mean": 0.0, "std": 0.0, "max": 0.0}'
content = content.replace('"l0_summary": {},', f'"l0_summary": {dummy_lx},')
content = content.replace('"l1_summary": {},', f'"l1_summary": {dummy_lx},')
content = content.replace('"l2_summary": {},', f'"l2_summary": {dummy_lx},')
content = content.replace('"linf_summary": {},', f'"linf_summary": {dummy_lx},')

with open("tests/test_phase11a_analysis_readiness.py", "w") as f:
    f.write(content)

with open("tests/test_phase11a_analysis_readiness.py", "r") as f:
    content = f.read()

content = content.replace('"tp": batches * 250,\n        "fp": batches * 250,\n        "tn": 0,\n        "fn": 0,', '"tp": batches * 125,\n        "fp": batches * 125,\n        "tn": batches * 125,\n        "fn": batches * 125,')

content = content.replace('"tp": 250, "fp": 250, "tn": 0, "fn": 0,', '"tp": 125, "fp": 125, "tn": 125, "fn": 125,')

with open("tests/test_phase11a_analysis_readiness.py", "w") as f:
    f.write(content)

with open("tests/test_phase11a_analysis_readiness.py", "r") as f:
    content = f.read()

content = content.replace('"defense_name": "Base"', '"defense_name": "afp"')
content = content.replace('"defense": "Base"', '"defense": "afp"')

with open("tests/test_phase11a_analysis_readiness.py", "w") as f:
    f.write(content)

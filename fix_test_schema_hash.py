with open("tests/test_phase11a_analysis_readiness.py", "r") as f:
    content = f.read()

dummy_hash = "b5bb9d8014a0f9b1d61e21e796d78dccdf1352f23cd32812f4850b878ae4944c" # sha256 of "dummy"
content = content.replace('"a" * 64', f'"{dummy_hash}"')

with open("tests/test_phase11a_analysis_readiness.py", "w") as f:
    f.write(content)

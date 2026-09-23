with open("scripts/build_analysis_tables.py", "r") as f:
    content = f.read()

content = content.replace(
    'if row["attack_scenario"] != "SilentProbing":',
    'if canonicalize_scenario(row["attack_scenario"]) != "SilentProbing":'
)

with open("scripts/build_analysis_tables.py", "w") as f:
    f.write(content)

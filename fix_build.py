import re

with open("scripts/build_analysis_tables.py", "r") as f:
    content = f.read()

# Add import
content = content.replace("from scripts.run_evaluation import (", "from src.recall_aware_ids.experiment.schemas import RunSummary\nfrom scripts.run_evaluation import (")

# Replace contract validation
old_validation = """    # Contract validation for ASR
    if row["attack_scenario"] == "SilentProbing":
        check_eq(summary.get("attack_success_rate"), None, f"{rid}: SilentProbing ASR must be null")
    else:
        # ASR is valid for other attacks
        pass"""

new_validation = """    # Contract validation for ASR and PR-AUC
    # We use the authoritative production schema to enforce contracts
    try:
        RunSummary(**summary)
    except Exception as e:
        raise ValueError(f"{rid}: Schema validation failed: {e}")"""

content = content.replace(old_validation, new_validation)

# Check run_id agreement across files
old_process_start = """    with open(run_dir / "run_summary.json") as f:
        summary = json.load(f)
    with open(run_dir / "completion.json") as f:
        completion = json.load(f)
    with open(run_dir / "confusion.json") as f:
        confusions = json.load(f)
    with open(run_dir / "config.json") as f:
        configs = json.load(f)"""

new_process_start = """    with open(run_dir / "run_summary.json") as f:
        summary = json.load(f)
    with open(run_dir / "completion.json") as f:
        completion = json.load(f)
    with open(run_dir / "confusion.json") as f:
        confusions = json.load(f)
    with open(run_dir / "config.json") as f:
        configs = json.load(f)
    with open(run_dir / "scores.json") as f:
        scores = json.load(f)
        
    check_eq(summary["run_id"], rid, f"{rid}: run_summary.json run_id mismatch")
    check_eq(completion["run_id"], rid, f"{rid}: completion.json run_id mismatch")
    check_eq(scores["run_id"], rid, f"{rid}: scores.json run_id mismatch")"""

content = content.replace(old_process_start, new_process_start)

with open("scripts/build_analysis_tables.py", "w") as f:
    f.write(content)

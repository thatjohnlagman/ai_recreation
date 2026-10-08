import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]

def run_audit():
    feat_names_path = PROJECT_ROOT / "artifacts" / "preprocessors" / "feature_names.json"
    feat_mask_path = PROJECT_ROOT / "artifacts" / "preprocessors" / "feature_mask.json"
    
    with open(feat_names_path) as f:
        feature_names = json.load(f)
        
    with open(feat_mask_path) as f:
        mask_data = json.load(f)
        feature_mask = mask_data["feature_mask"]
        
    audit_results = []
    
    for name, eligible in zip(feature_names, feature_mask):
        category = "ambiguous"
        evidence = "No authoritative schema document found"
        
        # Exact known categories from methodology or basic networking definitions
        lower = name.lower()
        
        # Flags are binary
        if "flag" in lower or name == "Fwd PSH Flags" or name == "Bwd PSH Flags" or name == "Fwd URG Flags" or name == "Bwd URG Flags":
            category = "binary flag"
            evidence = "Networking concept (TCP flags)"
            
        # Rates are continuous
        elif "/s" in lower or "rate" in lower:
            category = "continuous measurement"
            evidence = "Derived rate field"
            
        # Averages, Means, Stds, Variances are continuous
        elif "avg" in lower or "mean" in lower or "std" in lower or "var" in lower:
            category = "continuous measurement"
            evidence = "Statistical aggregation"
            
        # Raw counts
        elif "pkts" in lower and "/s" not in lower:
            category = "integer count"
            evidence = "Packet count"
        elif "byts" in lower and "/s" not in lower:
            category = "integer count"
            evidence = "Byte count"
        elif "cnt" in lower or "count" in lower:
            category = "integer count"
            evidence = "Explicit count field"
        elif "port" in lower or "protocol" in lower:
            category = "categorical code"
            evidence = "Networking standard"
            
        agrees = True
        ambiguity = ""
        
        if eligible and category != "continuous measurement":
            agrees = False
            ambiguity = f"Feature is {category} but is marked eligible under 'continuous_numerical' rules."
            
        audit_results.append({
            "name": name,
            "eligible": eligible,
            "category": category,
            "agrees": agrees,
            "evidence": evidence,
            "ambiguity": ambiguity
        })
        
    with open(PROJECT_ROOT / "artifacts" / "reports" / "feature_mask_audit.md", "w") as f:
        f.write("# Feature Mask Audit\n\n")
        f.write("| Feature Name | Eligible | Category | Agrees with `continuous_numerical`? | Evidence | Ambiguity |\n")
        f.write("|---|---|---|---|---|---|\n")
        for res in audit_results:
            f.write(f"| {res['name']} | {res['eligible']} | {res['category']} | {res['agrees']} | {res['evidence']} | {res['ambiguity']} |\n")
            
    print("Mask audit complete.")

if __name__ == "__main__":
    run_audit()

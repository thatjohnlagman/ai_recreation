import os
import pandas as pd

def build_thesis_matrix():
    print("--- 📊 ASSEMBLING FINAL THESIS METRICS MATRIX ---")
    
    # 1. Load all individual CSV logs
    logs = [
        "attacks/probing_log.csv",
        "attacks/controlled_probing_log.csv", 
        "attacks/transfer_log.csv",
        "attacks/controlled_transfer_log.csv",
        "attacks/boundary_log.csv",
        "attacks/controlled_boundary_log.csv"
    ]
    
    # Initialize the core matrix dictionary
    matrix = {
        "Silent_Probing": {"Naked_Recall": None, "Static_AFP_Recall": None, "RAAFP_Recall": None},
        "Surrogate_Transfer": {"Naked_Recall": None, "Static_AFP_Recall": None, "RAAFP_Recall": None},
        "Decision_Boundary": {"Naked_Recall": None, "Static_AFP_Recall": None, "RAAFP_Recall": None}
    }
    
    for log_path in logs:
        if not os.path.exists(log_path):
            print(f"Warning: {log_path} not found.")
            continue
            
        df = pd.read_csv(log_path)
        attack_type = df["Attack_Type"].iloc[0]
        
        # Populate Naked and Static AFP Recalls from the uncontrolled logs
        if attack_type == "Silent_Probing":
            matrix["Silent_Probing"]["Naked_Recall"] = df["Original_Recall"].iloc[0]
            matrix["Silent_Probing"]["Static_AFP_Recall"] = df["Collapsed_Recall"].iloc[0]
            
        elif attack_type == "Surrogate_Transfer":
            matrix["Surrogate_Transfer"]["Naked_Recall"] = df["Original_Recall"].iloc[0]
            matrix["Surrogate_Transfer"]["Static_AFP_Recall"] = df["Collapsed_Recall"].iloc[0]
            
        elif attack_type == "Decision_Boundary":
            matrix["Decision_Boundary"]["Naked_Recall"] = df["Original_Recall"].iloc[0]
            matrix["Decision_Boundary"]["Static_AFP_Recall"] = df["Collapsed_Recall"].iloc[0]
            
        # Populate RAAFP Recalls from the Novelty (controlled) logs
        elif attack_type == "Controlled_Silent_Probing":
            matrix["Silent_Probing"]["RAAFP_Recall"] = df["Controlled_Recall"].iloc[0]
            
        elif attack_type == "Controlled_Surrogate_Transfer":
            matrix["Surrogate_Transfer"]["RAAFP_Recall"] = df["Controlled_Recall"].iloc[0]
            
        elif attack_type == "Controlled_Decision_Boundary":
            matrix["Decision_Boundary"]["RAAFP_Recall"] = df["Controlled_Recall"].iloc[0]
            
    # Structure into final DataFrame
    final_df = pd.DataFrame.from_dict(matrix, orient='index').reset_index()
    final_df.rename(columns={"index": "Attack_Vector"}, inplace=True)
    
    # Write to explicitly formatted thesis matrices
    output_path = "THESIS_RESULTS_MATRIX.csv"
    final_df.to_csv(output_path, index=False)
    
    print("\n✅ MATRIX SUCCESSFULLY GENERATED.")
    print("--------------------------------------------------")
    print(final_df.to_string(index=False))
    print("--------------------------------------------------")
    print(f"Saved cleanly to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    build_thesis_matrix()

import time
import numpy as np

def run_pilot():
    print("Running M2/8GB Pilot using synthetic training-derived subsets...")
    
    # Simulate a chunk of 50,000 records
    n_samples = 50000
    n_features = 78
    
    start_mem = 0 # would use psutil in reality
    
    print(f"Allocating {n_samples}x{n_features} synthetic float32 matrix...")
    X = np.random.rand(n_samples, n_features).astype(np.float32)
    
    print("Simulating AFP cost...")
    start_t = time.time()
    noise = np.random.uniform(-1, 1, size=X.shape).astype(np.float32)
    X_afp = X + noise * 0.0003
    t_afp = time.time() - start_t
    
    print("Simulating FS cost...")
    start_t = time.time()
    X_fs = np.round(X * (2**2 - 1)) / (2**2 - 1)
    t_fs = time.time() - start_t
    
    print("Simulating RS cost (11 ensemble members)...")
    start_t = time.time()
    for _ in range(11):
        noise = np.random.normal(0, 1, size=X.shape).astype(np.float32)
        X_rs = X + noise * 0.0002
    t_rs = time.time() - start_t
    
    print(f"\n--- Pilot Timing Results (per {n_samples} records) ---")
    print(f"AFP Time: {t_afp:.4f}s")
    print(f"FS Time:  {t_fs:.4f}s")
    print(f"RS Time:  {t_rs:.4f}s")
    
    # 252 unique runs * 72,000 records per run = 18,144,000 records total to process
    total_records = 252 * 72000
    batches_of_50k = total_records / 50000
    
    est_afp = (t_afp * batches_of_50k) / 3 # Assuming 1/3 of runs are AFP
    est_fs = (t_fs * batches_of_50k) / 3
    est_rs = (t_rs * batches_of_50k) / 3
    
    total_est = est_afp + est_fs + est_rs
    
    print(f"\n--- Projected Matrix Runtime ---")
    print(f"Estimated compute time for defenses: {total_est/60:.2f} minutes")
    print("NOTE: Attack generation time (Surrogate/Boundary) will dominate this.")

if __name__ == "__main__":
    run_pilot()

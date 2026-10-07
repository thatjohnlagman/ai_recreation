import os
import sys

os.environ["IDS_DATA_PROFILE"] = "expanded"
sys.path.append(os.path.abspath("."))
from attacker_sim import run_continuous_stream, check_backend_compatibility, get_ctx
import requests

with open('.operator_token', 'r') as f:
    TOKEN = f.read().strip()
    
HEADERS = {"X-Operator-Token": TOKEN}
BASE_URL = "http://localhost:8000"

def set_mode(mode):
    resp = requests.post(f"{BASE_URL}/api/dashboard/set-mode", json={"mode": mode}, headers=HEADERS)
    if resp.status_code != 200:
        print(f"Failed to set mode! {resp.status_code}")

def reset():
    resp = requests.post(f"{BASE_URL}/api/dashboard/reset", headers=HEADERS)
    if resp.status_code != 200:
        print(f"Failed to reset! {resp.status_code}")

def run_test(mode, count=20000):
    print(f"\n--- Testing {mode.upper()} MODE ({count} flows) ---")
    reset()
    set_mode(mode)
    
    get_ctx(profile="expanded", seed=42)
    
    run_continuous_stream(delay=0.0, count=count)
    
    resp = requests.get(f"{BASE_URL}/api/history/sessions", headers=HEADERS)
    sessions = resp.json()
    sess = sessions[0]
    
    print(f"\nRESULTS FOR {mode.upper()}:")
    tp, fp, tn, fn = sess['tp'], sess['fp'], sess['tn'], sess['fn']
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f"TP: {tp}, FN: {fn}, TN: {tn}, FP: {fp}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1 Score: {f1:.2%}")
    return sess

if not check_backend_compatibility(require_profile="expanded"):
    print("Backend mismatch")
    sys.exit(1)

sess_ra = run_test("recall-aware", count=20000)
sess_base = run_test("base", count=20000)

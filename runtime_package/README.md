# Recall-Aware IDS Demonstration

This is a standalone, portable demonstration package for the Recall-Aware ML-IDS defense simulation.

## 1. What this tool demonstrates
This tool demonstrates the behavioral difference between standard baseline ML defense mechanisms and defenses augmented with a Recall-Aware feedback controller.

> **Scope Note**: This standalone tool is a flow-feature HTTP simulation for interactive demonstration. It does not reproduce or substitute for the 144-batch × 500-sample Phase 10/11 frozen evaluation cache or official thesis statistical tables.

## 2. What AFP is
**Adaptive Feature Poisoning (AFP)** is an inference-time defense mechanism. It adds deviation-scaled bounded noise across modifiable features (respecting the 63 modifiable / 15 protected feature mask) to disrupt adversarial perturbations. It does **not** project flows toward a benign centroid.
- Calibrated Phase 8 parameters: $\varepsilon = 0.0003$, $\alpha = 0.5$ (step size), clipping to feature training bounds.

## 3. What Recall-Aware adds
The **Recall-Aware (RA) controller (C1)** dynamically adjusts defense intensity based on observed detection performance. In this interactive demonstration, the local controller cadence evaluates after every 5 labeled attack decisions. In the formal frozen research protocol, Recall-Aware C1 evaluated over a window of 5 completed batches (500 flows/batch).

## 4. The Three Attacks
- **Silent Probing**: Sequential submission of unchanged evaluation flows (0 queries, no iterative perturbation crafted). Follows `runtime_package/attacks/silent_probing.py` and approved thesis methodology.
- **Surrogate Transferability**: Trains a local surrogate decision model using hard labels from the target oracle and crafts adversarial examples against the surrogate to test black-box transferability.
- **Decision Boundary**: Uses benign reference samples and 1D bisection boundary-search behavior to find adversarial examples with a query budget (50 max queries).

## 5. The Three Defenses
- **Adaptive Feature Poisoning (AFP)**: Deviation-scaled bounded noise applied to modifiable features ($\varepsilon = 0.0003$, $\alpha = 0.5$).
- **Randomized Smoothing (RS)**: Adds Gaussian noise $\mathcal{N}(0, \sigma^2)$ to modifiable features across an ensemble of 11 noise samples ($\sigma = 0.0002$).
- **Feature Squeezing (FS)**: Reduces numerical precision of features via bit-depth/step rounding (intensity $d = 2$).

## 6. Base vs Recall-Aware
- **Base**: The defense operates at a static, configured intensity.
- **Recall-Aware**: The defense intensity is dynamically scaled by the Recall Controller based on feedback.

## 7. How to Install
```bash
pip install -r requirements.txt
```
*(Tested environment: Python 3.13 / 3.11 with scikit-learn==1.6.1 matching the frozen RF serialization).*

## 8. How to Run the Offline CLI
Run from repository root:
```bash
python runtime_package/run.py --attack silent_probing --defense afp --controller recall-aware
python runtime_package/run.py --attack decision_boundary --defense afp --controller base
```

## 9. How to Run the Interactive SOC Dashboard
Start the local FastAPI server:
```bash
python server.py
```
Open `http://localhost:8000` in your web browser. All frontend assets (Leaflet, Chart.js) are bundled locally in `frontend/vendor/` for full offline rehearsal.

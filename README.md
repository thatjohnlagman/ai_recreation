# Recall-Aware IDS Demonstration

This is a standalone, portable demonstration package for the Recall-Aware Adaptive Feature Perturbation (AFP) ML-IDS.

## 1. What this tool demonstrates
This tool demonstrates the behavioral difference between a standard ML defense mechanism and the same mechanism augmented with a Recall-Aware feedback controller.

## 2. What AFP is
Adaptive Feature Poisoning (AFP) is the existing baseline defense mechanism. It works by perturbing features at inference time to disrupt adversarial attacks.

## 3. What Recall-Aware adds
The Recall-Aware (RA) controller dynamically adjusts the perturbation intensity of the defense (like AFP) based on a rolling window of recent recall. It is the proposed contribution of this thesis.

## 4. The Three Attacks
- **Silent Probing**: Iteratively perturbs features without querying the target model.
- **Surrogate Transferability**: Trains a local surrogate model using hard labels from the target and crafts adversarial examples against the surrogate.
- **Decision Boundary**: Uses benign reference samples and boundary-search behavior to find adversarial examples.

## 5. The Three Defenses
- **Randomized Smoothing (RS)**: Adds Gaussian noise to inputs.
- **Feature Squeezing (FS)**: Reduces the color depth or precision of features.
- **Adaptive Feature Poisoning (AFP)**: Defends by dynamically perturbing features based on a profile.

## 6. Base vs Recall-Aware
- **Base**: The defense operates at a static, configured intensity.
- **Recall-Aware**: The defense intensity is dynamically scaled by the Recall Controller.

## 7. How to Install
```bash
pip install -r requirements.txt
```

## 8. How to Run
```bash
python run.py --attack decision_boundary --defense afp --controller base
python run.py --attack decision_boundary --defense afp --controller recall-aware
```

## 9. How to Run a Comparison
Run the comparison mode to see No Defense vs Base vs Recall-Aware side-by-side:
```bash
python run.py --attack decision_boundary --defense afp --compare
```

## 10. What the dashboard means
You can also launch a Streamlit dashboard to interactively configure and run these simulations:
```bash
streamlit run dashboard.py
```
The dashboard will display the exact command-line output showing detection rates, evasions, and perturbation intensity.

## 11. Research vs Evaluation
This is a small-scale research demonstration using a minimal set of samples to prove the behavior of the system. It does not replace or represent the full 16M+ sample Phase 10 frozen evaluation cache.

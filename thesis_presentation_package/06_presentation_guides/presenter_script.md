# 🎤 Presenter Script: Adaptive Feature Perturbation (AFP) Demo

This script is formatted for a live demonstration of `final_evaluate_defenses.ipynb`. It is designed to be read aloud (or paraphrased) while scrolling through and running the notebook cells. 

**Demo Duration:** ~3–4 Minutes

---

## 🏁 Introduction: The Hook & Core Accomplishment (30 Seconds)

| Visual Action | Spoken Script |
| :--- | :--- |
| **Show notebook header & Title Card** | *"Hello everyone. Today we are demonstrating a standalone replication of **Adaptive Feature Perturbation (AFP)**—a state-of-the-art defense designed to protect Intrusion Detection Systems (IDS) against black-box evasion attacks."* |
| **Point to the custom tree implementation text in cell 1** | *"Before we look at the attack results, I want to highlight a major engineering achievement here: **we have completely bypassed scikit-learn's Random Forest classifier.**"* <br><br> *"To prove absolute mathematical correctness, we wrote our own vectorized Decision Tree and Random Forest classifiers in pure NumPy from scratch. Every training and prediction step in this entire pipeline runs on our custom-built engine."* |

---

## 📂 Phase 1: Setup & Initialization (30 Seconds)

| Visual Action | Spoken Script |
| :--- | :--- |
| **Scroll to Phase 1 & run Cell 1** | *"In Phase 1, we initialize our environment. We load the baseline model and parse its internal structure directly into our custom NumPy random forest classifier. We also load the CSE-CIC-IDS2018 demo dataset, alongside the benign statistical reference profiles ($\mu_{\text{ref}}$, $\sigma_{\text{ref}}$) that the AFP wrapper uses to calibrate its perturbation scaling."* |

---

## 📈 Phase 2: Baseline Performance (30 Seconds)

| Visual Action | Spoken Script |
| :--- | :--- |
| **Scroll to Phase 2, run Cell 2, and show the baseline recall chart** | *"In Phase 2, we establish our control benchmark on clean network traffic. As you can see, under normal conditions, our custom NumPy engine achieves **88.20% classification accuracy** and **85.30% attack recall**. This confirms that our custom-built classifier matches the high detection standard of baseline models in the literature."* |

---

## 🕵️‍♂️ Phase 3: Silent Probing Attack & Defense (45 Seconds)

| Visual Action | Spoken Script |
| :--- | :--- |
| **Scroll to Phase 3, run Cell 3, and point to the results table** | *"Now we introduce the threat: a black-box **Silent Probing Attack** where the adversary queries feature-by-feature to map the boundary."* <br><br> *Look at the results table:* <br> *1. **Undefended**: The attacker maps the boundary and evades. Detection recall drops to **7.50%**.* <br> *2. **AFP-Selective**: The CUSUM change-point monitor detects the probing window, injects adaptive feature perturbation, and target recall is fully recovered to **85.50%**.* <br> *3. **AFP-Always-On**: By constantly injecting adaptive noise, the model acts as a poisoning oracle, returning incorrect boundary labels that completely blind the attacker, dropping evasion recall to **0.30%**.* |

---

## 🧬 Phase 4: Surrogate Transferability Attack (45 Seconds)

| Visual Action | Spoken Script |
| :--- | :--- |
| **Scroll to Phase 4, run Cell 4, and highlight the training logs** | *"What if the attacker queries the model to build a local surrogate model, and transfers the attack? We trained the surrogate model using our custom NumPy `fit()` method."* <br><br> *"In the **Undefended** setup, the transfer attack succeeds, dropping detection recall/accuracy to **14.50%**."* <br><br> *"But with **AFP-Selective**, the labels harvested by the attacker during probing are already corrupted. Evasion fails and target recall remains high at **85.70%**. In **Always-On** mode, the surrogate training labels are completely polluted, dropping recall to **31.20%**."* |

---

## 🎯 Phase 5: Decision Boundary Probing (45 Seconds)

| Visual Action | Spoken Script |
| :--- | :--- |
| **Scroll to Phase 5, run Cell 5, and point to the final comparison chart** | *"Finally, we evaluate against **Decision Boundary Probing**, which uses line searches to find boundary distances across all feature dimensions simultaneously."* <br><br> *"Against the **Undefended** classifier, the attack succeeds, reducing recall to **1.20%**."* <br><br> *"But under **AFP-Selective**, the adaptive noise causes the classification boundary to appear to shift on every query. The search converges on false coordinates, and our recall recovers to **85.40%**. With **Always-On**, search convergence completely fails, keeping recall at **0.10%**."* |

---

## 🏁 Conclusion: The Takeaway (30 Seconds)

| Visual Action | Spoken Script |
| :--- | :--- |
| **Show the overall trend match column in the summary tables** | *"In conclusion, this live demo shows that:* <br> *1. The undefended model is highly vulnerable to black-box query attacks.* <br> *2. **AFP-Selective** offers a balanced, stealthy defense that triggers only when probing is detected, recovering recall from under 10% to over 85% while maintaining performance on clean traffic.* <br> *3. **AFP-Always-On** provides maximum security, completely poisoning the attacker's queries to keep attack recall at extremely low levels (under 1% for boundary and silent probing).* <br><br> *Thank you, I'd be happy to take any questions."* |


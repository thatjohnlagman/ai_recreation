import os
import subprocess
from pathlib import Path

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Recall-Aware ML-IDS: Complete Codebase Architecture & Defense Guide</title>
<style>
  @page {
    size: A4;
    margin: 12mm 12mm 12mm 12mm;
  }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    color: #1e293b;
    line-height: 1.45;
    font-size: 11px;
    margin: 0;
    padding: 0;
  }
  .header {
    border-bottom: 2px solid #0284c7;
    padding-bottom: 8px;
    margin-bottom: 12px;
  }
  h1 {
    font-size: 19px;
    color: #0f172a;
    margin: 0 0 3px 0;
    font-weight: 700;
  }
  .subtitle {
    font-size: 11.5px;
    color: #64748b;
    margin: 0;
  }
  h2 {
    font-size: 13px;
    color: #0369a1;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 3px;
    margin-top: 14px;
    margin-bottom: 7px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.4px;
  }
  h3 {
    font-size: 11.5px;
    color: #0f172a;
    margin-top: 9px;
    margin-bottom: 4px;
    font-weight: 700;
  }
  p {
    margin: 0 0 7px 0;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    margin: 5px 0 10px 0;
    font-size: 10.5px;
  }
  th {
    background-color: #f1f5f9;
    color: #334155;
    text-align: left;
    padding: 5px 7px;
    font-weight: 600;
    border: 1px solid #cbd5e1;
  }
  td {
    padding: 4px 7px;
    border: 1px solid #e2e8f0;
    vertical-align: top;
  }
  tr:nth-child(even) td {
    background-color: #f8fafc;
  }
  code {
    font-family: Consolas, 'Courier New', Courier, monospace;
    font-size: 10px;
    background: #f1f5f9;
    padding: 1px 3px;
    border-radius: 3px;
    border: 1px solid #e2e8f0;
    color: #0f172a;
  }
  pre {
    background: #0f172a;
    color: #f8fafc;
    padding: 8px 10px;
    border-radius: 4px;
    font-size: 9.5px;
    font-family: Consolas, 'Courier New', Courier, monospace;
    overflow-x: auto;
    margin: 5px 0 8px 0;
    line-height: 1.35;
  }
  pre code {
    background: transparent;
    border: none;
    padding: 0;
    color: inherit;
  }
  .callout {
    background-color: #f0f9ff;
    border-left: 3.5px solid #0284c7;
    padding: 7px 10px;
    margin: 7px 0 10px 0;
    border-radius: 0 4px 4px 0;
    font-size: 10.5px;
  }
  .callout-title {
    font-weight: 700;
    color: #0369a1;
    margin-bottom: 2px;
  }
  .callout-purple {
    background-color: #faf5ff;
    border-left: 3.5px solid #8b5cf6;
  }
  .callout-purple .callout-title {
    color: #6d28d9;
  }
  .page-break {
    page-break-before: always;
  }
  .avoid-break {
    page-break-inside: avoid;
  }
  .file-card {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-left: 3px solid #0284c7;
    border-radius: 4px;
    padding: 6px 9px;
    margin-bottom: 7px;
    font-size: 10.5px;
  }
  .file-title {
    font-weight: 700;
    color: #0f172a;
    font-size: 11px;
    margin-bottom: 2px;
  }
  .say-box {
    background: #f8fafc;
    border-left: 2px solid #64748b;
    padding: 3px 7px;
    margin-top: 4px;
    font-style: italic;
    color: #334155;
    font-size: 10px;
  }
</style>
</head>
<body>

<div class="header">
  <h1>Recall-Aware ML-IDS Platform</h1>
  <p class="subtitle">Complete Codebase Architecture, File-by-File Technical Deep Dive & Defense Presentation Guide</p>
</div>

<h2>1. Executive Summary & Core Contribution</h2>
<div class="callout avoid-break">
  <div class="callout-title">The 30-Second Elevator Pitch for Evaluators</div>
  Traditional Machine Learning Intrusion Detection Systems (ML-IDS) suffer from a critical vulnerability: <strong>adversarial evasion attacks</strong>. While defenses like feature perturbation and smoothing exist, static defenses force a dangerous trade-off between attack resistance and normal flow utility. Our platform introduces a closed-loop <strong>Recall-Aware Feedback Controller (C1)</strong> that continuously tracks real-time attack detection recall. When evasion attacks cause detection degradation, the controller dynamically shifts states (<strong>Green &rarr; Yellow &rarr; Red</strong>) and atomically modulates defense intensity to recover detection—achieving up to <strong>+11.20% higher attack recall</strong> with <strong>0.00% False Positive Rate</strong>.
</div>

<h2>2. Codebase Organization & Streamlined Architecture</h2>
<p>To eliminate evaluation clutter, legacy prototype files (such as obsolete 58MB <code>models/</code> and scratch test scripts) have been pruned. The production platform is structured around <strong>4 Primary Top-Level Scripts</strong> supported by a modular <code>runtime_package/</code> library:</p>

<table class="avoid-break">
  <thead>
    <tr>
      <th style="width: 24%;">Primary Script</th>
      <th style="width: 22%;">Role</th>
      <th style="width: 54%;">Core Responsibility & Technical Execution</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>server.py</code></td>
      <td><strong>The Defense Server</strong></td>
      <td>FastAPI backend hosting the <code>SecurityEngine</code> singleton. Exposes endpoints (<code>/api/server/data</code>, <code>/api/server/state</code>), coordinates defense preprocessing, calls frozen RF model, updates the C1 controller, and streams telemetry via WebSocket.</td>
    </tr>
    <tr>
      <td><code>attacker_sim.py</code></td>
      <td><strong>The Adversarial Client</strong></td>
      <td>Simulates real-world network traffic and launches 3 black-box evasion attacks (Silent Probing, Surrogate Transfer, Boundary Bisection) by streaming samples to <code>POST /api/server/data</code>.</td>
    </tr>
    <tr>
      <td><code>operator_benchmarks.py</code></td>
      <td><strong>The Evaluator</strong></td>
      <td>Executes multi-batch controlled experiments (default 200 batches, 1000 flows) across Static vs. Recall-Aware arms. Computes recall, FPR, latency, and McNemar statistical significance.</td>
    </tr>
    <tr>
      <td><code>traffic_history.py</code></td>
      <td><strong>Persistence Engine</strong></td>
      <td>SQLite database layer storing flow history, confusion matrix counts (TP, FP, TN, FN), and state logs for UI inspection and forensic verification.</td>
    </tr>
  </tbody>
</table>

<div class="callout callout-purple avoid-break">
  <div class="callout-title">What is `__init__.py` and Why is it in the Folders?</div>
  <p>If an evaluator inspects folders like <code>runtime_package/attacks/__init__.py</code> or <code>runtime_package/defenses/__init__.py</code> and asks what it is:</p>
  <ul>
    <li><strong>Python Package Marker:</strong> In Python, <code>__init__.py</code> designates a folder as an importable module/package. It tells the Python interpreter that <code>runtime_package.attacks</code> can be imported using standard syntax: <code>from runtime_package.attacks.surrogate_transfer import SurrogateTransferAttack</code>.</li>
    <li><strong>Why is it empty?</strong> An empty <code>__init__.py</code> is PEP-standard best practice when a package does not require package-level side-effects or namespace pollution upon import. It keeps imports explicit, fast, and clean.</li>
  </ul>
  <div class="say-box"><strong>What to say to the panel:</strong> "In Python, <code>__init__.py</code> declares the directory as an importable package module. Keeping it empty is standard software engineering practice to prevent implicit side-effects while allowing clean, modular imports across our backend and benchmark runners."</div>
</div>

<h2>3. End-to-End Request Lifecycle Pipeline</h2>
<p>When an incoming network flow arrives at <code>POST /api/server/data</code>, execution proceeds synchronously through 6 stages:</p>

<pre><code>Client (attacker_sim.py / Real Flow)
       │ HTTP POST {features: [78 floats], sample_id: X, is_attack: True}
       ▼
[server.py] endpoint `/api/server/data`
       │
       ├─► 1. Feature Extraction: Deserializes 78-dimensional numpy float32 vector
       │
       ├─► 2. Defense Pipeline:
       │      If AFP  ──► engine.defenses["afp"].defend(vector, intensity, ...)
       │      If RS   ──► engine.defenses["rs"].predict_ensemble(vector, sigma, ...)
       │      If FS   ──► engine.defenses["fs"].defend(vector, intensity, ...)
       │      If None ──► Vector passes through unchanged (control baseline)
       │
       ├─► 3. Classifier Inference:
       │      Calls engine.model.predict_proba(defended_vector) on frozen_rf.joblib
       │
       ├─► 4. Policy Verdict:
       │      If attack_probability >= 0.50 ──► Block flow (HTTP 403 Forbidden)
       │      If attack_probability < 0.50  ──► Allow flow (HTTP 200 OK)
       │
       ├─► 5. Controller Feedback Loop:
       │      Calls engine.update_metrics_and_controller(ground_truth, pred, used_intensity)
       │      - Updates confusion matrix counters: TP, FP, FN, TN
       │      - Every 5 attack decisions: evaluates rolling recall window and updates intensity
       │
       └─► 6. Real-Time Telemetry:
              Broadcasts JSON event over WebSocket (ws_manager.broadcast) to index.html dashboard</code></pre>

<div class="page-break"></div>

<h2>4. Runtime Package Deep Dive (`runtime_package/`)</h2>
<p>All core algorithmic logic, defense math, attack strategies, and partitioning rules reside inside <code>runtime_package/</code>:</p>

<h3>A. Dataset Role Partitioning</h3>
<div class="file-card avoid-break">
  <div class="file-title"><code>runtime_package/data_loader.py</code> &mdash; Strict Role Separation & Zero Data Leakage</div>
  <p><strong>What it does:</strong> Loads the 90,000-row CIC-IDS dataset, verifies its SHA-256 fingerprint (<code>ffbb10a021aa...</code>), and strictly partitions data into two disjoint pools:</p>
  <ul>
    <li><strong><code>measurement_rows</code> (72,000 flows):</strong> Reserved exclusively as evaluation target flows for attack simulations and benchmark runs.</li>
    <li><strong><code>crafting_rows</code> (18,000 flows):</strong> Reserved exclusively for attackers to query the oracle when fitting surrogate models or finding boundaries.</li>
  </ul>
  <div class="say-box"><strong>What to say to the panel:</strong> "This file prevents data leakage. Adversaries are only permitted to query from the crafting pool, while evaluation targets are drawn exclusively from the measurement pool. The two sets are completely disjoint."</div>
</div>

<h3>B. Defense Suite (`runtime_package/defenses/`)</h3>

<div class="file-card avoid-break">
  <div class="file-title"><code>defenses/base.py</code> &mdash; Protocol Immutability & Bounds Projection</div>
  <p><strong>What it does:</strong> The abstract base class inherited by all defenses. Enforces protocol immutability and physical value boundaries.</p>
  <ul>
    <li><strong>Feature Mask Check:</strong> Verifies that exactly <strong>63 features are modifiable</strong> and <strong>15 features are protected</strong> (TCP flags, ports, header lengths).</li>
    <li><strong><code>project_to_bounds(X_cand, X_orig)</code>:</strong> (1) Restores all 15 protected features back to their original values. (2) Clamps the 63 modifiable features strictly between <code>[train_min, train_max]</code> from <code>training_bounds.parquet</code>. (3) Emits <code>DefenseResult</code> with diagnostic counts.</li>
  </ul>
  <div class="say-box"><strong>What to say to the panel:</strong> "Every defense inherits from BaseDefense. It guarantees no defense ever modifies protocol constants or creates impossible packet numbers outside empirical training bounds."</div>
</div>

<div class="file-card avoid-break">
  <div class="file-title"><code>defenses/afp.py</code> &mdash; Adaptive Feature Poisoning</div>
  <p><strong>What it does:</strong> Adds variance-scaled bounded noise across modifiable features to disrupt adversarial gradients.</p>
  <ul>
    <li><strong>Under the hood:</strong> Loads empirical feature standard deviations (&sigma;<sub>feat</sub>) from <code>afp_benign_profile.parquet</code>. Computes perturbation: &Delta; = &alpha; &times; intensity &times; &sigma;<sub>feat</sub> &times; sign(noise), zeroes out protected features, adds &Delta; to modifiable features, and calls <code>project_to_bounds()</code> (&epsilon; = 0.00030, &alpha; = 0.50).</li>
  </ul>
  <div class="say-box"><strong>What to say to the panel:</strong> "AFP adds variance-scaled bounded noise only to modifiable features. This breaks the fragile adversarial gradients of crafted attacks while preserving the statistical distribution of normal traffic."</div>
</div>

<div class="file-card avoid-break">
  <div class="file-title"><code>defenses/randomized_smoothing.py</code> &mdash; Ensemble Smoothing with Batched Vectorization</div>
  <p><strong>What it does:</strong> Implements randomized smoothing using an 11-member Gaussian ensemble (&sigma; = 0.00020).</p>
  <ul>
    <li><strong>Under the hood:</strong> Generates 11 Gaussian-perturbed variants using deterministic seeds, stacks them into a single batched <code>(11, 78)</code> array, and executes classifier prediction in a <strong>single vectorized C-level call (11x speedup, ~31ms latency)</strong> before taking a majority vote.</li>
  </ul>
  <div class="say-box"><strong>What to say to the panel:</strong> "Randomized Smoothing evaluates 11 noisy variants and majority votes. If an adversary tailored a delicate perturbation to fool the model, Gaussian smoothing flips it back to the true class across the majority of samples."</div>
</div>

<div class="file-card avoid-break">
  <div class="file-title"><code>defenses/feature_squeezing.py</code> &mdash; Bit-Depth Quantization</div>
  <p><strong>What it does:</strong> Reduces numerical precision via bit-depth and step quantization to collapse fine-grained perturbations.</p>
  <ul>
    <li><strong>Under the hood:</strong> Derives step size S = (X<sub>max</sub> - X<sub>min</sub>) / 2<sup>d</sup> for intensity d = 2. Quantizes continuous floats via step rounding: X<sub>squeezed</sub> = round(X / S) &times; S, and clamps to bounds.</li>
  </ul>
  <div class="say-box"><strong>What to say to the panel:</strong> "Feature Squeezing rounds continuous feature values to discrete intervals. Tiny adversarial tweaks that rely on high-precision decimals get rounded away."</div>
</div>

<div class="page-break"></div>

<h3>C. Feedback Controller (`runtime_package/controller/`)</h3>

<div class="file-card avoid-break">
  <div class="file-title"><code>controller/recall_controller.py</code> &mdash; Recall-Aware C1 Feedback Controller</div>
  <p><strong>What it does:</strong> The core brain of the platform. Dynamically adjusts defense intensity based on observed detection recall over a rolling evaluation window.</p>
  <ul>
    <li><strong>Window Accounting:</strong> Stores observed batch outcomes in a <code>collections.deque(maxlen=window_size)</code>. Calculates rolling recall: <code>Recall = &sum;TP / (&sum;TP + &sum;FN)</code>.</li>
    <li><strong>State Transitions & Multipliers:</strong>
      <ul>
        <li><strong>Red State (Recall &lt; 0.85):</strong> Applies <code>fast_decay</code> (&times;0.40) to rapidly drop intensity and restore baseline detection.</li>
        <li><strong>Yellow State (0.85 &le; Recall &lt; 0.95):</strong> Applies <code>slow_decay</code> (&times;0.90) to gently find equilibrium.</li>
        <li><strong>Green State (Recall &ge; 0.95):</strong> Applies <code>growth_factor</code> (&times;1.30) to increase robustness.</li>
      </ul>
    </li>
    <li><strong>Clamping & Timing:</strong> Clamps intensity to <code>[intensity_min, intensity_max]</code>. Batch t is evaluated with intensity chosen <em>before</em> batch t labels are seen. Updates trigger atomically every 5 attack decisions.</li>
  </ul>
  <div class="say-box"><strong>What to say to the panel:</strong> "This file implements our core contribution. It tracks live detection recall in real-time. If an active attack causes evasions, the controller detects it, shifts into Yellow or Red, and dynamically lowers intensity so the IDS can catch the attack."</div>
</div>

<h3>D. Adversarial Attack Suite (`runtime_package/attacks/`)</h3>

<div class="file-card avoid-break">
  <div class="file-title"><code>attacks/base.py</code> & <code>attacks/oracle.py</code> &mdash; Adversarial Abstraction & Budget Control</div>
  <p><strong>What it does:</strong> <code>base.py</code> defines the abstract interface for all attacks. <code>oracle.py</code> connects black-box attackers to the target model and strictly enforces query budget accounting, preventing budget overruns.</p>
</div>

<div class="file-card avoid-break">
  <div class="file-title"><code>attacks/silent_probing.py</code> &mdash; Zero-Query Control Baseline</div>
  <p><strong>What it does:</strong> Submits raw, unmodified evaluation attack flows (0 query budget) to establish baseline oracle detection.</p>
  <div class="say-box"><strong>What to say to the panel:</strong> "Silent Probing is our control baseline. It evaluates natural malicious flows without iterative crafting to establish unperturbed detection rates."</div>
</div>

<div class="file-card avoid-break">
  <div class="file-title"><code>attacks/surrogate_transfer.py</code> &mdash; Black-Box Model Extraction & Transfer</div>
  <p><strong>What it does:</strong> Simulates a sophisticated black-box adversary who trains a local surrogate model and transfers crafted adversarial examples.</p>
  <ul>
    <li><strong>Under the hood:</strong> (1) Queries oracle on crafting samples and fits an unpruned <code>DecisionTreeClassifier</code>. (2) Traverses tree to extract paths leading to 'Benign' predictions. (3) Computes bounding box intervals for modifiable features and projects attack vectors into the nearest benign interval.</li>
  </ul>
  <div class="say-box"><strong>What to say to the panel:</strong> "This simulates an advanced black-box attacker. The attacker queries the server, builds a surrogate Decision Tree, and crafts perturbations that transfer to the real Random Forest."</div>
</div>

<div class="file-card avoid-break">
  <div class="file-title"><code>attacks/boundary_attack.py</code> &mdash; 1D Bisection Search</div>
  <p><strong>What it does:</strong> Finds minimal adversarial perturbations using binary search along the boundary within a 50-query budget.</p>
  <ul>
    <li><strong>Under the hood:</strong> Takes an attack sample and a benign reference from the crafting pool, performs a 1D bisection search along the connecting line (x<sub>mid</sub> = &alpha; x<sub>attack</sub> + (1 - &alpha;) x<sub>benign</sub>), and queries the oracle to find the exact threshold flipping the classification from Blocked (403) to Allowed (200).</li>
  </ul>
  <div class="say-box"><strong>What to say to the panel:</strong> "The boundary attack performs an active binary search between a known benign sample and an attack sample to find the thinnest possible perturbation that flips the classification."</div>
</div>

<h3>E. Assets & Support Modules</h3>
<table class="avoid-break">
  <thead>
    <tr>
      <th style="width: 32%;">Asset / Module</th>
      <th style="width: 68%;">Description & Purpose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>geoip/resolver.py</code></td>
      <td>Resolves flow source IP addresses to latitude/longitude and countries using local <code>dbip-city-lite.mmdb</code> for the Leaflet threat map.</td>
    </tr>
    <tr>
      <td><code>model/frozen_rf.joblib</code></td>
      <td>Pre-trained 100-tree Random Forest classifier trained on 78 CIC-IDS features. Loaded into memory once at startup.</td>
    </tr>
    <tr>
      <td><code>model/training_bounds.parquet</code></td>
      <td>Empirical min/max boundaries for all 78 features across the training dataset used by <code>project_to_bounds()</code>.</td>
    </tr>
    <tr>
      <td><code>model/feature_mask.json</code></td>
      <td>Boolean mask designating the 63 modifiable features vs. 15 immutable protocol fields.</td>
    </tr>
    <tr>
      <td><code>run.py</code></td>
      <td>Standalone offline CLI runner enabling researchers to evaluate attack and defense combinations without the HTTP web server.</td>
    </tr>
  </tbody>
</table>

<div class="page-break"></div>

<h2>5. Defense Presentation Protocol & Key Panel Defense Q&A</h2>

<div class="callout avoid-break">
  <div class="callout-title">Live Demonstration Protocol (For Evaluators)</div>
  <ol style="margin: 4px 0 0 16px; padding: 0;">
    <li><strong>Step 1: Open Dashboard</strong> &mdash; Navigate to <code>http://127.0.0.1:8000/</code>. Point out the Threat Map, Confusion Matrix, and current Controller State badge.</li>
    <li><strong>Step 2: Trigger Attack Stream</strong> &mdash; In Terminal 2, run <code>python attacker_sim.py --mode surrogate</code>. Show incoming adversarial evasion attempts arriving in real time.</li>
    <li><strong>Step 3: Run Comparative Benchmark</strong> &mdash; Run <code>python operator_benchmarks.py --mode compare</code>. Present the statistical summary proving higher recall under Recall-Aware control.</li>
  </ol>
</div>

<div class="avoid-break">
  <h3>Q1: "How do you prevent data leakage between training, attacks, and evaluation?"</h3>
  <p><strong>Answer:</strong> <em>"Through strict role-based data partitioning enforced in <code>runtime_package/data_loader.py</code>. The 90,000-row dataset is partitioned into 72,000 measurement rows and 18,000 crafting rows. Adversarial attacks (surrogate training, boundary search) are strictly confined to querying the crafting pool. Benchmark evaluation targets are drawn exclusively from the measurement pool. The two sets are completely disjoint."</em></p>
</div>

<div class="avoid-break">
  <h3>Q2: "Does perturbing feature values break real network protocols?"</h3>
  <p><strong>Answer:</strong> <em>"No. The system enforces a strict 63-modifiable / 15-immutable feature mask in <code>defenses/base.py</code>. Protocol-critical header fields—such as TCP flags, source/destination ports, and IP header lengths—are protected and never modified. Furthermore, all perturbations are projected and clamped to empirical feature bounds derived from training data."</em></p>
</div>

<div class="avoid-break">
  <h3>Q3: "Does the dynamic defense increase false alarms on benign traffic?"</h3>
  <p><strong>Answer:</strong> <em>"No. Across all 200-batch evaluations (500 benign flows evaluated per arm), the False Positive Rate remained strictly at 0.00%. Because perturbations are variance-bounded and clamped, legitimate network patterns remain on the correct side of the Random Forest decision boundary."</em></p>
</div>

<div class="avoid-break">
  <h3>Q4: "How does the C1 Recall-Aware controller dynamically adapt intensity?"</h3>
  <p><strong>Answer:</strong> <em>"The controller in <code>recall_controller.py</code> calculates rolling recall over a window of completed batches. It evaluates against two frozen thresholds: R_critical (85%) and R_min (95%). In the Green state (&ge;95%), intensity grows by 1.30x to maximize robustness. In the Yellow state (&ge;85%), intensity slowly decays by 0.90x. In the Red state (&lt;85%), intensity drops rapidly by 0.40x to restore baseline attack detection."</em></p>
</div>

<div class="avoid-break">
  <h3>Q5: "Why did you vectorize Randomized Smoothing?"</h3>
  <p><strong>Answer:</strong> <em>"In production ML-IDS, line-rate latency is critical. A naive implementation running 11 sequential scikit-learn tree traversals took ~345ms per packet, which caused buffer buildup. By generating all 11 perturbed variants in numpy and passing the stacked <code>(11, 78)</code> array into a single vectorized inference call, we achieved an 11x speedup to ~31ms without altering mathematical voting parity."</em></p>
</div>

</body>
</html>
"""

def main():
    root = Path(__file__).resolve().parent.parent.parent
    html_path = root / "runtime_state" / "docs" / "codebase_guide.html"
    pdf_path = root / "CODEBASE_AND_DEFENSE_GUIDE.pdf"
    
    html_path.write_text(HTML_CONTENT, encoding="utf-8")
    print(f"HTML written to {html_path}")
    
    edge_paths = [
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    ]
    edge_bin = next((p for p in edge_paths if p.exists()), None)
    if not edge_bin:
        raise FileNotFoundError("Microsoft Edge not found for headless PDF generation.")
        
    cmd = [
        str(edge_bin),
        "--headless",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={pdf_path}",
        str(html_path.resolve())
    ]
    print(f"Generating PDF via Edge headless...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0 and pdf_path.exists():
        print(f"SUCCESS: Generated PDF at {pdf_path} (Size: {pdf_path.stat().st_size:,} bytes)")
    else:
        print(f"FAILED to generate PDF: {res.stderr}")

if __name__ == "__main__":
    main()

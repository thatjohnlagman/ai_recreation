# Agent task: rebuild the IDS thesis frontend as a live network monitoring console

## 0. Read before you write

Do **not** generate code in your first response.

First, inspect the existing repository and report back:

1. Backend framework, entrypoint, and how the trained model is loaded and served.
2. The exact model input contract — the ordered feature list, dtypes, and any scaler/encoder artifacts applied before inference.
3. Where the AFP (Adversarial Feature Perturbation) defense is currently implemented, and how the recall-aware controller adjusts its perturbation budget.
4. The current frontend framework, routing, and state management.
5. Where the 20,000-record dataset slice lives and what columns it has.

Then state, in one short list, which parts of this spec conflict with what already exists. Wait for confirmation before implementing.

---

## 1. Context

This is the application layer for an undergraduate thesis:

> *Performance of Recall-Aware Control for Perturbation Defenses in Intrusion Detection Systems Against Black-Box Probing Attacks*

There is an existing, working backend and a trained binary classifier (Attack / Benign) built on **CSE-CIC-IDS2018**. The classifier and the AFP defense are **not** to be retrained or redesigned. This task is about the application surrounding them.

The current frontend presents the thesis as a narrative: five sections walking through preprocessing, baseline classification, measured performance, an attack/defense demo, and the experimental steps of the paper. It reads as a slideshow of the research.

**Replace it with an operational security console.** The user is a network defender watching traffic arrive in real time. The thesis result becomes a *report the console produces*, not the structure of the app.

---

## 2. Non-negotiable constraints

These exist because a thesis panel will attack the weak versions. Violating any of them makes the application indefensible.

**C1 — Attack technique must be inferred, never echoed.**
The simulator knows which technique it applied. The application must **not** read that label to populate the UI. A separate attribution engine has to infer the technique from observable session behaviour, and it must be able to be wrong. The simulator's technique label is used **only** for scoring attribution accuracy offline. If you find yourself passing `campaign.technique` into a display field, you have broken the requirement.

**C2 — Inference latency must be measured, not synthesized.**
Wall-clock timing on the server, around the actual model call, recorded separately for the AFP-enabled and AFP-disabled paths. No `Math.random()`, no hardcoded ranges, no client-side estimates.

**C3 — Emitted traffic must not be 50/50.**
The 20k pool is balanced (10k attack / 10k benign) for offline evaluation. The **live stream must run at a realistic base rate** — benign-dominant baseline (default 3% attack, configurable) with attacks arriving in bursts. A console where half the traffic is malicious is not a network, and it makes precision meaningless.

**C4 — Perturbed flows must stay physically realizable.**
After any adversarial transform, enforce: durations and counts non-negative; packet counts integral; `Flow Bytes/s` and `Flow Packets/s` consistent with duration and totals; `Down/Up Ratio` consistent with its components; flag counts integral and bounded by packet counts. Clamp or reject, and log rejections.

**C5 — One user-facing experimental control only: the AFP toggle.**
Pause and playback speed are permitted as *viewing* controls. Nothing else the user touches may change model, defense, or simulation behaviour. No threshold sliders, no epsilon controls, no technique pickers in the operator UI.

---

## 3. Architecture

```
ADVERSARY SIMULATION                      DEFENDER PIPELINE
  Record pool (20k)                         AFP layer (recall-aware control)
        |                                          |
  Campaign engine (assigns technique)        IDS model (attack / benign)
        |                                          |
  Probe transform (perturbs features)  ----> Attribution engine (infers technique)
        ^                                          |
        |______ returned label _____________|      v
                                             Console event
```

The dashed feedback path is load-bearing: the Decision Boundary Attack computes its next query from the model's previous returned label. The simulator therefore **cannot** be a pre-generated CSV — it must run in a live loop with the classifier.

---

## 4. Simulation engine

### 4.1 Record pool
Load the 20k records once into memory as a pool indexed by ground-truth label. Records are *drawn* from the pool, not played back in order. Track consumption so a full run eventually exercises the whole pool.

### 4.2 Arrivals
Poisson process, not fixed-interval ticks. Base rate λ modulated by a slow diurnal curve. Benign flows arrive continuously.

### 4.3 Campaigns
Attacks arrive as coherent campaigns, never as isolated i.i.d. rows. Attribution is session-level, so without a stable source identity there is nothing to attribute.

```
Campaign {
  campaign_id: string
  source_ip: string          # synthetic but stable across the campaign
  technique: "none" | "silent_probing" | "decision_boundary" | "surrogate_transfer"
  target_service: string
  n_records: int
  cadence_ms: {mean, variance}
  started_at: timestamp
  state: "active" | "completed"
}
```

Campaigns overlap. At least one long-running low-rate campaign should be able to coexist with short bursty ones — this is what makes the console interesting and what stresses attribution.

### 4.4 Technique transforms

Each is a function `(feature_vector, campaign_state, last_returned_label) -> perturbed_vector`.

| Technique | Query pattern | Perturbation |
|---|---|---|
| `none` | Normal cadence | Identity. Unmodified dataset row. This is the control arm. |
| `silent_probing` | Very low volume, long and high-variance inter-arrival gaps | Small-magnitude noise applied only to unconstrained features. Few records per campaign. |
| `decision_boundary` | High volume, tight cadence, closed-loop | Step size decays each query as a function of the previous returned label; trajectory converges toward model score ≈ 0.5. |
| `surrogate_transfer` | One burst, few queries, no feedback dependence | Perturbations precomputed offline against a locally trained surrogate. Larger norm, high mutual diversity, no interactive probing. |

Implement `surrogate_transfer` by training a small surrogate model offline (on a held-out slice, never on the test pool) and generating crafted samples ahead of the run. Document that the surrogate is trained without access to the target's parameters.

Apply C4 validity clamping after every transform.

---

## 5. Attribution engine

A **separate module** from the IDS classifier. It consumes a sliding window of queries grouped by `source_ip` and outputs a technique label plus confidence.

### 5.1 Session features
Compute over the window:

- `query_count`
- `iat_mean`, `iat_variance` — inter-arrival time
- `consecutive_l2_mean` — mean L2 distance between consecutive query feature vectors
- `consecutive_l2_trend` — slope of that distance over the window; shrinking is the boundary-search signature
- `boundary_proximity` — mean `|score − 0.5|`
- `manifold_distance` — Mahalanobis distance to the nearest benign cluster centroid (fit centroids once, offline, on benign training data; freeze them)
- `feedback_dependence` — correlation between the previous returned label and the direction of the next step

### 5.2 Classifier
A shallow decision tree or an explicit rule set over those seven features. **Explainability is a requirement**, not a preference — the UI must be able to show which evidence fired. Do not use an opaque model here.

### 5.3 Fifth state
Output `unattributed` when evidence is insufficient. Early in a campaign, low confidence is the correct answer. Confidence should climb as the window fills. Expose confidence as a number; the UI renders it.

### 5.4 Scoring
Persist the simulator's true technique alongside the attribution output **in the analytics store only** (never in the live event payload sent to the operator UI). Use it to compute an attribution confusion matrix for the report screen.

---

## 6. Defense state

Four outcomes, derived from ground truth and prediction. Map directly onto a confusion matrix.

| Ground truth | Prediction | State |
|---|---|---|
| Attack | Attack | `blocked` |
| Attack | Benign | `bypassed` |
| Benign | Benign | `allowed` |
| Benign | Attack | `false_positive` |

`false_positive` is the cost of AFP and the reason recall-aware control exists. Never hide or downweight it in the UI.

---

## 7. Shadow inference

Run **both** the AFP-enabled and AFP-disabled paths on every record. Display whichever the toggle selects; persist both.

This gives a genuine paired ON/OFF comparison on identical traffic without the user re-running anything, and makes paired statistical testing (e.g. McNemar) available for the report. Time each path independently — the overhead figure must remain real. If shadow inference measurably degrades throughput, make it a server-side config flag defaulting to on, not a user control.

---

## 8. Data contracts

### 8.1 Console event (streamed to the operator UI)

```json
{
  "event_id": "uuid",
  "timestamp": "iso8601",
  "source_ip": "string",
  "campaign_id": "string | null",
  "target_service": "string",
  "ground_truth": "attack | benign",
  "prediction": "attack | benign",
  "score": 0.0,
  "defense_state": "blocked | bypassed | allowed | false_positive",
  "afp_active": true,
  "inference_latency_ms": 0.0,
  "attribution": {
    "technique": "none | silent_probing | decision_boundary | surrogate_transfer | unattributed",
    "confidence": 0.0,
    "evidence": ["string"]
  },
  "features": { "<feature_name>": 0.0 },
  "perturbation_delta": { "<feature_name>": 0.0 }
}
```

`perturbation_delta` is null when AFP is off. **`true_technique` must not appear in this payload.**

### 8.2 Controller state (streamed alongside events)

```json
{
  "timestamp": "iso8601",
  "epsilon": 0.0,
  "rolling_recall": 0.0,
  "recall_floor": 0.0,
  "rolling_fpr": 0.0,
  "adjustment": "increase | decrease | hold"
}
```

### 8.3 Transport
WebSocket (or SSE) for the live stream. REST for report/analytics queries. Backpressure: if the client falls behind, drop intermediate events rather than queueing unboundedly — but never drop controller-state updates.

---

## 9. Screens

### 9.1 Operations console — the default view

- **Header**: AFP toggle, defense posture indicator, pause, playback speed. Nothing else.
- **Center**: live event stream, newest first. Columns: time, source, ground truth, prediction, attributed technique + confidence, defense state, latency. Colour-code defense state; `bypassed` and `false_positive` must be immediately visible.
- **Left rail**: live sparklines for 8–12 features only. Suggested: `Flow Duration`, `Flow Bytes/s`, `Flow Packets/s`, `Fwd Packet Length Mean`, `Bwd Packet Length Mean`, `Flow IAT Mean`, `SYN Flag Count`, `Init Fwd Win Bytes`, `Packet Length Std`, `Down/Up Ratio`. Do not attempt to render all ~80.
- **Right rail — the recall-aware controller.** Current perturbation budget ε, rolling recall, the recall floor, rolling FPR, and a short trace of ε adjusting itself over time. Read-only. This panel is the thesis title made visible and should be given real visual weight.

### 9.2 Event drill-down
Click a row. Show the full feature vector, model score, attribution evidence list, and a **pre/post perturbation delta view** — side-by-side feature values showing exactly what AFP changed. This is the mechanism made legible; make it the centrepiece of the drill-down.

### 9.3 Campaign view
Grouped by source. Per campaign: attributed technique and confidence over time, probes spent, bypass rate so far, first-seen and last-seen. Show attribution confidence *rising* as evidence accumulates.

### 9.4 Defense report
Rolling recall, FPR, bypass rate, latency p50/p95/p99, attribution confusion matrix, and the paired ON/OFF comparison **broken down per technique**. The `none` control arm gets its own row. This is where the old narrative sections belong, reframed as output.

---

## 10. Explicitly do not

- Do not retrain, fine-tune, or alter the IDS classifier or the AFP defense.
- Do not surface the simulator's true technique label in any operator-facing view or payload.
- Do not fabricate latency, confidence, or metric values for demo purposes anywhere in the codebase, including placeholder or loading states.
- Do not add user controls beyond the AFP toggle and viewing controls.
- Do not render all dataset features in the live monitor.
- Do not keep any of the five narrative sections as top-level navigation.
- Do not use an opaque model for attribution.

---

## 11. Build order

1. **Contract report** (section 0). Stop and confirm.
2. Simulation engine: pool, Poisson arrivals, campaign scheduler, `none` transform only. Verify base rate and validity clamping.
3. Live event stream end to end with a bare table. Real measured latency from day one.
4. Remaining three technique transforms, including the closed-loop feedback path for `decision_boundary`.
5. Attribution engine + offline scoring harness. Report attribution accuracy before building its UI.
6. Shadow inference and the persistence layer.
7. Operations console UI: stream, feature rail, controller rail.
8. Drill-down and campaign views.
9. Defense report.

Ship each step working before starting the next.

---

## 12. Acceptance criteria

- With AFP off and `decision_boundary` active, bypass rate is measurably higher than with AFP on, on identical traffic (via shadow inference).
- Attribution accuracy is reported with a confusion matrix and is **not** 100%. A perfect score means the true label is leaking; find the leak.
- Attribution confidence for a fresh campaign starts at `unattributed` and rises with query count.
- Measured latency differs between the AFP and non-AFP paths, and the difference is stable across runs.
- Live attack base rate matches the configured value (default 3%), not the pool's 50%.
- No perturbed flow in a 10-minute run violates the C4 validity constraints.
- Grepping the operator-facing code for the simulator's technique field returns nothing.

---

## 13. Stack notes

- Backend: `<<fill in — e.g. FastAPI, existing>>`
- Frontend: `<<fill in — e.g. React + Vite, existing>>`
- Model artifact: `<<fill in — path and format>>`
- Dataset slice: `<<fill in — path>>`

Match existing conventions in the repository. Do not introduce a new framework, state library, or styling system without asking first.

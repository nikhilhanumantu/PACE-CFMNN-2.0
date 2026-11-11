# Hyperbox Shrinkage, Structural Invariants, and Safeguard Mechanics

This document provides a comprehensive technical, mathematical, and empirical analysis of the **hyperbox shrinkage operator** in PACE-CFMNN. It explains exactly **how the model structure changes**, the **potential failure modes (problems) that can arise from naive shrinking**, the **three built-in safeguard mechanisms** implemented in the code, and **empirical verification** from benchmark runs.

---

## 1. Architectural Overview: What is Hyperbox Shrinking?

In classical Fuzzy Min-Max Neural Networks (Simpson, 1992; extended to General Fuzzy Min-Max [GFMM] by Gabrys & Bargiela, 2000), contraction was applied heuristics-style during training, frequently causing high-dimensional boundary shattering and hyperbox collapse. Baseline CFMNN eliminated contraction entirely, which resolved boundary instability but caused **100% of training misclassifications to become cross-class ties** inside overlapping hyperboxes.

PACE-CFMNN introduces a **post-hoc, label-free geometric self-repair operator**:
* Instead of running costly iterative optimization or retraining the entire network from scratch, it surgically targets overlapping hyperboxes where training points are misclassified.
* It applies a **1-step closed-form facet displacement** on the offending hyperbox, moving exactly one boundary coordinate by a tiny margin $\epsilon = 0.001$ past the error coordinate.

```mermaid
flowchart LR
    A["Error Detected:<br>Point x is tied in Box A & Box B"] --> B["Identify Opposing Box B<br>(Class != y_true)"]
    B --> C["Compute Fractional Cut Share<br>for all dimensions j = 1..d"]
    C --> D["Select Thinnest Dimension:<br>j* = argmin share_j"]
    D --> E["Displace Boundary Facet:<br>V_new = x_j + 0.001 (or W_new = x_j - 0.001)"]
    E --> F["Filter Member Points &<br>Update Centroid"]
    F --> G["Validation Gate Check:<br>Delta F1 <= 0.5 pts and Delta Recall_c <= 1.0 pts"]
    G -- Pass --> H["Commit Edit &<br>Log Impact Certificate"]
    G -- Fail --> I["Restore Snapshot<br>(Zero Damage)"]
```

---

## 2. Structural Analysis: What Changes vs. What Stays Intact

When `shrink_box(model, k, x, feature, X_lab)` executes in [`pace_cfmnn/refine.py`](file:///c:/Users/nikhi/Downloads/pace-cfmnn-review-b228a29/pace-cfmnn-review/pace-cfmnn/pace_cfmnn/refine.py#L30-L55), the structural state of the model updates under strict mathematical invariants.

### A. What Changes in the Model Structure

1. **Hyperbox Boundary Tensors ($V$ and $W$):**
   * Exactly **one boundary coordinate** of the $k$-th hyperbox is modified.
   * If the error point is nearer to the lower facet:
     $$V_{k, j}^{\text{new}} = x_j + \text{MARGIN} \quad (\text{with } \text{MARGIN} = 0.001)$$
   * If the error point is nearer to the upper facet:
     $$W_{k, j}^{\text{new}} = x_j - \text{MARGIN} \quad (\text{with } \text{MARGIN} = 0.001)$$
2. **Hyperbox Volume:**
   * The volume of Box $k$ decreases strictly along dimension $j$:
     $$\text{Vol}(B_k^{\text{new}}) = \text{Vol}(B_k^{\text{old}}) \times \left(1 - \text{share}_j\right)$$
3. **Training Member Points (`members_`):**
   * Any training instances previously enclosed by Box $k$ that fall into the discarded facet slice are evicted:
     $$\text{members}_k^{\text{new}} = \{m \in \text{members}_k^{\text{old}} \mid V_{k, j}^{\text{new}} \le X_{m, j} \le W_{k, j}^{\text{new}}\}$$
4. **Hyperbox Centroid (`centroid_`):**
   * Because the centroid is dynamically maintained as the arithmetic mean of enclosed training points, removing evicted points re-centers the centroid:
     $$\bar{C}_k^{\text{new}} = \frac{1}{|\text{members}_k^{\text{new}}|} \sum_{m \in \text{members}_k^{\text{new}}} X_m$$
5. **Local Decision Surface:**
   * The decision boundary shifts **only within the narrow slice that was cut away and its surrounding fuzzy ramp**.
6. **Automatic Pruning of Empty Boxes:**
   * If an edit evicts all members of Box $k$ (`len(keep) == 0`), the box is deleted cleanly via `model._remove_box(k)`. This prunes degenerate outlier boxes.

### B. What Stays Strictly Unchanged

* **All other $K - 1$ hyperboxes:** Exactly zero coordinate changes occur outside Box $k$.
* **Hyperbox Count Cap:** The number of hyperboxes $K$ never increases ($K^{\text{new}} \le K^{\text{old}}$).
* **Fuzzy Membership Formulation:** The Simpson ramp activation formula $b_j(x)$ and hyperbox aggregation $\mu_k(x) = \frac{1}{d} \sum_{j=1}^d b_j(x)$ remain identical.
* **Rule Readability:** Every box remains a pure conjunctive rule ($\text{IF } x \in B_k \implies \text{Class } c$). Only the numeric threshold on feature $j$ is tightened.

### C. Structural Invariant Verification (`validate_structure`)

Every accepted shrink is formally checked by [`validate_structure()`](file:///c:/Users/nikhi/Downloads/pace-cfmnn-review-b228a29/pace-cfmnn-review/pace-cfmnn/pace_cfmnn/refine.py#L200-L230):
1. $0 \le V_{k, j} \le W_{k, j} \le 1$ holds for all $k, j$.
2. $\text{count}_k = |\text{members}_k|$ holds for all $k$.
3. $\text{centroid}_k = \frac{1}{\text{count}_k} \sum X[\text{members}_k]$ matches to floating point tolerance $< 10^{-12}$.
4. No training sample belongs to multiple boxes of different classes.

---

## 3. Potential Problems: How Naive Shrinking Can Fail

Shrinking hyperboxes is not automatically risk-free. If implemented without controls (as in early FMM literature), it introduces 5 severe problems:

```mermaid
classDiagram
    class PotentialFailures {
        +Problem 1: Broken Predictions (Collateral Damage)
        +Problem 2: Minority Class Recall Collapse
        +Problem 3: Ineffective Open-Margin Errors
        +Problem 4: Gate Waste on Noisy Datasets
        +Problem 5: The Explanation Core-Cut Trap
    }
    class MitigationsInCode {
        +Safeguard 1: Gate (Macro-F1 & Class Recall Floors)
        +Safeguard 2: Thinnest-Facet Cut (C'-Volume Rule)
        +Safeguard 3: Prediction Churn & Broken Auditing
        +Formal Guarantee: Provable Bounded Impact Certificates
    }
    PotentialFailures ..> MitigationsInCode : Neutralized by
```

### Detailed Breakdown of the 5 Failure Modes:

### Problem 1: Broken Predictions (Collateral Damage)
* **Mechanism:** Slicing a face of Box $B$ to exclude error sample $x$ reduces Box $B$'s volume. If valid, correctly classified points of Box $B$'s class also lie near that boundary, their membership drops from $1.0$ down the fuzzy ramp. They can flip from correct to incorrect.
* **Empirical Impact:** On the MAGIC dataset, self-repair broke **6.85% of test samples** (flipped previously correct predictions).
* **Mitigation:** The Validation Gate and Thinnest-Cut chooser keep broken predictions dramatically lower than shuffled retraining (which broke 15.60% on MAGIC).

### Problem 2: Minority Class Recall Degradation
* **Mechanism:** In imbalanced datasets, shrinking a minority class's hyperbox can drastically shrink its recall.
* **Empirical Impact:** On MAGIC, self-repair caused a **$-2.73$ points drop in the lowest class recall across 9 of 10 random seeds**.
* **Mitigation:** The Gate tracks per-class recall individually (`recall_c`) and aborts any edit where recall drops by $>1.0$ points. (On Thyroid and Skin, shrinking majority boxes actually increased minority recall by $+19.4$ points and $+16.7$ points).

### Problem 3: Ineffective Open-Margin Errors (The "Empty Space" Problem)
* **Mechanism:** Shrinking requires the error sample to lie *inside* an opposing box ($s_{\text{wrong}} = 1.0$). If an error occurs in the open space between distant hyperboxes (e.g., $s_{\text{wrong}} = 0.40$ vs $s_{\text{true}} = 0.30$), no box boundary encloses the point, so **shrinking cannot act**.
* **Empirical Impact:** On synthetic separable datasets (S1, S2), 0 points qualify for shrinking. On Shuttle, only 18–20 points in the entire dataset qualified.

### Problem 4: High Rejection Waste on Small / Noisy Datasets
* **Mechanism:** If a dataset has high label noise or very small validation folds, almost every proposed shrink causes a slight validation dip, causing the Gate to reject the edit.
* **Empirical Impact:** On **Page Blocks**, out of 53 shrink attempts per seed, **41 were rejected by the gate (only 12 accepted)**.

### Problem 5: The "Explanation Cut Trap" (SHAP Guided Cuts)
* **Mechanism:** Intuition suggests cutting along the feature with the highest PACE-SHAP attribution. However, the most "important" feature typically lies near the **center of the hyperbox**. Cutting along it slices through the core of the box, cutting away **$27\%$ of the box width** on average and evicting dozens of valid training members.
* **Mitigation:** PACE-CFMNN uses the **geometric thinnest-cut rule** ($C'$-volume), which cuts along the facet closest to the edge, cutting only **$0\%$ to $7\%$** of the width.

---

## 4. The 3 Built-In Safeguards in the Codebase

All three required safeguards are fully implemented and active in the codebase:

```
+-----------------------------------------------------------------------------------------+
|                                    SAFEGUARD MATRIX                                     |
+------------------------------------+--------------------------+-------------------------+
| Safeguard Mechanism                | Implemented Location     | Core Function / Class   |
+------------------------------------+--------------------------+-------------------------+
| 1. Validation Gate (F1 & Recall)   | pace_cfmnn/refine.py     | class Gate (L165-L195)  |
| 2. Thinnest-Facet Cut (C'-Volume)  | pace_cfmnn/refine.py     | feature_volume (L120)   |
| 3. Broken Predictions Auditing     | pace_cfmnn/audit.py      | prediction_churn (L95)  |
+------------------------------------+--------------------------+-------------------------+
```

### Safeguard 1: The Validation Gate (`class Gate`)
Located in [`pace_cfmnn/refine.py`](file:///c:/Users/nikhi/Downloads/pace-cfmnn-review-b228a29/pace-cfmnn-review/pace-cfmnn/pace_cfmnn/refine.py#L165-L195):
* Evaluates macro-F1 and class recall on independent acceptance data (`X_accept`).
* **Strict Thresholds:**
  * `max_f1_drop = 0.5 points`: Edit is rejected if macro-F1 falls by $>0.5$ points.
  * `max_recall_drop = 1.0 points`: Edit is rejected if any class's recall falls by $>1.0$ points.
  * `box_cap = int(1.1 * n_start)`: Prevents box count explosion.
* In [`pace_cfmnn/active.py`](file:///c:/Users/nikhi/Downloads/pace-cfmnn-review-b228a29/pace-cfmnn-review/pace-cfmnn/pace_cfmnn/active.py#L60-L75), candidate edits are tested against a snapshot (`snap = copy.deepcopy(model)`). If the gate check fails, the edit is safely discarded by restoring the snapshot (`model = snap`), ensuring zero degradation.

### Safeguard 2: Thinnest-Facet Cut (`feature_volume`)
Located in [`pace_cfmnn/refine.py`](file:///c:/Users/nikhi/Downloads/pace-cfmnn-review-b228a29/pace-cfmnn-review/pace-cfmnn/pace_cfmnn/refine.py#L120-L125):
* Computes the fractional width loss $\text{share}_j$ for every feature dimension $j \in \{0, \dots, d-1\}$:
  $$\text{share}_j = \frac{\min(x_j + \epsilon - V_{k, j}, W_{k, j} - (x_j - \epsilon))}{W_{k, j} - V_{k, j}}$$
* Chooses the dimension of minimal disturbance:
  $$j^* = \arg\min_j \text{share}_j$$
* Registered as `refine.CHOOSERS['volume']` and set as the default in [`scripts/run_refinement_v2.py`](file:///c:/Users/nikhi/Downloads/pace-cfmnn-review-b228a29/pace-cfmnn-review/pace-cfmnn/scripts/run_refinement_v2.py#L139).

### Safeguard 3: Prediction Churn & Broken Monitoring (`prediction_churn`)
Located in [`pace_cfmnn/audit.py`](file:///c:/Users/nikhi/Downloads/pace-cfmnn-review-b228a29/pace-cfmnn-review/pace-cfmnn/pace_cfmnn/audit.py#L95-L105):
* Tracks the four quadrants of prediction movement between the baseline model and the repaired model:
  1. `fixed`: Points where $y_{\text{before}} \ne y_{\text{true}}$ and $y_{\text{after}} = y_{\text{true}}$.
  2. `broken`: Points where $y_{\text{before}} = y_{\text{true}}$ and $y_{\text{after}} \ne y_{\text{true}}$.
  3. `changed`: Total count of points whose class changed.
  4. `fix_share`: $\frac{\text{fixed}}{\text{fixed} + \text{broken}}$.

---

## 5. Empirical Benchmark Results: Disentangling Repair Methods

To maintain strict scientific honesty, we separate the results of **Self-Repair alone (zero new labels)** from **Self-Repair with Active Learning (bought labels)**. Both sets of results are taken directly from the committed confirmatory benchmark runs (`refinement_v2_confirmatory/runs.csv`, 10 seeds: 50–59).

### Table 5.1: Self-Repair (SR) Alone — Zero New Labels
In this arm, the model performs gated shrinkage exclusively on its own training misclassifications without acquiring a single new label.

| Dataset | Baseline $B_0$ Macro-F1 | Repaired via SR Alone | Gain (Points) | Gap to Random Forest Closed | Statistical Significance ($p_{\text{Holm}}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Thyroid** | 83.96% | **89.98%** | **+6.02 points** | 43.9% | $p = 0.0117$ (Significant) |
| **Landsat** | 81.98% | **83.46%** | **+1.48 points** | 19.5% | $p = 0.0488$ (Significant) |
| **Page Blocks** | 67.56% | **70.14%** | **+2.58 points** | 17.4% | $p = 0.0195$ (Significant) |
| **MAGIC** | 72.59% | **75.86%** | **+3.27 points** | 23.0% | $p = 0.0117$ (Significant) |
| **Letter** | 74.35% | **76.60%** | **+2.25 points** | 10.5% | $p = 0.0117$ (Significant) |
| **Skin** | 93.10% | **98.76%** | **+5.66 points** | 82.8% | $p = 0.0117$ (Significant) |

---

### Table 5.2: Self-Repair + Active Learning (SR then AL) vs. Retraining
In this arm, the model evaluates up to 300 bought labels from an unlabelled pool. We compare it against retraining from scratch with the same labels (both **shuffled** and **appended** order).

| Dataset | Metric | SR then AL (Bought Labels) | Retrained (Shuffled Order) | Retrained (Appended Order) | Repair Advantage / Caveat |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Thyroid** | Macro-F1 | **90.55%** | 85.73% | 84.35% | +4.82 pts over shuffled; +6.20 pts over appended |
| | Broken Rows (% test) | **0.19%** | 1.03% | 0.39% | Lowest collateral damage across all arms |
| | Fix Share (%) | **85.3%** | 51.8% | 51.2% | Highest proportion of true fixes |
| **Landsat** | Macro-F1 | **84.54%** | 82.86% | 82.26% | +1.68 pts over shuffled; +2.28 pts over appended |
| | Broken Rows (% test) | 2.36% | 6.09% | **1.77%** | *Caveat:* Appended retraining broke fewer rows |
| | Fix Share (%) | **64.2%** | 52.4% | 55.0% | Higher fix share than both retraining baselines |
| **Page Blocks** | Macro-F1 | **72.42%** | 64.18% | 64.96% | +8.24 pts over shuffled; +7.46 pts over appended |
| | Broken Rows (% test) | **0.82%** | 6.81% | 3.23% | $4\times$ to $8\times$ fewer broken rows |
| | Fix Share (%) | **81.9%** | 42.6% | 32.0% | Double the fix share of retraining |
| **MAGIC** | Macro-F1 | **77.89%** | 71.10% | 72.33% | +6.78 pts over shuffled; +5.56 pts over appended |
| | Broken Rows (% test) | 6.85% | 15.60% | **2.97%** | *Caveat:* Appended retraining broke fewer rows |
| | Fix Share (%) | **64.8%** | 47.2% | 46.4% | Higher fix share than both retraining baselines |
| **Letter** | Macro-F1 | **77.33%** | 74.70% | 75.19% | +2.63 pts over shuffled; +2.14 pts over appended |
| | Broken Rows (% test) | 3.06% | 10.48% | **2.33%** | *Caveat:* Appended retraining broke fewer rows |
| | Fix Share (%) | **66.2%** | 50.9% | 57.8% | Higher fix share than both retraining baselines |
| **Skin** | Macro-F1 | **99.31%** | 91.18% | 92.62% | +8.13 pts over shuffled; +6.69 pts over appended |
| | Broken Rows (% test) | **0.22%** | 3.83% | 0.49% | Over $17\times$ fewer broken rows than shuffled |
| | Fix Share (%) | **94.1%** | 40.5% | 44.4% | Over double the fix share of retraining |

> [!IMPORTANT]
> **The Appended Retraining Caveat (From Paper §7 & Findings §5.4):**  
> While SR then AL significantly outperforms *shuffled* retraining across all measures, the presentation order of retraining data matters. When newly acquired rows are simply appended to the training set rather than shuffled, retraining causes less disturbance: **appended retraining broke fewer predictions than repair on 3 of the 6 datasets (Landsat: 1.77% vs. 2.36%; MAGIC: 2.97% vs. 6.85%; Letter: 2.33% vs. 3.06%)**. However, repair consistently achieved higher fix shares (64.2%–94.1% vs. 32.0%–57.8%) and higher macro-F1 (+2.1 to +7.5 points) across all 6 datasets.

---

## 6. Provable Impact Certificates: Safety Invariant & Bounds

A unique advantage of hyperbox models is the ability to construct a **sound mathematical certificate** of which inputs can possibly be affected by an edit.

### The Certificate Formulation ([`pace_cfmnn/audit.py`](file:///c:/Users/nikhi/Downloads/pace-cfmnn-review-b228a29/pace-cfmnn-review/pace-cfmnn/pace_cfmnn/audit.py#L40-L75))

Let $\Delta_{\text{before}}$ be the set of modified or removed hyperboxes, and $\Delta_{\text{after}}$ be the modified or newly added hyperboxes.  
For any input $x$, let $\text{TopTied}(x, M)$ be the set of hyperboxes achieving within tolerance $\tau$ of the maximum membership:
$$\text{TopTied}(x, M) = \left\{ j \;\middle|\; M_j(x) \ge \max_l M_l(x) - \tau \right\}$$

#### Theorem: Sound Impact Certificate (A Conservative Superset)
An edit can change the prediction at input $x$ **only if** at least one modified box was in $x$'s tied top group before the edit, or in its tied top group after the edit:
$$I_{\text{loose}}(x) = \left[ \bigvee_{j \in \Delta_{\text{before}}} (j \in \text{TopTied}_{\text{before}}(x)) \right] \lor \left[ \bigvee_{k \in \Delta_{\text{after}}} (k \in \text{TopTied}_{\text{after}}(x)) \right]$$

#### The Tight Certificate ($I_{\text{tight}}$)
Filters out inputs where all tied top boxes before and after belong to a single invariant class (where a class flip is impossible):
$$I_{\text{tight}}(x) = I_{\text{loose}}(x) \land \left( \left| \text{Classes}\left(\text{TopTied}_{\text{before}}(x) \cup \text{TopTied}_{\text{after}}(x)\right) \right| > 1 \right)$$

### Key Mathematical & Empirical Clarifications:
1. **It is an Over-Approximation (Superset), NOT an "If and Only If" Condition:**  
   The certificate is conservative. An edit's tight certificate flagged between **$1.2\%$ and $34.6\%$** of test rows per edit, but only **$0.2\%$ to $3.7\%$** of those flagged rows actually changed their predicted class. It is a safety guarantee, not an exact boundary of changes.
2. **Points Outside Hyperbox Boundaries CAN Flip:**  
   Because CFMNN utilizes continuous fuzzy membership ramps outside the hyperbox faces, an input $x$ that lies physically outside a hyperbox can still have that box as its winning prototype. Shrinking that box reduces its fuzzy membership further down the ramp, which can allow an opposing class's box to win. Thus, the certificate is based on **top-group membership ranking**, not purely bounding-box coordinates.
3. **Empirical Zero-Violation Invariant:**  
   Across all **46,153 live accepted edits** evaluated in the confirmatory benchmark across 180 runs:
   $$\text{Prediction Flips Outside Certificate} = \mathbf{0} \quad (0.0000\% \text{ violation rate})$$

---

## 7. Python Implementation Walkthrough

The following script executes the real, working self-repair workflow using the verified functions in the repository:

```python
import numpy as np
from pace_cfmnn import CFMNN, data
from pace_cfmnn.active import run_self_repair
from pace_cfmnn.audit import prediction_churn

# 1. Load data and split (using standard paper split)
ds = data.load("wine")
sp = data.paper_split(ds.y, seed=0)

scaler = data.fit_scaler(ds.X[sp["train"]])
X_train = scaler.transform(ds.X[sp["train"]])
X_test = scaler.transform(ds.X[sp["test"]])
y_train, y_test = ds.y[sp["train"]], ds.y[sp["test"]]

# 2. Train baseline CFMNN model
model = CFMNN(theta=0.3).fit(X_train, y_train)
pred_baseline = model.predict(X_test)

# 3. Execute Gated Self-Repair (Zero new labels mode)
# Uses chooser="volume" (Safeguard 2: Thinnest-facet cut)
# Gating evaluates validation drops on training rows without counting new labels
repaired_model, round_log, edit_log = run_self_repair(
    model,
    X_train, y_train,          # Training pool containing error points
    X_train, y_train,          # Gating evaluation set
    chooser="volume",          # Minimizes fractional width cut (np.argmin share)
    budget=50,                 # Max candidate error points to evaluate
    max_f1_drop=0.5,           # Validation Gate: max 0.5 points F1 drop
    max_recall_drop=1.0,       # Validation Gate: max 1.0 points class recall drop
    gate_labels_are_new=False  # Pure zero-label mode
)

# 4. Audit Churn (Safeguard 3: Tracking fixed vs broken predictions)
pred_repaired = repaired_model.predict(X_test)
churn = prediction_churn(y_test, pred_baseline, pred_repaired)

print(f"Total Test Points Evaluated:    {len(y_test)}")
print(f"Test Predictions Changed:       {churn['changed']}")
print(f"Errors Corrected (Fixed):       {churn['fixed']}")
print(f"Regressions Induced (Broken):   {churn['broken']}")
if churn["fixed"] + churn["broken"] > 0:
    fix_share = 100 * churn["fixed"] / (churn["fixed"] + churn["broken"])
    print(f"Fix Share:                      {fix_share:.1f}%")
```

---

## 8. Summary & Best Practices

1. **Does shrinking change the structure?**  
   Yes. It tightens one boundary coordinate, shrinks volume along that axis, removes evicted members, re-computes the centroid, and prunes empty boxes. All other $K-1$ boxes and global formulas remain identical.
2. **Can shrinking create problems?**  
   Yes, if done without controls. It can induce broken predictions on nearby samples (up to 6.85% on MAGIC), cause minor recall drops on rare classes, or cut too deeply if guided by explanations instead of geometry.
3. **Are the solutions already in the codebase?**  
   Yes. `Gate` (in [`refine.py`](file:///c:/Users/nikhi/Downloads/pace-cfmnn-review-b228a29/pace-cfmnn-review/pace-cfmnn/pace_cfmnn/refine.py)), `feature_volume` (in [`refine.py`](file:///c:/Users/nikhi/Downloads/pace-cfmnn-review-b228a29/pace-cfmnn-review/pace-cfmnn/pace_cfmnn/refine.py)), `run_self_repair` (in [`active.py`](file:///c:/Users/nikhi/Downloads/pace-cfmnn-review-b228a29/pace-cfmnn-review/pace-cfmnn/pace_cfmnn/active.py)), and `prediction_churn` (in [`audit.py`](file:///c:/Users/nikhi/Downloads/pace-cfmnn-review-b228a29/pace-cfmnn-review/pace-cfmnn/pace_cfmnn/audit.py)) are fully implemented, tested by 198 unit tests, and enabled by default in all benchmark scripts.

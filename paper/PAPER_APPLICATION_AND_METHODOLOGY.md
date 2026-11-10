# PACE-CFMNN: Application Architecture and Methodology

> **Reference Document:** Corresponds to the 33-page paper draft *PACE: Exact Explanations, Trust Signals and Auditable Repair for Centroid-Based Fuzzy Min-Max Classifiers* (commit `b228a29` / rev 311).

---

## 1. Executive Summary & Research Questions

The **PACE-CFMNN** framework provides a unified suite for **exact, deterministic explainability (XAI)**, **unsupervised trust signals**, and **auditable, certified post-hoc repair** for Centroid-Based Fuzzy Min-Max Neural Networks (CFMNN).

```
                             INPUT POINT x
                                   │
                                   ▼
             ┌───────────────────────────────────────────┐
             │       CFMNN BASE CLASSIFIER               │
             │       • Hyperboxes [V_j, W_j]             │
             │       • Centroids μ_j                     │
             │       • Additive feature membership terms │
             └─────────────────────┬─────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        ▼                          ▼                          ▼
┌───────────────────────┐ ┌────────────────────────┐ ┌───────────────────────┐
│   PACE EXPLANATIONS   │ │     TRUST SIGNALS      │ │   AUDITABLE REPAIR    │
│ • PACE-SHAP (exact)   │ │ • Prediction Disagree  │ │ • 100% errors = ties │
│ • PACE-flip (occl.)   │ │ • Explanation Instab.  │ │ • Gated Face Shrink   │
│ • Contrastive (A vs B)│ │ • Corrected Uncert U*  │ │ • Input Certificates │
│ • Counterfactuals     │ │ • Order Ensemble       │ │ • Fix vs Break Audit  │
└───────────────────────┘ └────────────────────────┘ └───────────────────────┘
```

### The Four Research Questions (RQs)
* **RQ1 (Explanations):** Are PACE's explanations exact, faithful to the model, and faster than existing black-box methods (KernelSHAP, LIME, brute-force occlusion)?
* **RQ2 (Trust Signals):** Can signals derived from the model's hyperbox geometry and perturbation response reliably flag misclassified predictions without labels?
* **RQ3 (Auditable Repair):** When counting every annotation, can CFMNN repair its own training errors without new labels, and does buying new labels improve the model beyond self-repair?
* **RQ4 (Safety Certificates):** Can we bound, for each hyperbox edit, a finite set of inputs outside of which no prediction can change, and are repair's updates more targeted than retraining?

---

## 2. Base Classifier Architecture: CFMNN (`cfmnn_core.py`)

### 2.1 Hyperbox Formulation
CFMNN (Aadarsh & Sai Prasad, *PReMI 2023*) covers $C$ classes using axis-aligned hyperboxes $B_j = [V_j, W_j]$ in normalized space $[0, 1]^d$, where $V_j$ is the minimum coordinate vector and $W_j$ is the maximum coordinate vector.

Each box $j$ maintains:
1. Min corner: $V_j \in [0, 1]^d$
2. Max corner: $W_j \in [0, 1]^d$
3. Exemplar Sum: $S_j = \sum_{x \in \text{members}} x \in \mathbb{R}^d$
4. Exemplar Count: $N_j = |\text{members}| \in \mathbb{N}$
5. Centroid: $\mu_j = \frac{S_j}{N_j}$
6. Associated Class: $c_j \in \{0, \dots, C-1\}$

### 2.2 Simpson's Two-Sided Membership Ramp
The membership score is governed by Simpson's linear ramp function with slope parameter $\gamma$ (default $\gamma = 4.0$):
$$f(t, \gamma) = \max\left(0, 1 - \max(0, \gamma \min(1, t))\right)$$

For input point $x$ and box $j$, the coordinate-wise score along feature $i$ is:
$$m_{j,i}(x_i) = \frac{1}{2} \left[ f(x_i - W_{j,i}, \gamma) + f(V_{j,i} - x_i, \gamma) \right]$$

* **Inside Box:** If $V_{j,i} \le x_i \le W_{j,i}$, both differences are $\le 0 \implies m_{j,i} = 1.0$.
* **Outside Box:** Decreases linearly with slope $\frac{\gamma}{2}$ to a floor of $0.5$.
* **Key Invariant:** $m_{j,i}(x_i) \in [0.5, 1.0]$ for all $x, j, i$.

### 2.3 Total Membership & Support
The overall membership of point $x$ in box $j$ is the unweighted arithmetic mean over all $d$ features:
$$b_j(x) = \frac{1}{d} \sum_{i=1}^d m_{j,i}(x_i) \in [0.5, 1.0]$$

Class support is the maximum membership across all boxes of that class:
$$s_c(x) = \max_{j: c_j = c} b_j(x)$$
$$s'_c(x) = 2 s_c(x) - 1 \in [0, 1] \quad \text{(rescaled support)}$$

### 2.4 Decision Rule & Tie-Breaking
Prediction is evaluated via Algorithm 4 with dual-level tie resolution:
1. **Primary Rule:** Find all candidate boxes within tolerance $\tau = 10^{-9}$ of the maximum membership:
   $$\text{WinList}(x) = \left\{ j \;\middle|\; b_j(x) \ge \max_k b_k(x) - \tau \right\}$$
2. **Centroid Tie-Breaker:** If $|\text{WinList}| > 1$, the box whose centroid $\mu_j$ has the smallest squared Euclidean distance to $x$ wins:
   $$j^* = \arg\min_{j \in \text{WinList}(x)} \|x - \mu_j\|^2$$
3. **Deterministic Index Tie-Breaker:** If centroid distances also tie, the lowest box creation index wins.
4. **Predicted Class:** $y_{\text{pred}} = c_{j^*}$.

### 2.5 Critical Algorithmic Corrections Made to the Literature
The repository uncovered and fixed two fundamental bugs in the PReMI 2023 paper pseudocode:

| Bug in Literature | Original Paper Flaw | Repository Fix | Empirical Impact |
| :--- | :--- | :--- | :--- |
| **Centroid Accounting Bug** | Algorithm 2 skipped updating $S_j$ and $N_j$ if $b_{\max} = 1.0$ (point already inside box). Centroids were frozen from the first 1–2 points. | `centroid_rule="fixed"` updates $S_j \mathrel{+}= x$ and $N_j \mathrel{+}= 1$ for all absorbed points. | Shuttle accuracy jumped from **93.3% to 99.8%**; fixes 6.5% of test predictions. |
| **Search Expansion Collapse** | Algorithm 2 tested expansion only on the single best same-class box. If full, it spawned a dead point-box. | `expansion="search"` tries candidate boxes in descending order of membership (Simpson 1992 style). | Digits accuracy rose from **81.7% to 95.3%**; reduces box count by $2.5\times$ to $13\times$. |

---

## 3. The PACE Explanation Suite (`explain.py`)

Because $b_j(x) = \frac{1}{d} \sum_{i=1}^d m_{j,i}(x_i)$ decomposes additively, replacing any subset of features changes $b_j$ by an amount that depends on those features alone. PACE leverages this property to achieve exact, closed-form explanations without re-running the model.

```
                    PACE EXPLANATION METHODS
┌────────────────────────────────────────────────────────────────────────┐
│ 1. PACE-SHAP: Exact Shapley values of winning margin (d ≤ 12)          │
│ 2. PACE-SHAP-cond: Gaussian conditional replacement (sigma=0.1)        │
│ 3. PACE-flip: Exact sub-millisecond single-feature occlusion (32 quant)│
│ 4. Contrastive: "Why Class A, not Class B" (membership vs centroid)   │
│ 5. PACE-multi: Multi-box competitor contrastive with softmin weighting │
│ 6. Counterfactual: Targeted boundary projections with locked features  │
│ 7. PACE-sparse: Minimum-feature hyperbox boundary hops                 │
└────────────────────────────────────────────────────────────────────────┘
```

### 3.1 PACE-SHAP (Exact Shapley Attribution)
Explains the predicted class margin:
$$f(x) = s'_A(x) - \max_{c \neq A} s'_c(x)$$
Given background dataset $Z$, the Shapley value $\phi_i(x)$ is computed from per-feature membership terms:
$$b_j(x_S \cup z_{D \setminus S}) = b_j(z) + \sum_{i \in S} \frac{m_{j,i}(x_i) - m_{j,i}(z_i)}{d}$$

* **Exact Mode ($d \le 12$):** Evaluates all $2^d \le 4,096$ coalitions exactly using bit-shift tables. Matches `shap.ExactExplainer` to machine precision ($3.3 \times 10^{-16}$) while running **$16\times$–$30\times$ faster**.
* **Sampled Mode ($d > 12$):** Draws 16 antithetic permutation orders paired with their exact inverses (32 orders per background row), achieving $1\%$–$2\%$ error with guaranteed efficiency sum $\sum \phi_i = f(x) - \mathbb{E}[f(z)]$.
* **Provable Box Pruning:** Computes lower and upper membership bounds across all subsets:
  $$\text{floor}_j = b_j(z) + \sum_{i: \Delta_{j,i} < 0} \Delta_{j,i}, \quad \text{ceil}_j = b_j(z) + \sum_{i: \Delta_{j,i} > 0} \Delta_{j,i}$$
  Any box where $\text{ceil}_j < \max_k \text{floor}_k$ can never win under any coalition and is safely pruned, **eliminating $71\%$–$98\%$ of boxes upfront**.

### 3.2 PACE-flip (Fast Occlusion)
Computes the effect of replacing feature $i$ across 32 background quantile values:
* Standard occlusion re-runs the model $d \times 32$ times.
* PACE-flip pre-materializes a $(d, H, 32)$ lookup tensor of feature terms.
* Computes exact prediction flips in **$0.05$ to $1.4$ ms** (**$25\times$ to $134\times$ faster** than batched brute-force occlusion).

### 3.3 Contrastive Explanations ("Why Class A, not B")
* **Membership Mode ($s'_A > s'_B + \tau$):**
  $$e_i = \frac{m_{A,i}(x_i) - m_{B,i}(x_i)}{d}$$
* **Tie Mode ($|s'_A - s'_B| \le \tau$):**
  When memberships tie, the decision rests entirely on centroid proximity. PACE switches to centroid differentials:
  $$e_i = (x_i - \mu_{B,i})^2 - (x_i - \mu_{A,i})^2$$

### 3.4 Counterfactuals & PACE-sparse (`counterfactual.py`)
* **PACE Counterfactuals:** Finds the orthogonal projection of $x$ onto the boundary faces of target-class boxes, allowing immutable features (e.g., age, sex) to be locked.
* **PACE-sparse:** Searches for the minimum number of feature alterations by iterating over target-class box faces, scoring candidate moves directly from CFMNN's closed-form terms.

---

## 4. Trust Signals & Error Detection (`signals.py`)

PACE introduces unsupervised trust signals to detect likely misclassifications without ground-truth labels:

1. **Prediction Disagreement (PD):** 
   $$PD(x) = \frac{1}{K} \sum_{k=1}^K \mathbb{I}[f(x + \epsilon_k) \neq f(x)] \quad (K=50 \text{ Gaussian perturbations})$$
2. **Explanation Instability (EI):**
   Cosine distance drift of contrastive explanation vectors under the same $K=50$ perturbations.
3. **Corrected Uncertainty ($U^*$):**
   * *Flaw in Classic $U$:* Classic uncertainty assigned high confidence to ties because both class supports were $1.0$.
   * *The $U^*$ Formulation:* Explicitly segments tied from untied inputs:
     $$U^*(x) = \begin{cases} \frac{1 + \text{ratio}}{2} & \text{if tied, where } \text{ratio} = \frac{\|x - \mu_A\|}{\|x - \mu_B\|} \in [0, 1] \\ \frac{1 - (s'_A - s'_B)}{2} & \text{if untied} \end{cases}$$
     Forces all tied predictions into the highest risk interval $[0.5, 1.0]$.
4. **Order Ensemble Disagreement:**
   Disagreement across 5 CFMNN models trained on identical data presented in different row orders.

---

## 5. Auditable Self-Repair & Provable Certificates (`refine.py`, `audit.py`)

### 5.1 The 100% Cross-Class Tie Law
A core empirical discovery of the paper:
$$\forall x \in \text{Training Errors}, \quad b_{y_{\text{true}}}(x) = b_{y_{\text{pred}}}(x) = 1.0$$
Across all 13 benchmark datasets (347,339 verified training errors), **100% of errors are cross-class ties**. CFMNN never fails because it lacked a box for the true class; it fails because hyperboxes of different classes overlap, and the centroid heuristic chose the wrong class.

### 5.2 The Gated Shrink Operator
To fix an error at training sample $x$ without new annotations:
1. Identify the offending wrong-class box $k$ containing $x$.
2. Compute the minimum-volume face cut across all dimensions $i \in \{1, \dots, d\}$:
   $$\text{cut}_i = \begin{cases} x_i + 0.001 & \text{if cutting lower face } (x_i - V_{k,i} \le W_{k,i} - x_i) \\ x_i - 0.001 & \text{if cutting upper face } (W_{k,i} - x_i < x_i - V_{k,i}) \end{cases}$$
   Select dimension $i^* = \arg\min_i \frac{\Delta \text{width}_i}{\text{width}_i}$.
3. Shift the face by margin $\epsilon = 0.001$.
4. Eject training exemplars that fall outside the new face; recompute centroid $\mu_k$, sum $S_k$, and count $N_k$ from retained exemplars. If no exemplars remain, delete box $k$.

### 5.3 The Acceptance Gate (`Gate`)
An edit is applied provisionally and audited against acceptance rows. It is kept only if:
1. $\Delta \text{Macro-F1} \ge -0.5\%$
2. $\Delta \text{Recall}_c \ge -1.0\%$ for every class $c$
3. Total hyperboxes $N_{\text{boxes}} \le 1.1 \times N_{\text{initial}}$
Otherwise, the entire edit is rolled back in place.

### 5.4 Provable Input Certificates
To guarantee safe AI updates, PACE mathematically certifies which inputs an edit can affect:

$$\text{TopTied}(x, M) = \left\{ j \;\middle|\; b_j(x) \ge \max_k b_k(x) - \tau \right\}$$

Let $\Delta_{\text{before}}$ be the set of modified or deleted boxes, and $\Delta_{\text{after}}$ be modified or newly added boxes.

#### Theorem (Exact Impact Certificate)
An edit can change the prediction at input $x$ **if and only if** at least one changed box is among $x$'s top tied group before or after the edit:
$$I_{\text{loose}}(x) = \left[ \bigvee_{j \in \Delta_{\text{before}}} (j \in \text{TopTied}(x, M_{\text{before}})) \right] \lor \left[ \bigvee_{k \in \Delta_{\text{after}}} (k \in \text{TopTied}(x, M_{\text{after}})) \right]$$

#### Tight Certificate ($I_{\text{tight}}$)
Drops inputs where all tied top boxes before and after belong to a single invariant class:
$$I_{\text{tight}}(x) = I_{\text{loose}}(x) \land \left( \left| \text{Classes}(\text{TopTied}_{\text{before}}(x) \cup \text{TopTied}_{\text{after}}(x)) \right| > 1 \right)$$

* **Mathematical Guarantee:** $\forall x \notin I_{\text{tight}}, \quad \text{Pred}_{\text{after}}(x) \equiv \text{Pred}_{\text{before}}(x)$.
* **Empirical Validation:** Tested across **46,153 accepted edits**; **zero predictions changed outside the certificate**.

---

## 6. Implementation Summary

| Component | Source File | Core Functions / Classes | Primary Invariant / Output |
| :--- | :--- | :--- | :--- |
| **CFMNN Core** | `cfmnn_core.py` | `CFMNN`, `feature_terms`, `ramp` | Scikit-learn estimator; $m \in [0.5, 1.0]$. |
| **PACE-SHAP** | `explain.py` | `PaceShap`, `_exact`, `_sampled` | Exact Shapley margin attributions; bounds pruning. |
| **PACE-flip** | `explain.py` | `PaceFlip`, `pace_flip` | Sub-ms occlusion rankings across 32 quantiles. |
| **Trust Signals**| `signals.py` | `SignalComputer`, `uncertainty` | `PD`, `EI`, `U`, `U*`, `box_evidence`. |
| **Self-Repair** | `refine.py` | `shrink_box`, `face_cut`, `Gate` | Closed-form $\epsilon$-cut; rollback gate checks. |
| **Audit** | `audit.py` | `impact_set`, `certify`, `top_groups` | Exact loose and tight prediction certificates. |
| **Active Loop** | `active.py` | `run_refinement`, `run_self_repair` | Budgeted repair and annotation loops. |


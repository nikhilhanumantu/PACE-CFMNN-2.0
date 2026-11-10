# PACE-CFMNN: Experimental Results, Accuracy and Benchmarks

> **Reference Document:** Complete empirical compilation of model accuracy, cross-model benchmarks, and explanation metrics for **PACE-CFMNN**.
> **Data Sources:** Recomputed from committed raw CSV runs in `results/` and verified via `research_audit/CLAIMS.md`.

---

## 1. Direct Head-to-Head Comparison: Existing Standard CFMNN vs. This Application (PACE-CFMNN)

This section provides the central comparison between the **Existing Standard CFMNN** (baseline architecture with standard centroid and expansion rules) and **This Application (PACE-CFMNN)** with architectural enhancements, exact XAI engines, and auditable self-repair.

### 1.1 Base Model Accuracy, Macro-F1, and Model Complexity
Comparison on an 80/20 train/test split across 5 random seeds (mean over seeds).
* *Existing CFMNN:* Standard centroid rule (skipping internal points) and standard single-box expansion.
* *PACE-CFMNN (This Application):* Incorporates fixed centroid tracking and multi-box search expansion.

| Dataset | Existing CFMNN Accuracy (%) | PACE-CFMNN Accuracy (%) | Accuracy Improvement | Existing CFMNN Box Count | PACE-CFMNN Box Count | Model Compression (Complexity Reduction) | Existing CFMNN Macro-F1 (%) | PACE-CFMNN Macro-F1 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Digits** | 81.72% | **95.33%** | **+13.61%** | 1,316 | **293** | **$4.5\times$ fewer boxes** | 81.12% | **95.26%** (+14.1%) |
| **S1 (Synthetic)** | 78.86% | **88.74%** | **+9.88%** | 3,748 | **1,514** | **$2.5\times$ fewer boxes** | 78.84% | **88.74%** (+9.9%) |
| **S2 (Synthetic)** | 71.30% | **79.36%** | **+8.06%** | 3,755 | **1,534** | **$2.4\times$ fewer boxes** | 71.28% | **79.36%** (+8.1%) |
| **Shuttle** | 93.33% | **99.80%** | **+6.47%** | 27 | **14** | **$1.9\times$ fewer boxes** | 82.58% | **86.94%** (+4.4%) |
| **Wine** | 95.56% | **98.33%** | **+2.77%** | 123 | **31** | **$4.0\times$ fewer boxes** | 95.70% | **98.46%** (+2.8%) |
| **Thyroid** | 95.04% | **97.17%** | **+2.13%** | 688 | **53** | **$13.0\times$ fewer boxes**| 78.08% | **85.98%** (+7.9%) |
| **Breast Cancer** | 94.56% | **95.44%** | **+0.88%** | 374 | **44** | **$8.5\times$ fewer boxes** | 94.18% | **95.08%** (+0.9%) |

---

### 1.2 Performance After Auditable Self-Repair (Macro-F1 Progression)
In addition to the base classifier improvements, **This Application introduces label-free Self-Repair**, shrinking overlapping boxes where the base model errs:

| Dataset | Existing CFMNN Baseline (B0) | PACE-CFMNN Self-Repair (Zero New Labels) | Self-Repair Net Gain | PACE-CFMNN Active Refinement (SR + AL) | Total Net Gain |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Thyroid** | 84.0% | **90.0%** | **+6.0%** | **90.5%** | **+6.5%** |
| **Skin** | 93.1% | **98.8%** | **+5.7%** | **99.3%** | **+6.2%** |
| **MAGIC Telescope** | 72.6% | **75.9%** | **+3.3%** | **77.9%** | **+5.3%** |
| **Page Blocks** | 67.6% | **70.1%** | **+2.5%** | **72.4%** | **+4.8%** |
| **Letter Recognition**| 74.3% | **76.6%** | **+2.3%** | **77.3%** | **+3.0%** |
| **Landsat Satellite** | 82.0% | **83.5%** | **+1.5%** | **84.5%** | **+2.5%** |

* **Zero-Annotation Advantage:** Self-repair requires **0 additional labels**, closing between **10% and 83% of the accuracy gap to a Random Forest**.

---

### 1.3 Update Reliability: Standard Retraining vs. PACE Audited Repair
When modifying or updating a deployed model:

| Metric | Standard Retraining (Existing CFMNN) | PACE-CFMNN Self-Repair (This Application) | Advantage of This Application |
| :--- | :---: | :---: | :--- |
| **Fix Share (Edits that Fix vs. Break)** | 40.5% – 52.4% | **64.2% – 94.1%** | **+12% to +54%** higher ratio of genuine corrections |
| **Broken Predictions (Regressions)** | 3.8% – 15.6% of test set | **0.2% – 6.8%** of test set | Dramatically fewer negative side-effects |
| **Safety Guarantees** | None (global unpredictable shifts) | **Exact Input Certificates** | **0 flips outside certificate** across 46,153 edits |
| **Label Requirement** | Requires full dataset retraining | **Zero new labels** | Operates directly on detected training overlaps |

---

### 1.4 Comprehensive Architectural & Capability Comparison

| Capability | Existing Standard CFMNN | PACE-CFMNN (This Application) | Advantage / Performance Difference |
| :--- | :--- | :--- | :--- |
| **Centroid Tracking** | Skips points inside existing boxes; centroid frozen from 1–2 points | Dynamically incorporates every accepted point into running sum/counts | Eliminates arbitrary tie-break errors (**+6.5% on Shuttle**) |
| **Expansion Strategy** | Tests single best box only; triggers single-point box explosions | Fallback search expansion over eligible same-class hyperboxes | Reduces model complexity by **$2.5\times$ to $13\times$** |
| **Feature Attributions** | None native; requires black-box approximations (KernelSHAP) | Exact closed-form `PACE-SHAP` ($d \le 12$) & Antithetic sampling | **16–30× faster** than exact SHAP; 100% deterministic |
| **Decision-Flipping Speed** | Naive occlusion ($d \times 32$ full model passes) | Pre-computed lookup tensors (`PACE-flip`) | **25–134× faster** ($0.05$–$1.4$ ms latency) |
| **Contrastive Insights** | None | Dual-mode "Why Class A, not B" (membership & centroid) | Instant feature-level explanation of competitive margin |
| **Counterfactuals** | External black-box search (e.g. DiCE, 500–10,000 ms) | Direct geometric projection onto target hyperbox boundaries | **70–1,000× faster**; lands closer to real target data |
| **Confidence & Trust** | Winning margin only (collapses to 0 on ties) | Corrected $U^*$ (grades cross-class ties $[0.5, 1.0]$) + PD | Reliably flags suspicious predictions without labels |
| **Error Diagnosis** | Unknown failure mechanisms | Mathematical proof that **100% of training errors are ties** | Pinpoints exact overlapping boxes causing errors |
| **Post-Hoc Repair** | Retraining from scratch (high churn) | Gated minimal-volume face contraction (Self-Repair) | Surgical patch resolving overlaps with **0 new labels** |
| **Auditability** | None | Provable Impact Certificates ($I_{\text{loose}}$, $I_{\text{tight}}$) | Formally certifies untouched input regions |

---

## 2. Benchmark Datasets Overview

The experimental suite spans **13 primary datasets** across synthetic, UCI, and Scikit-Learn benchmarks:

| Dataset | Domain | Samples ($N$) | Features ($d$) | Classes ($C$) | Purpose in Benchmark |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **S1** | Synthetic | 5,000 | 10 | 2 | Additive ground truth (only $x_0, x_1, x_2$ matter; linear sum) |
| **S2** | Synthetic | 5,000 | 10 | 2 | Additive with class imbalance, label noise, shift variants |
| **S3** | Synthetic | 5,000 | 10 | 2 | Non-linear interaction ground truth (AND gate on $x_0, x_1$) |
| **Wine** | Chemical | 178 | 13 | 3 | Small tabular benchmark ($d > 12$ threshold) |
| **Breast Cancer** | Medical | 569 | 30 | 2 | High-dimensional real-world biomedical data |
| **Digits** | Image / OCR | 1,797 | 64 | 10 | Multi-class high-dimensional benchmark |
| **Thyroid** | Medical | 7,200 | 21 | 3 | Highly imbalanced real-world benchmark |
| **Shuttle** | Aerospace | 58,000 | 9 | 7 | Dense aerospace sensor data (99.5% inside points) |
| **Skin** | Vision / Bio | 245,057 | 3 | 2 | Large-scale 3D RGB skin segmentation |
| **MAGIC** | Physics | 19,020 | 10 | 2 | Atmospheric Cherenkov telescope gamma rays |
| **Letter** | Image / OCR | 20,000 | 16 | 26 | Multi-class letter recognition (high geometric complexity) |
| **Landsat** | Remote Sensing| 6,435 | 36 | 6 | Satellite multi-spectral pixel classification |
| **Page Blocks** | Document AI | 5,473 | 10 | 5 | Text/graphic segmentation (heavy class imbalance) |

---

## 3. Comparison with General Machine Learning Baselines

All models evaluated on an **80/20 stratified train/test split** across **5 random seeds** (mean $\pm$ standard deviation):

| Dataset | Training Rows | CFMNN Accuracy (%) | 10-NN (%) | Random Forest (200 trees) (%) | Accuracy Deficit (CFMNN vs. Best) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Shuttle** | 46,400 | **99.8 ± 0.0** | 99.8 ± 0.0 | 100.0 ± 0.0 | **$-0.2\%$** |
| **Wine** | 142 | **98.3 ± 2.5** | 95.6 ± 3.2 | 100.0 ± 0.0 | **$-1.7\%$** |
| **Breast Cancer** | 455 | **95.4 ± 1.1** | 97.4 ± 1.5 | 97.0 ± 1.0 | **$-1.9\%$** |
| **Thyroid** | 5,760 | **97.2 ± 0.6** | 94.2 ± 0.2 | 99.6 ± 0.1 | **$-2.5\%$** |
| **Digits** | 1,437 | **95.3 ± 0.5** | 98.0 ± 0.7 | 97.6 ± 0.2 | **$-2.7\%$** |
| **Skin** | 196,045 | **94.7 ± 2.1** | 99.9 ± 0.0 | 100.0 ± 0.0 | **$-5.2\%$** |
| **Landsat** | 5,148 | **84.3 ± 1.1** | 89.8 ± 0.4 | 91.7 ± 0.9 | **$-7.4\%$** |
| **S2 (Synthetic)** | 4,000 | **80.6 ± 1.6** | 85.8 ± 1.8 | 90.1 ± 1.3 | **$-9.5\%$** |
| **S1 (Synthetic)** | 4,000 | **89.6 ± 1.1** | 95.1 ± 0.8 | 99.6 ± 0.1 | **$-10.0\%$** |
| **Page Blocks** | 4,378 | **86.3 ± 5.6** | 94.8 ± 0.4 | 97.3 ± 0.6 | **$-11.0\%$** |
| **S3 (Synthetic)** | 4,000 | **87.0 ± 0.8** | 88.7 ± 0.7 | 100.0 ± 0.0 | **$-13.0\%$** |
| **MAGIC Telescope** | 15,216 | **72.5 ± 0.7** | 83.6 ± 0.4 | 88.3 ± 0.8 | **$-15.8\%$** |
| **Letter Recognition**| 16,000 | **74.6 ± 0.7** | 94.2 ± 0.4 | 96.2 ± 0.2 | **$-21.5\%$** |

---

## 4. Interpretable / Readable Baselines Benchmark

Tested against other interpretable models on 20 random seeds (seeds 70–89), using paired two-sided Wilcoxon tests with Holm correction:

| Dataset | CFMNN ($\theta=0.3$) | CFMNN (Tuned $\theta$) | CART Tree (Tuned) | Logistic Reg. (Tuned) | EBM (Main Effects) | GFMM ($\theta=0.3$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **S1** | 89.1% | 90.6% | 97.6% | **100.0%** | **100.0%** | 89.2% |
| **S2** | 80.8% | 81.6% | 88.0% | **90.9%** | 90.5% | 79.8% |
| **S3** | 87.8% | 88.1% | **99.9%** | 82.1% | **99.9%** | 83.4% |
| **Wine** | **98.3%** | 98.2% | 89.6% | 98.2% | 98.1% | 96.0% |
| **Breast Cancer** | 95.4% | 95.9% | 93.7% | 97.7% | **98.3%** | 95.1% |
| **Digits** | 96.0% | 95.8% | 84.5% | 97.2% | 97.3% | **98.3%** |
| **Thyroid** | 97.4% | 97.4% | **99.6%** | 95.2% | **99.6%** | 96.2% |
| **Shuttle** | 99.8% | 99.9% | **100.0%** | 95.8% | **100.0%** | 88.5% |
| **Skin** | 96.0% | 99.8% | **99.9%** | 91.9% | 97.2% | 98.8% |
| **MAGIC** | 71.6% | 81.8% | 85.0% | 79.0% | **85.8%** | 73.4% |
| **Letter** | 75.1% | **95.1%** | 86.6% | 77.3% | 87.7% | 84.1% |
| **Landsat** | 85.1% | **90.0%** | 86.6% | 85.8% | **90.0%** | 86.3% |
| **Page Blocks** | 85.0% | 95.4% | 96.8% | 95.6% | **97.3%** | 90.2% |

### Head-to-Head Win / Tie / Loss Summary
Counts represent datasets where CFMNN was **Statistically Better / Level / Worse** (Holm-corrected $p < 0.05$):
* **CFMNN (Tuned $\theta$) vs. CART Tree:** Better on 5, Level on 0, **Worse on 8**
* **CFMNN (Tuned $\theta$) vs. Logistic Regression:** **Better on 7**, Level on 2, Worse on 4
* **CFMNN (Tuned $\theta$) vs. EBM:** Better on 2, Level on 2, **Worse on 9**
* **CFMNN ($\theta=0.3$) vs. GFMM ($\theta=0.3$):** Better on 4, Level on 3, **Worse on 6**

---

## 5. Explanation Faithfulness & Speed Benchmark (RQ1)

Tested on **10 seeds (30–39)** across 50 test points per seed (36 on Wine).
* **Deletion:** Lower AUC is better (faster margin collapse = higher faithfulness).
* **Insertion:** Higher AUC is better (faster margin recovery = higher faithfulness).

### 5.1 Deletion & Insertion Margin AUC Table
*(Values: Deletion AUC / Insertion AUC; bold indicates best method in each row).*

| Dataset | PACE-SHAP (Ours) | KernelSHAP (20 rows) | Strong KernelSHAP (2048 smp) | LIME | PACE-flip | Random Ranking |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **S1** | **$-0.139$ / $0.645$** | $-0.133$ / $0.635$ | $-0.131$ / $0.636$ | $-0.110$ / $0.609$ | $-0.089$ / $0.617$ | $0.271$ / $0.296$ |
| **S2** | **$-0.132$ / $0.609$** | $-0.125$ / $0.592$ | $-0.128$ / $0.593$ | $-0.095$ / $0.565$ | $-0.076$ / $0.561$ | $0.256$ / $0.257$ |
| **S3** | **$-0.055$ / $0.639$** | $-0.041$ / $0.626$ | $-0.044$ / $0.627$ | $+0.012$ / $0.569$ | $+0.006$ / $0.595$ | $0.318$ / $0.314$ |
| **Wine** | **$-0.076$ / $0.533$** | $-0.068$ / $0.524$ | $-0.070$ / $0.526$ | $-0.039$ / $0.492$ | $-0.055$ / $0.489$ | $0.236$ / $0.239$ |
| **Breast Cancer** | **$+0.111$ / $0.699$** | $+0.174$ / $0.634$ | $+0.122$ / $0.687$ | $+0.141$ / $0.680$ | $+0.137$ / $0.685$ | $0.398$ / $0.414$ |
| **Thyroid** | $+0.549$ / $0.915$ | $+0.575$ / $0.892$ | $+0.562$ / $0.903$ | $+0.571$ / $0.874$ | **$+0.544$ / $0.918$** | $0.739$ / $0.736$ |
| **Shuttle** | $+0.370$ / $0.744$ | **$+0.365$ / $0.747$** | $+0.366$ / $0.747$ | $+0.381$ / $0.721$ | $+0.374$ / **$0.750$** | $0.552$ / $0.536$ |
| **MAGIC** | **$-0.052$ / $0.290$** | $-0.035$ / $0.272$ | $-0.038$ / $0.279$ | $-0.011$ / $0.235$ | $+0.015$ / $0.234$ | $0.140$ / $0.093$ |
| **Letter** | **$-0.690$ / $-0.000$** | $-0.678$ / $-0.038$ | $-0.684$ / $-0.022$ | $-0.656$ / $-0.136$ | $-0.644$ / $-0.072$ | $-0.381$ / $-0.423$ |
| **Landsat** | **$-0.329$ / $0.231$** | $-0.217$ / $0.077$ | $-0.303$ / $0.203$ | $-0.295$ / $0.126$ | $-0.255$ / $0.134$ | $-0.050$ / $-0.057$ |
| **Page Blocks** | $+0.050$ / $0.204$ | $+0.046$ / $0.205$ | **$+0.043$ / $0.212$** | $+0.049$ / $0.196$ | $+0.051$ / $0.202$ | $0.128$ / $0.095$ |

### 5.2 Runtime and Speedup Benchmarks
Timing measured per explanation (batch of 50 points; Intel Core i7-8650U):

| Dataset | PACE-flip Latency | Brute-force Occlusion | PACE-flip Speedup | PACE-SHAP (Exact) | ExactExplainer (`shap`) | PACE-SHAP Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **S1** | 0.81 ms | 71.3 ms | **$88\times$** | 2.5 s | 53.7 s | **$21\times$** |
| **Wine** | 0.22 ms | 10.8 ms | **$49\times$** | 0.12 s | 3.1 s | **$26\times$** |
| **Shuttle** | 0.05 ms | 4.0 ms | **$80\times$** | 19 ms | 478 ms | **$25\times$** |
| **Landsat** | 1.12 ms | 158.0 ms | **$141\times$** | 0.9 s *(sampled)* | 41.7 s *(exact)* | **$46\times$** |

---

## 6. Counterfactual Explanations Benchmark

Evaluated across 10 seeds (seeds 40–49), comparing PACE with DiCE (Mothilal et al., 2020):

| Dataset | Size of Change (PACE / DiCE) | Features Changed (PACE / DiCE) | Distance to Real Data (PACE / DiCE) | Latency (PACE / DiCE) |
| :--- | :---: | :---: | :---: | :---: |
| **S1** | **0.39** / 0.41 | 1.58 / **1.32** | **1.16** / 1.23 | **7.0 ms** / 6,784 ms |
| **Wine** | **0.70** / 0.80 | 3.44 / **2.35** | **1.32** / 1.67 | **4.0 ms** / 560 ms |
| **Breast Cancer** | **1.43** / 1.66 | 10.96 / **4.51** | **1.68** / 3.01 | **9.5 ms** / 1,243 ms |
| **Thyroid** | **0.16** / 0.29 | 1.90 / **1.60** | **0.25** / 0.51 | **3.0 ms** / 961 ms |
| **Shuttle** | **0.12** / 0.25 | 3.29 / **1.37** | **0.15** / 0.28 | **7.6 ms** / 1,045 ms |
| **MAGIC** | **0.25** / 0.33 | 2.56 / **1.41** | **0.26** / 0.46 | **7.8 ms** / 2,966 ms |
| **Letter** | **0.21** / 0.32 | 2.66 / **1.45** | **0.72** / 0.92 | **30.8 ms** / 10,622 ms |
| **Landsat** | **1.02** / 1.10 | 10.64 / **3.32** | **1.69** / 2.53 | **24.4 ms** / 3,767 ms |
| **Page Blocks** | **0.24** / 0.35 | 2.18 / **1.37** | **0.24** / 0.51 | **7.9 ms** / 555 ms |

---

## 7. Tuning vs. Repairing Trade-off (Study §10)

Comparison of choosing $\theta$ via 3-fold cross-validation vs. self-repair at default $\theta = 0.3$:

| Measure | CFMNN ($\theta=0.3$, Repaired) | CFMNN ($\theta$ Tuned via Validation, No Repair) |
| :--- | :---: | :---: |
| **Accuracy Comparison** | Won on 1 dataset (*Thyroid*) | **Won on 7 datasets** (up to $+17.5\%$ on *Letter*) |
| **Hyperbox Count** | **565 boxes** (*Letter*) | **5,325 boxes** (*Letter*, $9.4\times$ explosion) |
| **Provable Safety Bounds** | **Exact impact certificates** | Zero bounds (retrained from scratch) |
| **Edit Overhead** | Zero retraining; $O(1)$ local cuts | Full cross-validation grid search |

**Core Takeaway:** Practitioners seeking pure predictive accuracy should tune $\theta$ via cross-validation; practitioners requiring compact, human-auditable rule sets with localized safety bounds should utilize self-repair at $\theta = 0.3$.

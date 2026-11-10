# PACE-CFMNN: Experimental Results, Accuracy and Benchmarks

> **Reference Document:** Complete empirical compilation from the 33-page paper draft *PACE: Exact Explanations, Trust Signals and Auditable Repair for Centroid-Based Fuzzy Min-Max Classifiers* (commit `b228a29` / rev 311).
> **Data Sources:** Recomputed from committed raw CSV runs in `results/` and audited via `research_audit/CLAIMS.md`.

---

## 1. Benchmark Datasets Overview

The experimental suite spans **13 primary datasets** across synthetic, UCI, and Scikit-Learn benchmarks:

| Dataset | Domain | Samples ($N$) | Features ($d$) | Classes ($C$) | Primary Purpose in Paper |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **S1** | Synthetic | 5,000 | 10 | 2 | Additive ground truth (only $x_0, x_1, x_2$ matter; linear sum) |
| **S2** | Synthetic | 5,000 | 10 | 2 | Additive with class imbalance, label noise, shift variants |
| **S3** | Synthetic | 5,000 | 10 | 2 | Non-linear interaction ground truth (AND gate on $x_0, x_1$) |
| **Wine** | Chemical | 178 | 13 | 3 | Small tabular benchmark ($d > 12$ threshold) |
| **Breast Cancer** | Medical | 569 | 30 | 2 | High-dimensional real-world biomedical data |
| **Digits** | Image / OCR | 1,797 | 64 | 10 | Multi-class high-dimensional benchmark |
| **Thyroid** | Medical | 7,200 | 21 | 3 | Paper benchmark dataset; highly imbalanced |
| **Shuttle** | Aerospace | 58,000 | 9 | 7 | Dense aerospace sensor data (99.5% inside points) |
| **Skin** | Vision / Bio | 245,057 | 3 | 2 | Large-scale 3D RGB skin segmentation |
| **MAGIC** | Physics | 19,020 | 10 | 2 | Atmospheric Cherenkov telescope gamma rays |
| **Letter** | Image / OCR | 20,000 | 16 | 26 | Multi-class letter recognition (high geometric complexity) |
| **Landsat** | Remote Sensing| 6,435 | 36 | 6 | Satellite multi-spectral pixel classification |
| **Page Blocks** | Document AI | 5,473 | 10 | 5 | Text/graphic segmentation (heavy class imbalance) |

### Excluded Datasets (From the 2023 Literature)
* **Sepsis (Scientific Reports 2020):** Dropped due to severe label collinearity. Only 975 unique input coordinates exist for 110,204 rows, and 91.5% of rows share identical inputs with opposite mortality outcomes. Best possible accuracy equals constant majority guessing (92.65%); CFMNN scored 48.65%.
* **CoverType (581,012 rows, 54 attributes) & Crop Mapping (325,834 rows, 175 attributes):** Omitted to keep all pre-registered, multi-seed benchmarks executable within laptop computational constraints.

---

## 2. Base Model Accuracy: CFMNN vs. Standard ML Baselines

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

### Comparison to Published Literature (Aadarsh & Sai Prasad, PReMI 2023)
Comparing our single CFMNN model with the published Apache Spark ensemble:

| Dataset | Ours (Paper's Rules) (%) | Ours (With Centroid & Search Fixes) (%) | Original Published Paper Ensemble (%) |
| :--- | :---: | :---: | :---: |
| **Thyroid** | 95.0% | **97.2%** | 96.9% |
| **Shuttle** | 93.3% | **99.8%** | 94.0% |
| **Skin** | 96.9% | **94.7%** | 97.6% |

---

## 3. Interpretable / Readable Baselines Benchmark

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

## 4. Explanation Faithfulness & Speed Benchmark (RQ1)

Tested on **10 seeds (30–39)** across 50 test points per seed (36 on Wine).
* **Deletion:** Lower AUC is better (features removed in descending attribution order; faster margin collapse = higher faithfulness).
* **Insertion:** Higher AUC is better (features restored in descending order; faster margin recovery = higher faithfulness).

### 4.1 Deletion & Insertion Margin AUC Table
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

### 4.2 Statistical Significance (Pre-registered Hypotheses)
* **H1 (PACE-SHAP vs. KernelSHAP on Deletion):** PACE-SHAP is **significantly more faithful on 6 of 11 datasets** (S3, Wine, Breast Cancer, Thyroid, Letter, Landsat), level on 5, and worse on **0**.
* **H2 (PACE-SHAP vs. KernelSHAP on Insertion):** PACE-SHAP is **more faithful on 7 of 11 datasets**, level on 4, and worse on **0**.
* **H4 (PACE-SHAP vs. Strong KernelSHAP):** More faithful on 3 datasets by deletion, 4 by insertion, and worse on **0**.
* **PACE-SHAP vs. LIME:** More faithful on **9 of 11** by deletion, **10 of 11** by insertion.
* **PACE-SHAP vs. PACE-flip:** More faithful on **8 of 11** by both deletion and insertion.

### 4.3 Runtime and Speedup Benchmarks
Timing measured per explanation (batch of 50 points; Intel Core i7-8650U):

| Dataset | PACE-flip Latency | Brute-force Occlusion | PACE-flip Speedup | PACE-SHAP (Exact) | ExactExplainer (`shap`) | PACE-SHAP Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **S1** | 0.81 ms | 71.3 ms | **$88\times$** | 2.5 s | 53.7 s | **$21\times$** |
| **Wine** | 0.22 ms | 10.8 ms | **$49\times$** | 0.12 s | 3.1 s | **$26\times$** |
| **Shuttle** | 0.05 ms | 4.0 ms | **$80\times$** | 19 ms | 478 ms | **$25\times$** |
| **Landsat** | 1.12 ms | 158.0 ms | **$141\times$** | 0.9 s *(sampled)* | 41.7 s *(exact)* | **$46\times$** |

---

## 5. Trust Signals & Error Detection Benchmark (RQ2)

Area Under the Precision-Recall Curve (AUPRC) on unlabelled test data across 10 confirmatory seeds (seeds 10–19). Higher is better:

| Dataset | Baseline Error Share | Plain Margin | PD + U* (PACE) | Full PRS* (PACE) | Trust Score (Baseline) | 10-NN Disagree (Baseline) | Order Ensemble (CFMNN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Synthetic (6 sets)**| 0.17 | 0.39 | 0.41 | 0.35 | 0.55 | **0.69** | 0.65 |
| **Digits** | 0.04 | 0.39 | 0.40 | 0.19 | **0.86** | 0.82 | 0.60 |
| **Thyroid** | 0.03 | 0.26 | 0.23 | 0.23 | 0.11 | 0.09 | **0.43** |
| **Skin** | 0.05 | 0.22 | 0.42 | 0.38 | 0.98 | **0.99** | 0.75 |
| **MAGIC** | 0.28 | 0.33 | 0.44 | 0.40 | 0.69 | **0.72** | 0.59 |
| **Letter** | 0.26 | 0.41 | 0.59 | 0.56 | **0.97** | **0.97** | 0.78 |
| **Landsat** | 0.15 | 0.32 | 0.40 | 0.36 | **0.74** | 0.73 | 0.54 |
| **Page Blocks** | 0.14 | 0.19 | 0.44 | 0.37 | 0.85 | **0.86** | 0.62 |

### Key Finding on Trust:
* Model-intrinsic signals (`PD + U*`) beat the plain margin on **6 of 8 units**.
* Adding Explanation Instability (`EI`) **worsened performance on 6 of 8 units**.
* External label-based baselines ($k$-NN and Trust Score) thoroughly beat PACE signals on **7 of 8 units**.

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

* **PACE vs. DiCE Trade-off:** PACE finds counterfactuals **$70\times$ to $1,000\times$ faster** that land **closer to real data** on all 11 datasets, but DiCE changes fewer features.
* **PACE-sparse:** Modifies fewer features than DiCE on **10 of 11 datasets**, but runs slower ($13$ ms to $4.4$ s).

---

## 7. Auditable Repair & Active Learning (RQ3 & RQ4)

Evaluated across 10 confirmatory seeds (seeds 50–59) on 6 primary datasets:

### 7.1 From / To Macro-F1 Progression (%)

| Dataset | Starting Model (B0) | Self-Repair (SR, 0 labels) | Active Learning (AL) | SR then AL | CFMNN Retrained (Same Labels) | Random Forest (Same Labels) | Gap Closed to RF by SR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Thyroid** | 84.0% | **90.0%** | 90.3% | **90.5%** | 85.7% | 97.7% | **43.8%** |
| **Landsat** | 82.0% | **83.5%** | 83.9% | **84.5%** | 82.9% | 89.6% | **19.7%** |
| **Page Blocks**| 67.6% | **70.1%** | 72.4% | **72.4%** | 64.2% | 82.4% | **16.9%** |
| **MAGIC** | 72.6% | **75.9%** | 77.7% | **77.9%** | 71.1% | 86.8% | **23.2%** |
| **Letter** | 74.3% | **76.6%** | 76.3% | **77.3%** | 74.7% | 95.9% | **10.6%** |
| **Skin** | 93.1% | **98.8%** | 99.2% | **99.3%** | 91.2% | 99.9% | **83.8%** |

* **Zero-Label Self-Repair Gain:** Self-repair alone raises macro-F1 by **$+1.5$ to $+6.0$ points** across all 6 datasets without purchasing a single new label.
* **Retraining Inefficacy:** Retraining CFMNN from scratch on the newly acquired labels improved macro-F1 by only $-3.4$ to $+1.8$ points (never statistically significant).

### 7.2 Safety Audits & Certificate Precision (RQ4)

| Measure | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fix Share (SR then AL)** | **85.3%** | **64.2%** | **81.9%** | **64.8%** | **66.2%** | **94.1%** |
| **Fix Share (Retrained, Shuffled)** | 51.8% | 52.4% | 42.6% | 47.2% | 50.9% | 40.5% |
| **Broken Predictions (SR then AL)** | 0.19% | 2.36% | 0.82% | 6.85% | 3.06% | 0.22% |
| **Broken Predictions (Retrained)** | 1.03% | 6.09% | 6.81% | 15.60% | 10.48% | 3.83% |
| **Tight Certificate Size (% of Test)**| 1.2% | 1.9% | 34.6% | 7.9% | 1.5% | 4.1% |
| **Actual Flips Inside Certificate** | 0.7% | 3.4% | 0.2% | 1.9% | 3.7% | 0.7% |
| **Prediction Flips OUTSIDE Certificate**| **0** | **0** | **0** | **0** | **0** | **0** |

* **Fix Share Lead:** Repair updates are **$12$ to $54$ points higher** in fix share than retraining.
* **Certificate Invariant:** Across **46,153 accepted edits**, exactly **0 predictions flipped outside the certified impact set**.

---

## 8. Tuning vs. Repairing Trade-off (Study §10)

Comparison of choosing $\theta$ via 3-fold cross-validation vs. self-repair at default $\theta = 0.3$:

| Measure | CFMNN ($\theta=0.3$, Repaired) | CFMNN ($\theta$ Tuned via Validation, No Repair) |
| :--- | :---: | :---: |
| **Accuracy Comparison** | Won on 1 dataset (*Thyroid*) | **Won on 7 datasets** (up to $+17.5\%$ on *Letter*) |
| **Hyperbox Count** | **565 boxes** (*Letter*) | **5,325 boxes** (*Letter*, $9.4\times$ explosion) |
| **Provable Safety Bounds** | **Exact impact certificates** | Zero bounds (retrained from scratch) |
| **Edit Overhead** | Zero retraining; $O(1)$ local cuts | Full cross-validation grid search |

**Core Conclusion:** Practitioners needing pure predictive accuracy should tune $\theta$ via cross-validation; practitioners requiring compact, human-auditable rule sets with localized safety bounds should utilize self-repair at $\theta = 0.3$.


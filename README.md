# PACE-CFMNN 2.0: Exact Explanations, Trust Signals, and Auditable Repair for Centroid-Based Fuzzy Min-Max Classifiers

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-198%20passed-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/license-Research%20Only-yellow.svg)](README.md)

This repository contains the complete implementation, pre-registered benchmarks, research audit scripts, datasets, and research paper for **PACE-CFMNN 2.0**.

---

## 📁 Repository Structure

```
PACE-CFMNN-2.0/
├── paper/                                    # Research Paper & Documentation
│   ├── PACE-CFMNN paper (rev 311).pdf        # Full 33-page paper draft
│   ├── PAPER_APPLICATION_AND_METHODOLOGY.md  # Detailed system architecture, math & algorithms
│   ├── PAPER_RESULTS_AND_BENCHMARKS.md       # Comprehensive empirical tables & benchmark data
│   └── CFMNN_THEORY_REPAIR_AND_VERIFICATION.md # Contraction theory, iterations, cost & correctness proof
├── pace-cfmnn/                               # Python Package & Experimental Suite
│   ├── pace_cfmnn/                           # Core source code
│   │   ├── cfmnn_core.py                     # Centroid-based Fuzzy Min-Max Classifier
│   │   ├── explain.py                        # PACE-SHAP, PACE-flip, Contrastive explainers
│   │   ├── signals.py                        # Trust signals: PD, EI, U, U*
│   │   ├── refine.py                         # Gated shrink box repair operator & rollback gate
│   │   ├── active.py                         # Budgeted active learning & self-repair loop
│   │   ├── audit.py                          # Provable prediction impact certificates
│   │   └── counterfactual.py                 # Hyperbox counterfactuals & PACE-sparse
│   ├── data/                                 # UCI & synthetic benchmark datasets
│   ├── tests/                                # 198 unit tests covering all modules & invariants
│   ├── scripts/                              # Benchmark & experiment reproduction scripts
│   ├── results/                              # Pre-registered runs, raw CSVs & findings
│   ├── research_audit/                       # Verification scripts auditing all paper claims
│   ├── requirements.txt                      # Python dependencies
│   └── README.md                             # Package guide
└── REVIEW_GUIDE.md                           # Original peer-review package guide
```

---

## ⚡ Quick Start

### 1. Installation
Clone the repository and install dependencies (Python 3.10+):
```bash
git clone https://github.com/nikhilhanumantu/PACE-CFMNN-2.0.git
cd PACE-CFMNN-2.0/pace-cfmnn
pip install -r requirements.txt
```

### 2. Run the Test Suite
Verify that all 198 tests pass:
```bash
python -B -m unittest discover -s tests -t .
```

### 3. Basic Python Usage
```python
from pace_cfmnn import CFMNN, data
from pace_cfmnn.explain import PaceFlip, PaceShap, contrastive
from pace_cfmnn.counterfactual import counterfactual

# 1. Load dataset & normalize to [0, 1]
ds = data.load("S1")
split = data.paper_split(ds.y, seed=0)
scaler = data.fit_scaler(ds.X[split["train"]])
X_train = scaler.transform(ds.X[split["train"]])
y_train = ds.y[split["train"]]
X_test = scaler.transform(ds.X[split["test"]])

# 2. Train CFMNN classifier
model = CFMNN(theta=0.3, gamma=4.0, centroid_rule="fixed", expansion="search")
model.fit(X_train, y_train)

# 3. Exact Shapley explanations (PACE-SHAP)
explainer_shap = PaceShap(model, background=X_train[:200])
shap_values = explainer_shap.explain(X_test[:5])

# 4. Sub-millisecond decision flips (PACE-flip)
explainer_flip = PaceFlip(model, background=X_train[:200])
flip_rankings = explainer_flip.explain(X_test[:5])

# 5. Contrastive explanations ("Why Class A and not Class B?")
why = contrastive(model, X_test[:5])

# 6. Auditable Counterfactuals
cf = counterfactual(model, X_test[0], target=1 - model.predict(X_test[:1])[0],
                    X_train=X_train, y_train=y_train)
```

---

## 🔬 Core Innovations

1. **Exact Shapley Margin Values (`PACE-SHAP`):** Exploits the additive per-feature membership formula of hyperboxes to evaluate exact Shapley values for $d \le 12$ without model re-runs (**16–30× faster** than `shap.ExactExplainer`).
2. **Sub-Millisecond Occlusion (`PACE-flip`):** Computes exact single-feature decision flips over 32 background quantiles in $0.05$–$1.4$ ms (**25–134× faster** than batched occlusion).
3. **The 100% Cross-Class Tie Law:** Discovers that across all 13 benchmark datasets (347,339 errors checked), 100% of CFMNN training errors are cross-class ties between overlapping hyperboxes.
4. **Label-Free Auditable Self-Repair:** Shrinks the winning wrong-class box by $\epsilon = 0.001$, closing **10%–83% of the accuracy gap to Random Forests** without purchasing new labels.
5. **Provable Prediction Certificates:** Mathematically bounds the set of inputs whose prediction can change after an edit. Across 46,153 accepted edits, **0 predictions changed outside the certificate**.

---

## 📊 Comprehensive Documentation

* 📖 **[Application Architecture & Methodology](paper/PAPER_APPLICATION_AND_METHODOLOGY.md):** Complete mathematical formulas, hyperbox geometry, Simpson ramp equations, and certificate proofs.
* 📈 **[Experimental Results & Benchmarks](paper/PAPER_RESULTS_AND_BENCHMARKS.md):** Full accuracy comparison tables (CFMNN vs RF, 10-NN, EBM, CART, GFMM), faithfulness AUCs, speedups, and audit logs.
* 🛠️ **[Contraction Theory, Cost & Correctness Verification](paper/CFMNN_THEORY_REPAIR_AND_VERIFICATION.md):** Proof of the 100% cross-class tie law, 1-step closed-form shrink dynamics, 5 mathematical thresholds, computational complexity, and correctness proofs.
* 🔍 **[Research Claims Audit](pace-cfmnn/research_audit/CLAIMS.md):** 134 programmatic claims verified against committed raw CSV runs.

---

## 📜 Citation & License

Research code and documentation. For academic reference, see [`paper/PACE-CFMNN paper (rev 311).pdf`](paper/).


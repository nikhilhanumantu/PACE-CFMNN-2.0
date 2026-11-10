# Final review: changelog

Every change made in the final review, with where it landed. Issue IDs refer to [ISSUES.md](ISSUES.md).

## Repository (branch `final-review`)

Base: `b508703` (merge of `rq1-extensions` and `refinement-v2` into `final-review`).

| Commit | Files | Change | Issues |
|---|---|---|---|
| `0614e66` | `research_audit/CLAIMS.md` | New: claims ledger for paper rev 262 (134 rows, endpoint table) | P1–P5, W1–W11 |
| `0614e66` | `research_audit/ISSUES.md`, `research_audit/CHANGELOG.md`, `research_audit/SMOKE.md` | New: issues log, this changelog, smoke-run results | all |
| `0614e66` | `research_audit/scripts/claims_*.py`, `common.py` and their `*_out.txt` | New: read-only scripts that recompute every paper number from the committed `runs.csv` files | P1–P4 |
| `0614e66` | `research_audit/scripts/check_tie_errors.py`, `results/final_review/tie_errors/` | New post hoc check: share of training errors that are cross-class ties (100% in 18 of 18 cases) | P5 |
| `0614e66` | `research_audit/scripts/check_readme_commands.py` and its output | New: checks that every README command exists, with valid flags | D2, C3 |
| `0614e66` | `research_audit/scripts/smoke_all.py`, `smoke_redirect.py` | New: one-dataset, one-seed smoke run of every script, with outputs outside the repository | D4, C3 |
| `0614e66` | `tests/test_final_review.py` | New: 13 hand-checked tests (worked example, U and U\*, Holm, non-inferiority, fix share, gate limits, edge cases) | C1 |
| `0614e66` | `results/FINDINGS.md` | §5 rewritten for the three repair runs; §6 and §7 bullets; header note on "no difference"; P3 and P4 corrected; counterfactual wording | D1, P3, P4, W9 |
| `0614e66` | `README.md` | Status, test count (178), three repair commands, repair key results, layout rows, intro, counterfactual wording | D2, W9 |
| `0614e66` | `requirements.txt` | Comment on which scripts need shap, lime and dice-ml | D3 |

No committed result, `runs.csv`, `summary.md` or pre-registration was edited. No method, default,
hyperparameter or dataset choice was changed. The commit after `0614e66` only fills these commit ids into this file and REPORT.md.

**One unplanned write, with no change.** The first version of `check_readme_commands.py` called
`--help` on every script. Three scripts have no argument parser, so they ran in full:
- `knn_reference.py` rewrote `results/error_detection_confirmatory/knn_reference.csv` and `.md`;
- `counterfactual_table_seeds.py` rewrote `results/review_v1/counterfactual_table.md`;
- `pace_shap_settings.py` was stopped after 5 minutes, before it writes anything.

All three rewritten files are byte-identical to the committed versions (`git hash-object` against
`git rev-parse HEAD:<file>`), so nothing changed, and both outputs reproduce exactly. The checker
was then fixed (C3).

## Paper (Claude Doc `9z3UBSnVXvHC3zKSEJywSS`)

Rev 262 at the start. All edits are small find-and-replace edits through the Docs connector. The
title, RQs, contributions and section structure are unchanged. Each edit was read back at rev 281.

| Doc rev | Where | Change | Issue |
|---|---|---|---|
| 263 | §7 RQ4 table | Letter, tight certificate per edit: 1.6 → 1.5 | P1 |
| 264 | Figure 2 caption | Grey intervals excluding zero without significance: add MAGIC | P2 |
| 265 | §7 sparse counterfactuals | 86–97% → 85–97% | P3 |
| 266 | §7 trust signals, "U\* is confirmed" | Add "and not significantly better on Page Blocks" | P4 |
| 267 | §7 RQ4 paragraph | "on up to 2,000 test rows (548–2,000 per dataset)" | W1 |
| 268 | §7 RQ4 paragraph | Appended retraining "was planned as a comparison to describe, with no test" | W2 |
| 269 | Abstract | Bought labels "plus the gate's labelled acceptance rows" | W3 |
| 270 | Abstract | Retraining "helped significantly only once (Letter, by 0.85 points, in a post hoc test)" | W4 |
| 271–272 | §7 repair bullet | "Retraining on the same labels helps little", with both ranges and tests | W4 |
| 273 | §4 PACE-SHAP item 2 | "16 orders … and adds each one's reverse (32 orders in all)" | W5 |
| 274 | §4 first paragraph | "the same call on the same points always gives the same answer" | W6 |
| 275 | §5 "A correction to U" | Tie defined by equal highest membership | W7 |
| 276 | §7 conditional replacement | Diagonal scale instead of Aas, Jullum and Løland's full covariance | W8 |
| 277–279 | §7 RQ2 v2 and counterfactual bullets | "no significant difference" (three places) | W9 |
| 280 | §7 RQ4 bullet | 51% is of the explained rows outside every tight certificate | W10 |
| 281 | §8 "Boxes, prototypes and clusters" | Add Quteishat and Lim (rule extraction) | R1 |
| 282 | References | Add A. Quteishat and C. P. Lim, *Applied Soft Computing* 8(2):985–995, 2008 | R1 |

## Reframe (user request, 2026-11-06)

After the review, the user asked to reframe the paper around what is strong and true. They also pointed out that "more faithful on only 3 of 13" undersold PACE-SHAP, which is level on the rest, never worse, and faster.

The reframe:
- changes no result, method or setting;
- keeps the title, RQs and section headings;
- keeps all 6 contributions; one is reworded and moved to second place.

One number is new to the paper: matched KernelSHAP took 3.5–10× as long as PACE-SHAP. It was already in FINDINGS §4.2. It is now checked by `research_audit/scripts/check_matched_speed.py` (CLAIMS row F10b).

**Paper (doc rev 282 → 293)**

| Doc rev | Where | Change |
|---|---|---|
| 283 | Abstract ¶2 | Rewritten. PACE-SHAP computes the values KernelSHAP estimates. With the same rows its values never change between runs, and it was more stable on 10 of 11 datasets. It was never significantly less faithful. Its lead over KernelSHAP's usual setting (6 of 11) comes mostly from the setup. Like for like it was better on 3 of 13, level on 10, worse on none, and KernelSHAP took 3.5–10× as long. Dropped for length: the stronger-KernelSHAP result and the LIME and KernelSHAP speed ratios (still in §7). |
| 283 | Abstract ¶3 | Rewritten. It opens with the accuracy cost (0.2–21.5 points). Then the tie finding, self-repair (+1.5–6.0 points on all 6, winning back 10–83% of the gap to a Random Forest), bought labels, certificates and fix share. |
| 284 | §7 like-for-like bullet | "only" dropped. Adds "level on the rest" and "KernelSHAP took 3.5–10× as long on every dataset (timed while other runs shared the laptop)". |
| 285 | §1 "Our idea" | Two sentences added. We do not claim CFMNN is more accurate. Our aim is to explain, and safely fix, a model whose rules people can read. |
| 286 | Contribution 1 (PACE-SHAP) | Adds: values do not change between runs; never significantly less faithful; given the same rows, better on 3 of 13, level on the rest, 3.5–10× faster. |
| 287–288 | Contribution "Self-repair" | Label becomes "Why CFMNN errs, and self-repair with certificates". The tie finding is added to the body. |
| 289 | Contribution "Trust signals" | States the result: the model's own signals lose to standard methods that use the training labels. Order-ensemble copies add to those methods, as bootstrap copies do. |
| 290 | Conclusion | PACE-SHAP sentence: more stable, never significantly less faithful, lead mostly from the setup, and given the same rows level on most datasets and 3.5–10× faster. |
| 291 | Conclusion | Self-repair wins back 10–83% of the gap to a Random Forest. |
| 292 | Contributions | The self-repair item moves to second place. No text refers to contributions by number. |
| 293 | Abstract ¶2 | "more faithful by deletion on 3 of 13". By insertion it is 2 of 13. |

**Wording changed from the approved plan, for accuracy.** The plan said PACE-SHAP's values "do not drift between runs". They do vary when new background rows are drawn, so the text now says "with the same rows its values never change between runs" (abstract) and "do not change between runs" (contribution). The trust contribution also keeps the bootstrap caveat.

**Checks.**
- Every changed block was read back at rev 292.
- The outline shows the title, RQs and all 11 headings unchanged, and 6 contributions.
- There are no `TBD` or bracket citations.
- Every number in the new text is in CLAIMS rows A1–A16, F10, F10b or Q6.

**Repository (this commit)**

| Files | Change |
|---|---|
| `research_audit/scripts/check_matched_speed.py` and its `_out.txt` | New, read-only. KernelSHAP time over PACE-SHAP time from `extensions_v1/conditional/runs.csv`: 3.46–10.01 (ratio of means), 130 of 130 slower |
| `research_audit/CLAIMS.md` | Row F10b, and a reframe note in the summary |
| `README.md`, `results/FINDINGS.md` §8.5 | Like-for-like result: "only" dropped; level on the rest, never worse, 3.5–10× as long |
| `research_audit/CHANGELOG.md`, `research_audit/REPORT.md` | This section, and the report's paper-changes line and review notes |

No committed result, `runs.csv`, `summary.md` or pre-registration was edited, and no study was rerun.

## Readable baselines (user request, 2026-11-06 to 2026-11-07)

The user asked for a test, planned in advance, of CFMNN against other readable models, with θ chosen
by validation. The user chose to add EBM and GFMM, to use 20 seeds (70–89), to run EBM with main
effects only and GFMM untuned at θ = 0.3 (both for time), and to use six processes overnight.

**Repository**

| Commit | Files | Change |
|---|---|---|
| `42a71d4` | `scripts/run_readable_baselines.py`, `tests/test_readable_baselines.py`, `results/readable_baselines/PREREGISTRATION.md`, `requirements.txt` | New: the script, 12 tests and the pre-registration, pushed at 17:12 before the run (started 17:13:05). `interpret-core` and `hyperbox-brain` added as optional packages |
| `ba21cd3` | `results/readable_baselines/runs.csv`, `summary.md`, `run_log.txt` | The run's own output, unedited: 2,080 rows, finished 2026-11-07 04:28, exit code 0, no failures |
| this commit | `research_audit/scripts/claims_readable.py` and its `_out.txt` | New, read-only. Redoes every test with `common.paired`. Its verdict counts match the run's `summary.md` exactly |
| this commit | `research_audit/CLAIMS.md` | Rows RB1–RB15, an endpoints row and a summary note |
| this commit | `results/FINDINGS.md` | New §9; bullets in §6 and §7; a header link. In §7, "better on only 3 of 13" now reads "better on 3 of 13 … level on the rest and never worse", as the user asked after the reframe |
| this commit | `README.md` | The run's command, the key results, the optional packages, and the test count (178 → 190) |
| this commit | `research_audit/scripts/check_readme_commands_out.txt` | Rerun after the new README command: 39 of 39 exist, with valid flags |
| this commit | `research_audit/CHANGELOG.md`, `research_audit/REPORT.md` | This section and the report's section |

No pre-registration, earlier `runs.csv`, `summary.md` or paper-number file was edited. No method,
default, hyperparameter or dataset choice was changed after the push, and no study was rerun.

**Paper (doc rev 293 → 302)**

| Doc rev | Where | Change |
|---|---|---|
| 294 | §7, after the base-model tables | New paragraph "Against other readable models": setup, a 7-row table of counts, and 5 bullets (EBM strongest; θ = 0.3 not the most accurate choice; trees smaller; reference models stay ahead; macro-F1 and grid-end checks) |
| 295 | Abstract ¶3 | After the accuracy cost: an EBM was more accurate on 12 of 13 datasets and a tree on 10; θ by validation helped on 7, with up to 23× as many boxes |
| 296 | §1 "Our idea" | The base run is named as such. A later test found an EBM more accurate on 12 of 13 |
| 297–298 | Settings | Seeds 70–89 for the readable-model test; interpret-core 0.7.8 (EBM) and hyperbox-brain 0.1.6 (GFMM) |
| 299 | Deviations | "readable models" added to the list of tests planned in advance |
| 300 | Limitations, accuracy bullet | Other readable models were more accurate on most datasets; EBM had main effects only, GFMM was untuned, and several tuned models chose the end of their grid |
| 301 | Conclusion | "on every dataset we tested" → "on every dataset in our base run". At θ = 0.3, CFMNN tied the Random Forest on Wine in the new run |
| 302 | References | Added Breiman, Friedman, Olshen and Stone (CART, 1984); Lou, Caruana and Gehrke (KDD 2012); Nori, Jenkins, Koch and Caruana (InterpretML, arXiv:1909.09223, 2019). The last two were checked on the web |

**Checks.**
- Every changed block was read back at rev 302.
- The title, RQs and section headings are unchanged.
- There are no `TBD` markers or bracket citations.
- Every number in the new text is in CLAIMS rows RB1–RB15.

## Repair at the θ chosen by validation (user request, 2026-11-07 to 2026-11-08)

The readable-baselines run left a question: why repair at θ = 0.3 instead of choosing a smaller θ?
The user asked for a test planned in advance, chose 20 seeds (70–89) run overnight, and started it
on 2026-11-08 at 01:06.

**Repository**

| Commit | Files | Change |
|---|---|---|
| `22e687d` | `scripts/run_repair_theta.py`, `tests/test_repair_theta.py`, `results/repair_theta/PREREGISTRATION.md` | New: the script, 8 tests and the pre-registration, pushed on 2026-11-07 at 10:40, before the run |
| `4dd2e59` | `results/repair_theta/runs.csv`, `summary.md`, `run_log.txt` | The run's own output, unedited: 1,040 rows, finished 2026-11-08 16:10, exit code 0, no failures |
| this commit | `research_audit/scripts/claims_repair_theta.py` and its `_out.txt` | New, read-only. Redoes every test with `common.paired`. Its verdict counts match the run's `summary.md` exactly |
| this commit | `research_audit/CLAIMS.md` | Rows RT1–RT14, an endpoints row and a summary note |
| this commit | `results/FINDINGS.md` | New §10; a bullet in §6 and two in §7; a header link |
| this commit | `README.md` | The run's command, the key results, the script list, and the test count (190 → 198) |
| this commit | `research_audit/scripts/check_readme_commands_out.txt` | Rerun after the new README command: 40 of 40 exist, with valid flags |
| this commit | `research_audit/CHANGELOG.md`, `research_audit/REPORT.md` | This section, and the report's section. The report's open question is marked as now tested |

No pre-registration, earlier `runs.csv`, `summary.md` or paper-number file was edited. No method,
default, hyperparameter or dataset choice was changed after the push, and no study was rerun.

**Paper (doc rev 302 → 311)**

| Doc rev | Where | Change |
|---|---|---|
| 303, 307 | §7, bullet "θ = 0.3 was not the most accurate choice" | The new result: θ chosen by validation without repair beat repair at θ = 0.3 on 7 datasets (by up to 17.5 points), and lost only on Thyroid. With repair at both, the chosen θ was better on 8. At the chosen θ, repair helped on 3 (at most 1.6 points). Macro-F1 gave similar counts. Repair at θ = 0.3 still helped on 7 and added no box, so its case is the smaller model |
| 304 | Settings | Seeds 70–89 also for the test of repair at the chosen θ |
| 305 | Deviations | "repair at the chosen θ" added to the list of tests planned in advance |
| 306 | Limitations, accuracy bullet | Repair at θ = 0.3 did not make up for choosing θ by validation, and tried only 300 edits, a small share of the training errors on the large datasets (about 4,000 on MAGIC) |
| 308 | Limitations, repair bullet | "Repair was tested on 6 datasets with 10 seeds, one budget and one gate" → "with one budget and one gate: on 6 datasets with 10 seeds for RQ3 and RQ4, and on 13 with 20 seeds in the test of repair at the chosen θ" |
| 309 | Abstract | "On the 6 datasets we checked, every training error of CFMNN is a tie" → "On all 13 datasets, every training error of CFMNN we checked is a tie" (user request) |
| 310 | Abstract | After the self-repair sentence: "In a second planned test, it did not match choosing θ by validation, which stayed more accurate on 7 of 13 datasets, but it added no boxes." (user request) |
| 311 | §6, tie paragraph | Added: in the test of repair at the chosen θ, the tie held for all 347,339 training errors on all 13 datasets, at both θ, over 20 seeds |

**Checks.**
- Every changed block was read back at rev 311.
- The title, RQs and section headings are unchanged.
- There are no `TBD` markers or bracket citations.
- Every number in the new text is in CLAIMS rows RT1–RT14.

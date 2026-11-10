# Claims ledger: paper rev 262 against the committed results

Every number, comparison and claim in the paper (Claude Doc rev 262), traced to a committed file and recomputed from the raw `runs.csv` files, not from the rounded summaries.

- **Scripts:** `research_audit/scripts/`. These are `claims_repair.py`, `claims_rq1.py`, `claims_base_rq2.py`, `claims_figures.py`, `check_tie_errors.py` and `check_matched_speed.py`, plus the helpers in `common.py`.
- **Outputs:** each script writes a `*_out.txt` file next to it. `check_tie_errors.py` writes to `results/final_review/tie_errors/`.
- **Tests used:** the paper's own conventions. Each comparison is a paired two-sided Wilcoxon test over seeds, with Holm correction across datasets. A result counts as significant when the Holm-adjusted p is below 0.05.

**Status**

| Status | Meaning |
|---|---|
| Verified | The recomputed value matches the paper at the paper's rounding |
| Mismatch | The number or list is wrong |
| Wording | The number is right, but the sentence says more, or something different, than the evidence shows |
| Untraceable | No committed file supports it |

Each Mismatch and Wording row points to an item in `ISSUES.md`.

**Summary:**

- 134 claim rows checked. Many rows cover a whole table or list, so well over 500 single numbers were compared.
- 122 rows Verified. Four of these also carry a wording note.
- 4 rows Mismatch, all small: P1, P2, P3 and P4.
- 8 rows Wording only. With the four notes on Verified rows, these give items W1–W11 in ISSUES.md.
- 0 Untraceable. The one claim that had no backing file, "100% of training errors are ties", now has one: a new post hoc check, `results/final_review/tie_errors/`.
- Every pre-registered endpoint is reported, including the ones that went against the method. See "Pre-registered endpoints" at the end.
- **After the fixes:** all four Mismatch rows and all Wording rows except W11 (a note, no change needed) were corrected in the paper at doc revs 263–280. The rows below still show what rev 262 said; [ISSUES.md](ISSUES.md) gives the new wording and its revision.
- **Reframe (revs 283–293, user request):** the abstract, intro, contributions and conclusion were reworded. Only one number in the new text was not already in the paper: matched KernelSHAP took 3.5–10× as long as PACE-SHAP. It has its own row, F10b, and script, `check_matched_speed.py`. Every other number in the new text is in rows A1–A16, F10 or Q6.
- **Readable baselines (revs 294–302):** a new pre-registered run (seeds 70–89) added the §7 paragraph "Against other readable models", its table and 5 bullets, a sentence pair in the abstract, and a sentence in §1, Settings, Deviations, Limitations and the Conclusion, plus 3 references. Every number is in rows RB1–RB15 and is recomputed by `claims_readable.py`, whose verdict counts match the run's own `summary.md`. All 15 rows are Verified.
- **Repair at the chosen θ (revs 303–311):** a second pre-registered run on the same seeds (70–89) extended the §7 bullet "θ = 0.3 was not the most accurate choice", and added a clause to Settings, Deviations and two Limitations bullets. At the user's request (revs 309–311), the abstract now gives the result and states the tie finding for all 13 datasets, and §6 gives the new evidence for it; rows A10 and R1 keep their old wording, as the rows above do. Every number is in rows RT1–RT14 and is recomputed by `claims_repair_theta.py`, whose verdict counts match the run's own `summary.md`. All 14 rows are Verified.

## Abstract

| ID | Claim | Source | Recomputed | Status |
|---|---|---|---|---|
| A1 | PACE-SHAP equals shap's exact explainer to rounding error, 16–30× faster, for d ≤ 12 | review_v1/exact | max abs diff 3.3e-16; speed-up 16.5–30.4× | Verified |
| A2 | Above 12 features, a 1–2% error against exact values | review_v1/exact (32 orders) | Wine 0.96%, Letter 1.70% | Verified |
| A3 | More faithful than KernelSHAP on 6 datasets, never significantly less; against the stronger KernelSHAP, better on 3 | faithfulness_v3 | 6/11 deletion, 0 worse; strong: 3/11 | Verified |
| A4 | Same rows and features: more faithful on only 3 of 13 | extensions_v1/conditional (usual tests) | 3/13 deletion (S3, BC, Landsat) | Verified |
| A5 | More stable than KernelSHAP even with new background rows (10 of 11) | review_v1/faithfulness `stability_new` | 10 better, 1 level (Page Blocks) | Verified |
| A6 | PACE-flip: 0.05–1.4 ms; 25–134× faster than batched occlusion; about 90–300× faster than LIME; 300–1,200× faster than KernelSHAP | faithfulness/timing, review_v1/occlusion_batched | 0.047–1.367 ms; 25–134×; 89–316×; 292–1,194× | Verified |
| A7 | Own signals beat the plain margin but not the standard methods | error_detection_confirmatory | PD+U\* > margin on 6/8; TS and kNN > PD+U\* on 7/8; order ensemble on 8/8 | Verified |
| A8 | The order ensemble improves the combination on most datasets; bootstrap copies do as well | error_detection_v2, review_v1/ensembles | 5/8 higher; bagged vs order level on 8/8 | Verified |
| A9 | CFMNN is 0.2–21.5 points below the better of 10-NN and RF on all 13 | review_v1/accuracy | 13/13, gap 0.2–21.5 | Verified |
| A10 | Every training error is a cross-class tie on the 6 datasets | results/final_review/tie_errors (new, post hoc) | 100% at seeds 0, 10, 50 (66–10,980 errors per case) | Verified (see P5) |
| A11 | Self-repair raised macro-F1 on all 6, by 1.5–6.0 | refinement_v2_confirmatory S3a | +1.48 to +6.02, Holm p < 0.05 on 6/6 | Verified |
| A12 | "Buying up to 300 labels … added a further 0.7–2.0 points on 3" | v2 P3 | +0.73 (Letter), +1.08 (Landsat), +2.03 (MAGIC) | Verified, Wording W3 |
| A13 | "retraining CFMNN on the same labels did not help" | v2, post hoc | shuffled − start −3.4 to +1.8, never significant; appended +0.85 on Letter, significant post hoc | Wording W4 |
| A14 | 0 prediction changes outside the certificate in 46,153 edits | v2 C4 | 0 / 46,153 / 180 runs | Verified |
| A15 | Of the changed predictions, 64–94% were corrections, against 41–52% for retraining | v2 fix share | SR+AL 64.2–94.1; shuffled 40.5–52.4 | Verified |
| A16 | The explanation-chosen cut did no better than the thinnest slice | refinement_confirmatory R1 | no significant difference on any dataset | Verified |

## Sections 3 and 7: the base model

| ID | Claim | Source | Recomputed | Status |
|---|---|---|---|---|
| B1 | The centroid fix raises Shuttle from 93.3% to 99.8% | g0 | 93.33 → 99.80 | Verified |
| B2 | Expansion search: Digits 81.7 → 95.3; Skin 97.0 → 94.7 | g0 | 81.72 → 95.33; 97.01 → 94.74 | Verified |
| B3 | Sepsis: ours 48.65% (5 seeds), paper's ensemble 68.76%, always "survives" 92.65% | g0 runs; base paper value recorded in g0/summary.md | 48.65; 68.76 (quoted); majority 92.65 | Verified |
| B4 | Accuracy table: 13 rows × (training rows, 3 models with ± sd, gap) | review_v1/accuracy | all 78 cells match | Verified |
| B5 | Gap under 3 points on Shuttle, Wine, BC, Thyroid, Digits; 5.2 and 7.4 on Skin and Landsat; 9.5–21.5 on S1–S3, PB, MAGIC, Letter | same | same lists and values | Verified |
| B6 | RF has 200 trees in this table | scripts/accuracy_table.py | n_estimators=200 | Verified |
| B7 | Parity table: ours with the paper's rules 95.0 / 93.3 / 96.9; both fixes 97.2 / 99.8 / 94.7; paper 96.9 / 94.0 / 97.6 | g0 (both CFMNN columns, from runs.csv); paper column from the base paper (g0/summary.md) | all match | Verified |
| B8 | Within 2 points of the paper's ensemble on Thyroid, Shuttle, Skin | same | 1.9, 0.7, 0.7 | Verified |

## Section 4: PACE

| ID | Claim | Source | Recomputed | Status |
|---|---|---|---|---|
| M1 | Toy example: memberships 0.97 and 0.93; the 4-row table; margin drops | hand check (summary of earlier session) and `tests/` | match | Verified |
| M2 | PACE-flip skips 71–98% of boxes on all 11 datasets | speed/runs.csv + speed_uci/runs.csv | 71.1–98.2% | Verified |
| M3 | PACE-SHAP sampled mode: "samples 32 orders of the features per background row, each paired with its reverse" | pace_cfmnn/explain.py `_sampled` | 32 orders per row = 16 random orders and their 16 reverses | Wording W5 |
| M4 | "uses a fixed seed, so it gives the same answer every time" | explain.py: one generator for the whole batch | the same call gives the same answer; the same point in a different batch can get other orders | Wording W6 |
| M5 | 200 background rows (71 on Wine); at most 4,096 coalitions | code and settings | match | Verified |
| M6 | The margin "is 0 for a tied point" | cfmnn_core, explain | true for cross-class ties (the code's tie) | Verified |

## Section 5: trust signals

| ID | Claim | Source | Recomputed | Status |
|---|---|---|---|---|
| T1 | 50 nudges, 5% of each feature's standard deviation, clipped to [0, 1]; redraw with probability 0.1 if ≤ 10 distinct values | signals.py | match | Verified |
| T2 | PRS = mean percentile rank | pace_score.py | match | Verified |
| T3 | "When a point lies inside boxes of two classes (a tie)" | signals.py: a tie = top two class supports within 2·tol | the code's tie is equal top scores; inside two boxes is the common case, not the definition | Wording W7 |
| T4 | Planned U ranks ties as less uncertain than most points | error_detection/summary.md; U formula | match | Verified |

## Section 6: repair

| ID | Claim | Source | Recomputed | Status |
|---|---|---|---|---|
| R1 | Every training error is a tie inside an own-class and a wrong-class box; 100% on 6 datasets at seeds 0 and 10 | before: only PREREGISTRATION and a proposal. Now: results/final_review/tie_errors | 100% at seeds 0, 10 and 50 | Verified (P5: evidence added) |
| R2 | Shrink moves the face 0.001 past the point, on the feature with the smallest share of width cut; empty box removed | refine.py | MARGIN = 1e-3; C′-volume chooser | Verified |
| R3 | Gate: macro-F1 −0.5, class recall −1, box cap 1.1× | refine.Gate | match | Verified |
| R4 | Points tried: 300, or a quarter of the pool if smaller | run_refinement_v2 `budget = min(300, len(Xp)//4)` | 273 on Page Blocks, 300 elsewhere | Verified |
| R5 | Certificate argument (loose; tight = drop single-class groups) | audit.certify | match | Verified |
| R6 | Fix share = fixed ÷ (fixed + broken) | run_refinement_v2 | match | Verified |

## Section 7: faithfulness (RQ1)

| ID | Claim | Source | Recomputed | Status |
|---|---|---|---|---|
| F1 | 50 points (36 Wine), 10 seeds; first run 0–9; second run 30–39 | runs.csv | match | Verified |
| F2 | PACE-SHAP never significantly less faithful than any method in the second run | faithfulness_v3 | 0 worse, all methods, both tests | Verified |
| F3 | More faithful than KernelSHAP by deletion on 6 of 11 (S3, Wine, BC, Thyroid, Letter, Landsat), by insertion on 7 | same | same lists | Verified |
| F4 | Against the stronger KernelSHAP: 3 by deletion (BC, Thyroid, Landsat), 4 by insertion | same | same | Verified |
| F5 | Relative lead: 2–4% on S1–S3, Wine, Letter; 10–28% on MAGIC, Thyroid, BC; 67% on Landsat; d_z 0.6–8.6; Shuttle and PB: KernelSHAP slightly ahead, not significantly | same | 1.6–4.1%; 9.9–28.2%; 67.2%; d_z 0.61–8.63; Holm p 0.55 and 0.63 | Verified |
| F6 | Figure 2: 22 rows of mean, interval and Holm p | widget data vs faithfulness_v3 | all 22 rows match to 4 decimals | Verified |
| F7 | Figure 2 caption: the intervals that exclude zero without significance are S2 and MAGIC (blue) and S3, Wine and Letter (grey) | same | grey also includes MAGIC (upper end −0.00003) | Mismatch P2 |
| F8 | Mean deletion AUC table (66 cells) and its bold marks | faithfulness_v3 | all 66 match; bold matches | Verified |
| F9 | PACE-SHAP keeps every feature; shap keeps at most 10 by default; BC and Landsat have 30 and 36 features | code, data | match | Verified |
| F10 | Matched KernelSHAP (200 rows, all features, 200 coalitions): 3/13 deletion (S3, BC, Landsat), 2/13 insertion (BC, Landsat), never worse | extensions_v1/conditional | same | Verified |
| F10b | (Added at rev 284, also in the abstract, contribution 1 and the conclusion.) Matched KernelSHAP took 3.5–10× as long as PACE-SHAP on every dataset; timed while other runs shared the laptop | extensions_v1/conditional `ms_per_explanation`; `check_matched_speed.py` | ratio of mean times 3.46–10.01; slower in 130 of 130 dataset-seed pairs (1.16–12.14) | Verified |
| F11 | PACE-SHAP-20 never significantly less faithful than KernelSHAP | faithfulness_v3 | 0 worse | Verified |
| F12 | PACE-SHAP beats PACE-flip on 8/11 by both tests; level on Thyroid, Shuttle, PB | same | same | Verified |
| F13 | First run: KernelSHAP beat PACE-flip on 6 (S1–S3, Wine, MAGIC, Letter); PACE-flip won on 3 (BC, Thyroid; Landsat by deletion) | faithfulness | same | Verified |
| F14 | Occlusion gives PACE-flip's result up to rounding | review_v1/occlusion_batched; faithfulness | scores differ by ≤ 5.4e-15; deletion AUCs differ by ≤ 1.2e-3 in 23 of 110 dataset-seeds, where equal scores are put in a different order | Verified (note in ISSUES W11) |
| F15 | Stability: fixed seed identical; new seed: PACE-SHAP sampled 0.94–0.99, LIME 0.52–0.90, KernelSHAP 0.15–0.73 | faithfulness_v3 | same | Verified |
| F16 | Post hoc: PACE-SHAP-20 with new rows more stable than KernelSHAP on only 5/11 | same | 5/11 | Verified |
| F17 | Planned test: more stable on all 11 with fixed rows, 10/11 with new rows (level PB) | review_v1/faithfulness | same | Verified |
| F18 | Ground truth top-3: LIME 0.89–0.92, PACE-SHAP 0.79–0.85, KernelSHAP 0.72–0.81, PACE-flip 0.49–0.55 | faithfulness_v3 | same | Verified |
| F19 | LIME less faithful than PACE-SHAP on 9 of 11 | same (deletion) | 9/11 | Verified |
| F20 | 42–47% of PACE-flip's single changes involve the relevant features | faithfulness_v3/flip_mass | 41.7–47.1% | Verified |
| F21 | Ties in the second run: PB 79%, MAGIC 75%, Letter 55%, Landsat 25%, Shuttle 11% | review_v1/ties_v3.csv | 78.6, 75.0, 55.4, 24.8, 11.2 | Verified |
| F22 | Raw curves: 10/11 by deletion (level PB), 11/11 by insertion | review_v1/faithfulness | same | Verified |
| F23 | Untied points, 9 datasets with ≥ 10 in every seed: 7 by deletion, 6 by insertion; never worse | same | same | Verified |
| F24 | LIME with 200 rows less faithful on 9/11 deletion, 11/11 insertion | same | same | Verified |
| S1 | Speed table numbers: 0.05–1.4 ms; 29–141×, 89–316×, 292–1,194×; batching saves at most 2.2×; still 25–134× | faithfulness/timing; occlusion_batched | 0.047–1.367; 29–141; 89–316; 292–1,194; 1.01–2.23; 25–134 | Verified |
| S2 | Figure 3: 55 time values | widget vs faithfulness/timing | all match to 4 decimals | Verified |
| S3 | PACE-SHAP 1.2–3.0× as long as KernelSHAP on 9 of 11, less on Wine and Shuttle | faithfulness_v3/timing | 1.22–3.04 on 9; 0.43 and 0.73 | Verified |
| S4 | PACE-SHAP-20 4.3–14× faster than KernelSHAP | same | 4.35–14.47 | Verified |
| S5 | Exact: within 3.3e-16 of ExactExplainer and every-coalition KernelSHAP; 16–30× faster; S1 2.5 s vs 53.7 s; Shuttle 19 vs 478 ms | review_v1/exact | 3.33e-16; 16.5–30.4×; 2,499 vs 53,708 ms; 19.2 vs 477.7 ms | Verified |
| S6 | Sampled 32 orders: 1.0% (Wine), 1.7% (Letter); top-3 0.995 and 1.0; Letter exact 41.7 s vs 0.9 s | same | 0.96%, 1.70%; 0.9954, 1.0; 41,655 vs 909 ms | Verified |
| C1 | Conditional game: off the data at most 10% outside Shuttle, up to 67% for the usual game; Shuttle 57% and 71% | extensions_v1/conditional/realism.csv | 10.1, 67.1; 56.7, 70.6 | Verified |
| C2 | Conditional PACE-SHAP beats conditional KernelSHAP on 6/13 by both tests, never worse | conditional/runs.csv | same | Verified |
| C3 | PACE-SHAP beat its conditional version on the conditional test on 10/13 (deletion), 9/13 (insertion); Shuttle the exception (0.452 vs 0.482, 9 of 10 seeds) | same | same | Verified |
| C4 | At σ = 0.1 the game rests on about one row once half the features are kept, on 7 of 13 | extensions_v1/posthoc_review.md §2 (seed 40) | effective rows 1.1–1.2 at d/2 on S1–S3, S1c, S2c, Wine, Letter | Verified |
| C5 | σ checks: near PACE-SHAP on synthetic at σ = 1; slightly more faithful on Shuttle and MAGIC at 0.5 and 1; the noisy copy gets slightly more credit; MAGIC off 5% / 12% / 2% / 20% | conditional/posthoc_sigma.csv and .md | match | Verified |
| C6 | Four times the rows halved the loss on S1, changed little elsewhere | conditional/posthoc_rows.md | S1 loss 0.061 → 0.028 | Verified |
| C7 | "Gaussian weights as in Aas, Jullum and Løland" | explain.py `cond_scale`: per-feature standard deviation | diagonal scale, not Aas et al.'s full covariance | Wording W8 |
| L1 | On Shuttle, Letter and Landsat, 63–70% of evaluated inputs lie off the data | realism.csv `test_int` | 62.6–70.4 | Verified |

## Section 7: trust signals (RQ2)

| ID | Claim | Source | Recomputed | Status |
|---|---|---|---|---|
| E1 | Confirmatory table: 8 units × 7 columns (56 cells), bold | error_detection_confirmatory | all 56 match; bold matches | Verified |
| E2 | Excluded: Wine, BC, Shuttle, with as few as 3, 20, 39 errors | same | 3, 20, 39 | Verified |
| E3 | PD + U\* beats the margin on 6/8, all 10 seeds each; not Digits (small, ns) or Thyroid (slightly worse) | same | same | Verified |
| E4 | EI makes the combination worse on 6/8, never better | same (PRS\* vs PD+U\*) | 6 lower, 2 level | Verified |
| E5 | U\* beats U on all 6 real datasets with ≥ 1% ties, all 10 seeds | same | 6/6, 10/10 each | Verified |
| E6 | "The planned U is no better than random on MAGIC and Letter" | same | also not significantly better on Page Blocks (Holm p 0.17) | Mismatch P4 |
| E7 | TS and kNN beat PD + U\* on 7/8; order ensemble on 8/8 | same | same | Verified |
| E8 | 10-NN more accurate than CFMNN on 12 of 13 (92% vs 74% on Letter); Thyroid the exception | knn_reference.csv | 12/13; 92.2 vs 74.1 | Verified |
| E9 | Combination table: 8 × 5 (40 cells), bold | error_detection_v2 | all 40 match | Verified |
| E10 | Page Blocks 0.10 vs 0.14; two confirmatory seeds at 0.20 and 0.24 | both runs | 0.103 vs 0.141; 0.198, 0.237 | Verified |
| E11 | Order ensemble added on 5/8 in all 10 seeds; Thyroid 0.12 → 0.46; no difference on Skin and PB; lower on Digits | error_detection_v2 | 5 higher (10/10 seeds each), 2 level, Digits lower | Verified, Wording W9 ("no difference") |
| E12 | Planned ensembles test: repeated on 5 units, lower on Digits; bagged as good on 8/8; order adds to bagged on 4 (synthetic, Thyroid, MAGIC, Letter); SGD order ensemble helps only synthetic, worse on Skin | review_v1/ensembles | same | Verified |
| E13 | Strongest detector beats kNN on 8/8 and TS on 7/8 | error_detection_v2 | same | Verified |
| E14 | Geometry signals: significant only on Letter, +0.0015 | same | +0.0015, Holm p 0.016 | Verified |

## Section 7: counterfactuals, global importance and interactions (RQ1)

| ID | Claim | Source | Recomputed | Status |
|---|---|---|---|---|
| K1 | Counterfactual table, 88 cells, bold = Holm p < 0.05; both valid on 27–30 of 30 | extensions_v1/counterfactuals | all 88 match (time = median over the both-valid points); bold matches | Verified |
| K2 | Model confirms 99–100% of PACE's, 97–100% of DiCE's | same | 99.3–100; 97.3–100 | Verified |
| K3 | Closer to real data on all 11; smaller change on 5 of 8 real datasets "and the same on the other 6"; 70–1,000× faster; DiCE fewer features on 10/11 "(the same on S3)" | same | 11/11; 5 sig, 6 not significant; 70–1,043×; 10/11 | Verified, Wording W9 ("the same") |
| K4 | Nearest unlike neighbour changes 6–30 features, more than twice either method; keeps Thyroid's locks in 9% | same | 5.9–30.3; ratio ≥ 2.33; 9.3% | Verified |
| K5 | PACE-sparse: 0.1–1.1 fewer features than DiCE on 10/11; smaller change on all 11 | same | 0.13–1.14 on 10; 11/11 | Verified |
| K6 | Single feature for 86–97% of points on S1–S3, MAGIC, Letter, PB; DiCE 56–68% | same | 85.5–97.3%; DiCE 56.1–68.3% | Mismatch P3 |
| K7 | Shuttle: 0.9 more features than DiCE | same | +0.94 | Verified |
| K8 | Closer to data than DiCE on the 8 real datasets; 5–55% farther than PACE on all 11; 13 ms to 4.4 s; faster than DiCE on 10/11, 3.5× slower on BC | same | match (4.7–54.8%; 12.9 ms–4.36 s; 3.51×) | Verified |
| G1 | All rankings find the 3 relevant synthetic features | global_importance | gt_top3 = 1.0 for every method | Verified |
| G2 | PACE-flip agrees with permutation importance more than KernelSHAP on 10/11, significant on 3 (Shuttle, Letter, Landsat); MAGIC KernelSHAP ahead, ns | global_importance (seeds 0–9) | same | Verified |
| G3 | At most 11 s per dataset on average (14.3 s slowest run) vs about 3 minutes for KernelSHAP | same | 11.1 s; 14.3 s; 175.6 s | Verified |
| G4 | Limiting PACE-flip to 50 points removes about a quarter of its lead (PB) | extensions_v1/global_importance/posthoc_points_page_blocks.md | lead 0.330 → 0.244 (26%) | Verified |
| G5 | Second run: PACE-SHAP vs KernelSHAP, no significant difference on any of 11; PACE-flip lead replicated on Letter, Landsat, appeared on PB, not on Shuttle | extensions_v1/global_importance | same | Verified |
| I1 | (x0, x1) strongest pair on S3 in all 10 seeds, and on S1 and S2 too | extensions_v1/interactions | 10/10 on each | Verified |
| I2 | Figure 4: 30 per-seed leads; means 2.06, 1.62, 1.35; S3's range overlaps S1's but not S2's | widget vs runs.csv | all match (max diff 5e-5) | Verified |

## Section 7: repair (RQ3, RQ4)

| ID | Claim | Source | Recomputed | Status |
|---|---|---|---|---|
| Q1 | First run: explained cut level with the thinnest cut on all 6, worse than random on Letter; shrink beat add-only on all 6 by 0.7–8.6; order ensemble beat random choice on 3 | refinement_confirmatory | R1 none; R3 Letter; R2 0.74–8.61 on 6; S1 on 3 | Verified |
| Q2 | Exploratory: SR within half a point of 300 bought labels on 5/6; acceptance rows 548–24,506; retraining gained nothing; best cut −0.6 to +1.1 over the thinnest | refinement_v2_explore | MAGIC −2.44, others ≥ −0.5; 548–24,506; −0.57 to +1.13 | Verified |
| Q3 | Split 50/20/10/10/10; 10 seeds 50–59; margins 1 and 2 points | code, PREREGISTRATION | match | Verified |
| Q4 | Labels beyond training rows: 821–24,806 | v2 design | 273 + 548 to 300 + 24,506 | Verified |
| Q5 | RQ3 table (42 cells) | refinement_v2_confirmatory | all match | Verified |
| Q6 | SR +1.5 to +6.0 on all 6; closes 10–83% of the gap to RF | v2 S3a | +1.48 to +6.02; 10.5–82.8% | Verified |
| Q7 | SR then AL beat SR on Landsat (+1.1), MAGIC (+2.0), Letter (+0.7); Holm p = 0.049 for the first and last | v2 P3 | +1.08, +2.03, +0.73; 0.0488, 0.0117, 0.0488 | Verified |
| Q8 | SR no worse than AL (1 point) only on Letter; within half a point on Thyroid, Landsat, Skin; trails by 2.3 and 1.8 on PB and MAGIC | v2 S3b | same | Verified |
| Q9 | SR then AL beat shuffled retraining by 1.7–8.2 on all 6; appended retraining changed macro-F1 by −2.6 to +0.9 | v2 S4c; descriptive | same | Verified, Wording W4 |
| Q10 | Lowest class recall: kept within 2 points on 4/6, Thyroid +20, Skin +18; not on PB or MAGIC; SR lowered MAGIC's by 2.7 in 9 of 10 seeds | v2 S3c, S3d | +20.48, +17.87; −2.73, 9 losses | Verified |
| Q11 | RF ahead by 5–19 on 5 of 6; kNN ahead on Landsat, MAGIC, Letter, Skin, behind on PB, far behind on Thyroid | v2 | 5.01–18.53 (Skin 0.62); match | Verified |
| Q12 | "computed each accepted edit's certificates on 2,000 test rows" | run_refinement_v2 `min(2000, n_test)` | 548–2,000 rows | Wording W1 |
| Q13 | "Retraining with the new rows appended was not part of the plan" | PREREGISTRATION lines 101, 152 | it was planned, as a descriptive arm with no test | Wording W2 |
| Q14 | RQ4 table, 42 cells | v2 | 41 match; Letter tight certificate 1.546 is printed 1.6 | Mismatch P1 |
| Q15 | 0 misses in 46,153 edits (both certificates); 0 in 69,488 exploratory edits | v2 and explore | 0 / 46,153; 0 / 69,488 | Verified |
| Q16 | Tight certificate 1.2–35% of test rows; 0.2–3.7% of those changed; union 5–74% | v2 | 1.19–34.55; 0.15–3.74; 5.36–74.01 | Verified |
| Q17 | "Outside the tight certificate, PACE-flip's top 3 … changed on up to 51% of the explained rows (MAGIC)" | v2 `expl_tight_out_top3` = share of the explained rows outside the union of tight certificates | 0.51 of those rows (MAGIC) | Wording W10 |
| Q18 | Fix share beat shuffled retraining on all 6 by 12–54 points; fewer broken; fewer top-3 changes on all 5; 1.7–8.2 more accurate | v2 P4, S4a–c | +11.86 to +53.60; all significant | Verified |
| Q19 | Shuffled retraining changed 2.4–30% of predictions, appended 0.7–8% | v2 | 2.36–30.00; 0.73–8.00 | Verified |
| Q20 | Against appended retraining: higher fix share on all 6, 2.1–7.5 points more accurate, but broke more on Landsat, MAGIC, Letter | v2 | 6/6; 2.14–7.46; those 3 | Verified |
| Q21 | SR then AL changed PACE-flip's top 3 on 26–78% of explained rows | v2 | 0.26–0.78 | Verified |

## Deviations, settings, limitations, conclusion

| ID | Claim | Source | Recomputed | Status |
|---|---|---|---|---|
| D1 | Three deviations (LIME statistics in v3; conditional run restarted after 6 of 130 jobs; matched comparison missing from earlier drafts) | run logs, earlier review | match | Verified |
| D2 | Seeds per run, as listed in Settings | runs.csv seed columns | all match | Verified |
| D3 | Settings: θ, γ, KernelSHAP 200 samples, LIME 1,000, DiCE 1,000 and 5, ensemble 5, RF 100 and kNN 10 in repair, PACE-flip 100 rows | code | match | Verified |
| D4 | Software versions (Python 3.14, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, pandas 3.0.5, shap 0.54.1, lime 0.2.0.1, dice-ml 0.12) | installed environment | all match | Verified |
| L2 | Limitations repeat the numbers above (1–2%, 0.94–0.99, 63–70%, 55–79%, 0.2–21.5, 2.7, …) | as above | match | Verified |
| X1 | Conclusion repeats A1–A16 | as above | match | Verified |

## Section 7: against other readable models (revs 294–302)

Source: `results/readable_baselines/runs.csv` (pre-registered, seeds 70–89), recomputed by
`research_audit/scripts/claims_readable.py` (output in `claims_readable_out.txt`). Codes P1–P4 and
S1–S3 are the run's pre-registered comparisons.

| ID | Claim | Source | Recomputed | Status |
|---|---|---|---|---|
| RB1 | Abstract: an EBM was more accurate on 12 of 13 datasets and a tree on 10 (the explained model, θ = 0.3) | S1c, S1a | 0 / 1 / 12; 3 / 0 / 10 | Verified |
| RB2 | Abstract: choosing θ by validation helped on 7, with up to 23× as many boxes | S2; mean boxes | 7 / 6 / 0; ratios 6.24–23.21 (Skin 23.21) | Verified |
| RB3 | §1: in the base run CFMNN is less accurate than the better of NN and RF on every dataset; in a later test an EBM was more accurate on 12 of 13 | B4 (review_v1/accuracy); S1c | 13 of 13; 12 of 13 | Verified |
| RB4 | §7 table, 7 rows × 3 counts | P1, P2, P3, S1a, S1b, S1c, P4 | 5/0/8, 7/2/4, 2/2/9, 3/0/10, 4/2/7, 0/1/12, 4/3/6: all 21 match | Verified |
| RB5 | EBM: at θ = 0.3 level only on Wine, up to 14.2 points less elsewhere; tuned CFMNN beat it only on Skin and Letter | S1c, P3 | Wine level; MAGIC −14.20; Skin +2.54, Letter +7.37 | Verified |
| RB6 | θ by validation more accurate on 7, by 0.2–20.0 (Letter 20.0, Page Blocks 10.3, MAGIC 10.1), never less | S2 | +0.16 to +19.97; 19.97, 10.33, 10.12; 0 worse | Verified |
| RB7 | On 6 of these it chose θ = 0.05, the smallest tried, with 6–23× as many boxes (Letter 5,325 against 565) | settings and rules | θ = 0.05 in 13–20 of 20 seeds on Shuttle, Skin, MAGIC, Letter, Landsat, Page Blocks; 6.24–23.21×; 5,324.9 vs 564.8 | Verified |
| RB8 | The tuned tree had fewer leaves than CFMNN-0.3 had boxes on 10 of 13; a leaf tests 2.3–9.3 features; a box limits 79–100% of them | sizes | 10 (not Shuttle, Skin, Letter); 2.25–9.29; 0.79–1.00 | Verified |
| RB9 | Tuned CFMNN trailed the better of tuned kNN and RF on every dataset, by up to 11.9; at θ = 0.3 it tied the RF on Wine (98.3%) and trailed elsewhere by 0.2–21.3 | gap, per-dataset means | −11.92 to −0.05; Wine 0.000 (98.33 both); −0.21 to −21.29 | Verified |
| RB10 | Macro-F1: the same counts for the tree and LogReg; EBM (2, 3, 8), GFMM (4, 5, 4) | S3 | 5/0/8, 7/2/4, 2/3/8, 4/5/4 | Verified |
| RB11 | Several tuned models often chose the end of their grid (θ = 0.05, C = 100, k = 1) | settings | in more than 10 of 20 seeds: θ = 0.05 on 6 datasets, C = 100 on 6, k = 1 on 4 (and 1,024 leaves on Letter) | Verified |
| RB12 | Setup: 3-fold CV on the training rows; θ 0.5–0.05, 4–1,024 leaves, C 0.01–100; EBM main effects only and GFMM at θ = 0.3, both untuned, fixed before the run | PREREGISTRATION.md (pushed in 42a71d4 before run_log's start); scripts/run_readable_baselines.py | match | Verified |
| RB13 | Settings: seeds 70–89; interpret-core 0.7.8 (EBM), hyperbox-brain 0.1.6 (GFMM) | runs.csv seeds; installed versions | 20 seeds 70–89; 0.7.8, 0.1.6 | Verified |
| RB14 | Limitations: other readable models more accurate on most datasets; EBM main effects only, GFMM untuned, grid ends | RB1, RB11, RB12 | match | Verified |
| RB15 | Conclusion: "on every dataset in our base run" (was "on every dataset we tested", no longer true once the new run tied on Wine) | B4; RB9 | match | Verified |

## Section 7, abstract and §6: repair at the θ chosen by validation (revs 303–311)

Source: `results/repair_theta/runs.csv` (pre-registered, seeds 70–89), recomputed by
`research_audit/scripts/claims_repair_theta.py` (output in `claims_repair_theta_out.txt`). Codes P1,
P2 and S1–S3 are the run's pre-registered comparisons. All but the last are by accuracy.

| ID | Claim | Source | Recomputed | Status |
|---|---|---|---|---|
| RT1 | θ chosen by validation without repair was more accurate than repair at θ = 0.3 on 7 datasets, by up to 17.5 points (Letter); repair at θ = 0.3 was more accurate only on Thyroid | P1 | 1 / 5 / 7; worse by 0.09–17.53 (Letter −17.53); better: Thyroid +1.07 | Verified |
| RT2 | With repair at both θ, the chosen θ was more accurate on 8 datasets and never less | S2 | 8 / 5 / 0 | Verified |
| RT3 | At the chosen θ, repair was more accurate on 3 datasets, by at most 1.6 points, and never less | P2 | 3 / 10 / 0; Thyroid +1.55, Page Blocks +0.26, Skin +0.14 | Verified |
| RT4 | On Letter, MAGIC and Landsat the chosen θ left only 2–11 training errors on average | B0-val train_errors | Letter 5.70, MAGIC 10.95, Landsat 1.85 | Verified |
| RT5 | Macro-F1 gave similar counts | S3 | P1 1/6/6, P2 3/10/0, S1 7/6/0, S2 7/6/0 (accuracy: 1/5/7, 3/10/0, 7/6/0, 8/5/0) | Verified |
| RT6 | Repair at θ = 0.3 was still more accurate than no repair on 7 datasets and added no box | S1; boxes | 7 / 6 / 0; repair added a box in 0 of 260 pairs at θ = 0.3 (and 0 of 260 at the chosen θ) | Verified |
| RT7 | "A second test planned in advance, on the same seeds and split" | PREREGISTRATION.md, pushed in 22e687d (2026-11-07 10:40) before run_log's start (2026-11-08 01:06); B0 check | the 520 models without repair reproduce the readable-baselines run exactly (max difference 0.0) | Verified |
| RT8 | Settings: seeds 70–89 for this test | runs.csv seeds | 20 seeds, 70–89 | Verified |
| RT9 | Deviations: this test was planned in advance, with no change after the push | 22e687d; `git log` for scripts/run_repair_theta.py, pace_cfmnn/active.py, pace_cfmnn/refine.py | no commit to these files between 22e687d and the run | Verified |
| RT10 | Limitations: repair tried only 300 edits, a small share of the training errors on the large datasets (about 4,000 on MAGIC) | edits_logged; B0-0.3 train_errors | 300 used on all 20 seeds of Skin, MAGIC, Letter, Landsat and Page Blocks at θ = 0.3; MAGIC 4,048.35 on average | Verified |
| RT11 | Limitations: repair was tested with one budget and one gate, on 6 datasets with 10 seeds for RQ3 and RQ4, and on 13 with 20 seeds here | refinement_v2_confirmatory runs.csv; this run | 6 datasets, seeds 50–59; 13 datasets, seeds 70–89; budget 300 (RQ3: 273 on Page Blocks, where its pool was smaller), RQ3's gate | Verified |
| RT12 | Abstract: "On all 13 datasets, every training error of CFMNN we checked is a tie between a box of the right class and a box of the wrong class" (was "On the 6 datasets we checked") | this run's B0 rows; A10 | all 520 models without repair (13 datasets × 20 seeds × 2 θ): train_errors_in_winning_box = train_errors_own_box_full = train_errors | Verified |
| RT13 | Abstract: "In a second planned test, it did not match choosing θ by validation, which stayed more accurate on 7 of 13 datasets, but it added no boxes" | P1; boxes | 1 / 5 / 7; repair added a box in 0 of 520 models | Verified |
| RT14 | §6: "it held for all 347,339 training errors on all 13 datasets, at both θ, over 20 seeds" | this run's B0 rows | 347,339 errors over the 353 of 520 models that had any; 520 of 520 all ties | Verified |

## Pre-registered endpoints: all reported?

| Run | Endpoints in the plan | In the paper |
|---|---|---|
| Explanations v3 (seeds 30–39) | PACE-SHAP vs KernelSHAP (deletion, main); insertion; strong KernelSHAP; PACE-SHAP-20; stability; ground truth | All reported, including Shuttle and Page Blocks, where KernelSHAP is ahead (not significant) |
| Review tests (seeds 50–69) | exact values and speed; raw and untied curves; stability with new rows; LIME with 200 rows; ensembles; accuracy | All reported, including the bagged ensemble doing as well (against the order-ensemble story) |
| Conditional (B3) | conditional vs usual; matched KernelSHAP; realism; noisy copy | All reported, including "rarely more faithful" and "less credit to the noisy copy, not more" |
| Counterfactuals (B1) | PACE-sparse vs DiCE | Reported, including Shuttle's loss and BC's slower time |
| Global importance (B2), interactions (B4) | as planned | Reported, including the weak B4 test |
| Trust, confirmatory | PD+U\* vs margin; EI; U\* vs U; vs baselines | All reported, including EI failing and the baselines winning |
| Trust, combination | order ensemble added; geometry signals | Reported, including Digits being lower |
| Repair v1 (seeds 10–19) | R1 (explained cut, main), R2, R3, S1 | Reported; R1 failed and is stated |
| Repair v2 (seeds 50–59) | P3, S3a–d, P4, S4a–c, C4 | All 9 endpoints reported, including S3b holding only on Letter and S3d (MAGIC's lowest recall falls) |
| Readable baselines (seeds 70–89) | P1–P4, S1–S3; descriptive gap, settings, sizes, times | All reported in §7, including CFMNN losing to EBM on most datasets, to the tree at θ = 0.3, and θ = 0.3 being a poor choice on 7 datasets. Times are in FINDINGS §9 only |
| Repair at the chosen θ (seeds 70–89) | P1, P2, S1, S2, S3; descriptive boxes, errors, edits, fixed and broken, gap, times | P1, P2, S1, S2 and S3 reported in §7, including repair at θ = 0.3 losing to the chosen θ on 7 datasets. The descriptive items are in FINDINGS §10, and the edit budget in Limitations |

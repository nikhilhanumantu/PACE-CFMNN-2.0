# PACE-CFMNN: review package

This is a snapshot of the `final-review` branch at commit `b228a29`, the same code as the open pull
request into main: https://github.com/Srujankasturi/pace-cfmnn/pull/18

## What is inside

| Path | What it is |
|---|---|
| `paper/PACE-CFMNN paper (rev 311).pdf` | The paper as it stands (33 pages) |
| `pace-cfmnn/` | The code, tests, results and audit files at `b228a29` (no `.git` folder) |
| `pace-cfmnn/data/` | The UCI data files, so everything runs as is. They are not in git; `data/README.md` says where each came from |

## Where to start

1. **The paper.** RQ1–RQ4 are in Section 1, and the results in Section 7.
2. **`pace-cfmnn/results/FINDINGS.md`.** Section 6 lists what we can claim, and Section 7 what we must not. Sections 9 and 10 are the two newest studies.
3. **`pace-cfmnn/research_audit/REPORT.md`.** The final review, and how each later study changed it.
4. **`pace-cfmnn/research_audit/CLAIMS.md`.** Every number in the paper, with the file it comes from.

## New since the last review round

- **Readable baselines** (FINDINGS §9; seeds 70–89, planned in advance). CFMNN against a decision tree, logistic regression, an EBM and GFMM, and with θ chosen by validation. At θ = 0.3, the EBM is more accurate on 12 of 13 datasets and the tree on 10.
- **Repair at the θ chosen by validation** (FINDINGS §10; seeds 70–89, planned in advance).
  - Choosing θ by validation, without repair, beats repair at θ = 0.3 on 7 of 13 datasets.
  - At the chosen θ, repair adds at most 1.6 points.
  - Repair never adds a box, while the smaller θ needs 6–23× as many.
- **Ties.** Every training error checked is a tie between a box of its own class and a box of a wrong class, now on all 13 datasets.
- **The paper's abstract, §6, §7, Settings, Deviations and Limitations** were updated to match (revs 294–311).

## Running it

Python 3.14 was used. Package versions are in the paper's Settings paragraph.

```
cd pace-cfmnn
pip install -r requirements.txt
python -B -m unittest discover -s tests -t .
```

The suite has 198 tests and takes about 2 minutes. All of them pass on this snapshot.

To recheck the paper's numbers from the committed results, run any script in
`research_audit/scripts/`, for example:

```
python -B research_audit/scripts/claims_repair_theta.py
```

They only read. Each one's `_out.txt` file shows what it printed for us.

The full experiments take hours: the last two took 11 and 15 h on a laptop. Their results are
already in `results/`, so there is no need to rerun them. The README lists every command and its
run time.

## Useful things to check

- **Claims against evidence.** Does each claim in the paper match its source in `CLAIMS.md`? Is anything said more strongly than the results allow?
- **Pre-registrations.** Each planned run has a `PREREGISTRATION.md` in its `results/` folder. Was each run done as planned? The deviations are listed in the paper.
- **The two new studies.** Is "the case for θ = 0.3 is a smaller model, not accuracy" a fair reading of §9 and §10?
- **Known weak points.** Is each one stated clearly enough?
  - CFMNN is less accurate than the reference models and most readable models.
  - The certificates are exact but loose.
  - Several tuned models chose the end of their grid.
  - Repair used a 300-edit budget.
- **Code.** The core is in `pace_cfmnn/`, and the experiments are in `scripts/`.

Comments are easiest as review comments on pull request #18, or as notes in a file sent back.

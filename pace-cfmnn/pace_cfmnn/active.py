"""Budgeted active-learning loop for selective refinement (blueprint 7.1).

The pool's labels sit behind a ``LabelOracle`` and are revealed only for the
points a selector picks. Each round:

1. score the remaining pool with the current model and pick the top b points
   (ties broken at random, seeded);
2. reveal their labels;
3. for each point, try its candidate edits in order: the shrink (if the arm
   has one and the point is misclassified inside a wrong-class box), then the
   plain add. Each is gated on the acceptance rows; a failed edit is rolled
   back and the next is tried. A point whose edits all fail is labelled but
   left out of the model.

Test rows, if given, are only measured after each round for the learning
curve. Nothing in the loop reads them.

Selectors (higher = picked first):

* ``random``;
* ``margin``: the plain margin uncertainty 1 - (s'_(1) - s'_(2));
* ``pd_u_star``: PD + U*, percentile ranks fitted on the inner rows with the
  current model (EI is skipped; it adds nothing, see RQ2);
* ``order_ensemble``: the share of 5 CFMNNs, trained on the current labelled
  data in other orders, that disagree with the current model;
* ``knn`` (refinement v2): the share of the 10 nearest labelled rows whose
  label differs from the prediction (RQ2's strongest single detector);
* ``oracle_errors`` (refinement v2, reference only): reads the pool's hidden
  labels and picks the points the model gets wrong. It is an upper bound on
  what any selector could do, not a method.

``run_self_repair`` (refinement v2) runs the shrink on the training rows the
model gets wrong. Their labels are already known, so it buys no new labels.

Every round log counts labels: ``pool_labels`` bought from the oracle,
``gate_labels`` read by the gate (the acceptance rows, reused for every edit),
and ``total_labels``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from .cfmnn_core import CFMNN
from .evaluate import classification_metrics
from .neighborhood import Perturber
from .pace_score import PRS
from .refine import CHOOSERS, Gate, shrink_box, shrink_target, validate_structure
from .signals import SignalComputer, uncertainty
from .trust_baselines import knn_disagreement, order_ensemble_disagreement

# Arm name -> feature chooser for the shrink (None = add only). "oracle" tries
# every feature through the gate and keeps the best (an upper bound).
OPERATIONS = {"A+B": None, "A+B+C": "explained", "A+B+C'-random": "random",
              "A+B+C'-volume": "volume", "A+B+C-fewest": "fewest",
              "A+B+C-paceshap": "pace_shap", "A+B+C-oracle": "oracle"}


class LabelOracle:
    """Holds the pool's labels; reveals each one once, and logs every read."""

    def __init__(self, y):
        self._y = np.asarray(y)
        self._log: list[int] = []

    def reveal(self, idx) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64)
        if len(np.unique(idx)) != len(idx) or np.isin(idx, self._log).any():
            raise ValueError("each pool label may be revealed only once")
        self._log.extend(idx.tolist())
        return self._y[idx].copy()

    @property
    def revealed(self) -> np.ndarray:
        return np.asarray(self._log, dtype=np.int64)

    def peek_for_reference(self, idx) -> np.ndarray:
        """Hidden labels for the ``oracle_errors`` reference selector only. Not a purchase."""
        return self._y[np.asarray(idx, dtype=np.int64)].copy()


@dataclass
class Context:
    """What a selector may see. No pool labels, no acceptance or test rows.

    ``y_cand_hidden`` is filled only for the ``oracle_errors`` reference arm.
    """

    model: CFMNN
    X_cand: np.ndarray
    X_lab: np.ndarray
    y_lab: np.ndarray
    X_inner: np.ndarray
    perturber: Perturber | None
    rng: np.random.Generator
    seed: int
    y_cand_hidden: np.ndarray | None = None


def select_random(ctx: Context) -> np.ndarray:
    return ctx.rng.uniform(size=len(ctx.X_cand))


def select_margin(ctx: Context) -> np.ndarray:
    return uncertainty(ctx.model, ctx.model.predict_details(ctx.X_cand))[2]


def select_pd_u_star(ctx: Context) -> np.ndarray:
    sc = SignalComputer(ctx.model, ctx.perturber, ei=False)
    prs = PRS().fit(sc.compute(ctx.X_inner, seed=10_000 + ctx.seed))
    return prs.score(sc.compute(ctx.X_cand, seed=ctx.seed), use=("pd", "u_star"))[0]


def select_order_ensemble(ctx: Context) -> np.ndarray:
    pred = ctx.model.predict(ctx.X_cand)
    return order_ensemble_disagreement(ctx.model, ctx.X_lab, ctx.y_lab, ctx.X_cand, pred,
                                       n_models=5, seed=ctx.seed)


def select_knn(ctx: Context) -> np.ndarray:
    pred = ctx.model.predict(ctx.X_cand)
    return knn_disagreement(ctx.X_lab, ctx.y_lab, ctx.X_cand, pred, k=10)


def select_oracle_errors(ctx: Context) -> np.ndarray:
    if ctx.y_cand_hidden is None:
        raise ValueError("oracle_errors needs the hidden pool labels")
    return (ctx.model.predict(ctx.X_cand) != ctx.y_cand_hidden).astype(float)


SELECTORS = {"random": select_random, "margin": select_margin, "pd_u_star": select_pd_u_star,
             "order_ensemble": select_order_ensemble, "knn": select_knn,
             "oracle_errors": select_oracle_errors}


def top_b(scores: np.ndarray, b: int, rng: np.random.Generator) -> np.ndarray:
    """Positions of the b highest scores; ties broken by a seeded random order."""
    order = rng.permutation(len(scores))
    return order[np.argsort(-np.asarray(scores)[order], kind="stable")][:b]


def _oracle_cut(snap: CFMNN, k: int, x, t, row: int, X_lab, gate: Gate, before: dict):
    """Try the shrink on every feature (each followed by the add) and keep the gate's best.

    The best accepted trial has the highest acceptance macro-F1, then the
    smallest share of width cut, then the lowest feature index. Returns
    (model or None, info, metrics after or None).
    """
    best = None
    for i in range(snap.n_features_in_):
        trial = copy.deepcopy(snap)
        info = shrink_box(trial, k, x, i, X_lab)
        trial.add_points(x, [t], rows=[row])
        ok, _, after = gate.check(before, trial)
        if ok:
            key = (after["macro_f1"], -info["share_cut"], -i)
            if best is None or key > best[0]:
                best = (key, trial, info, after)
    if best is None:
        return None, {}, None
    return best[1], best[2], best[3]


def run_refinement(model: CFMNN, X_train, y_train, X_pool, oracle: LabelOracle, X_inner,
                   X_acc, y_acc, selector: str, operation: str = "A+B", budget: int = 300,
                   n_rounds: int = 10, seed: int = 0, perturber: Perturber | None = None,
                   X_test=None, y_test=None, box_cap_factor: float = 1.1,
                   max_f1_drop: float = 0.5, max_recall_drop: float = 1.0,
                   background=None, on_accept=None):
    """Run one arm. ``model`` must be fitted on ``X_train`` in this row order; it is not changed.

    ``background`` holds reference rows for the ``pace_shap`` chooser.
    ``on_accept(before, after, removed)`` is called after every accepted edit,
    with the model before and after it and the index of a box the shrink
    deleted (or None); ``audit.impact_set`` uses it.

    Returns (refined model, per-round log, per-edit log).
    """
    if selector not in SELECTORS or operation not in OPERATIONS:
        raise ValueError("unknown selector or operation")
    if selector == "pd_u_star" and perturber is None:
        raise ValueError("pd_u_star needs a fitted Perturber")
    model = copy.deepcopy(model)
    chooser = OPERATIONS[operation]
    rng = np.random.default_rng(seed)
    X_lab = np.array(X_train, dtype=float)
    y_lab = np.asarray(y_train).copy()
    X_pool = np.asarray(X_pool, dtype=float)
    validate_structure(model, X_lab, y_lab)
    gate = Gate(np.asarray(X_acc, float), np.asarray(y_acc), int(np.floor(box_cap_factor * model.n_boxes_)),
                max_f1_drop, max_recall_drop)
    before = gate.measure(model)
    b = int(np.ceil(budget / n_rounds))
    remaining = np.arange(len(X_pool))
    rounds, edits = [], []
    n_gate = len(np.asarray(y_acc))

    def log_round(r: int, used: int) -> None:
        rec = {"round": r, "labels": used, "pool_labels": used, "gate_labels": n_gate,
               "total_labels": used + n_gate, "boxes": model.n_boxes_,
               "accept_macro_f1": before["macro_f1"]}
        if X_test is not None:
            rec.update({f"test_{k}": v for k, v in
                        classification_metrics(y_test, model.predict(X_test)).items()})
        rounds.append(rec)

    log_round(0, 0)
    used = 0
    for r in range(1, n_rounds + 1):
        take = min(b, budget - used, len(remaining))
        if take <= 0:
            break
        ctx = Context(model, X_pool[remaining], X_lab, y_lab, np.asarray(X_inner, float),
                      perturber, rng, seed * 1000 + r)
        if selector == "oracle_errors":
            ctx.y_cand_hidden = oracle.peek_for_reference(remaining)
        pick = remaining[top_b(SELECTORS[selector](ctx), take, rng)]
        labels = oracle.reveal(pick)
        remaining = remaining[~np.isin(remaining, pick)]
        for x, t, p in zip(X_pool[pick], labels, pick):
            row = len(X_lab)
            X_lab = np.vstack([X_lab, x])
            y_lab = np.append(y_lab, t)
            det = model.predict_details(x[None])
            t_idx = int(np.searchsorted(model.classes_, t))
            wrong = int(det.cls[0]) != t_idx
            k = shrink_target(model, x, t_idx, det) if wrong else -1
            cands = (["shrink"] if (chooser and k >= 0) else []) + ["add"]
            for cand in cands:
                snap = copy.deepcopy(model)
                info = {}
                if cand == "shrink" and chooser == "oracle":
                    trial, info, after = _oracle_cut(snap, k, x, t, row, X_lab, gate, before)
                    ok, reason = trial is not None, "" if trial is not None else "no feature passes"
                    if ok:
                        model = trial
                else:
                    if cand == "shrink":
                        i = CHOOSERS[chooser](model, x, k, int(det.cls[0]), t_idx, rng,
                                              X_lab=X_lab, background=background)
                        info = shrink_box(model, k, x, i, X_lab)
                    model.add_points(x, [t], rows=[row])
                    ok, reason, after = gate.check(before, model)
                edits.append({"round": r, "pool_index": int(p), "edit": cand, "was_wrong": wrong,
                              "shrink_possible": k >= 0, "accepted": ok, "reason": reason,
                              "x_correct_after": bool(model.predict(x[None])[0] == t),
                              **{f"shrink_{a}": v for a, v in info.items()}})
                if ok:
                    before = after
                    if on_accept is not None:
                        on_accept(snap, model, info.get("box") if info.get("removed") else None)
                    break
                model = snap
            used += 1
        validate_structure(model, X_lab, y_lab)
        log_round(r, used)
    return model, rounds, edits


def run_self_repair(model: CFMNN, X_train, y_train, X_gate, y_gate, chooser: str = "volume",
                    budget: int = 300, n_rounds: int = 10, seed: int = 0, X_test=None, y_test=None,
                    box_cap_factor: float = 1.1, max_f1_drop: float = 0.5,
                    max_recall_drop: float = 1.0, gate_labels_are_new: bool = True,
                    background=None, on_accept=None):
    """Repair the model on its own training errors: no new labels are bought.

    Every training row the model gets wrong lies inside a box of its own class
    and inside a box of the wrong class, and the centroid rule picks the wrong
    one (checked on the v1 datasets). So the shrink can be applied to these
    rows directly: move one face of the winning wrong-class box past the row,
    then gate the edit. The row is already a member of its own box, so nothing
    is added.

    Up to ``budget`` training errors are tried, in a seeded random order, in
    ``n_rounds`` rounds; a row the model already gets right by its turn is
    skipped. The gate reads ``X_gate``, ``y_gate``: the acceptance rows, or the
    training rows (pass ``gate_labels_are_new=False``, so no labels are counted).

    Returns (repaired model, per-round log, per-edit log).
    """
    if chooser not in CHOOSERS:
        raise ValueError("unknown chooser")
    model = copy.deepcopy(model)
    rng = np.random.default_rng(seed)
    X_lab = np.array(X_train, dtype=float)
    y_lab = np.asarray(y_train).copy()
    validate_structure(model, X_lab, y_lab)
    gate = Gate(np.asarray(X_gate, float), np.asarray(y_gate),
                int(np.floor(box_cap_factor * model.n_boxes_)), max_f1_drop, max_recall_drop)
    before = gate.measure(model)
    t_all = np.searchsorted(model.classes_, y_lab)
    errors = np.flatnonzero(model.predict_details(X_lab).cls != t_all)
    order = rng.permutation(errors)[:budget]
    n_gate = len(np.asarray(y_gate)) if gate_labels_are_new else 0
    rounds, edits = [], []

    def log_round(r: int, tried: int) -> None:
        rec = {"round": r, "labels": 0, "pool_labels": 0, "gate_labels": n_gate,
               "total_labels": n_gate, "train_errors_tried": tried, "boxes": model.n_boxes_,
               "accept_macro_f1": before["macro_f1"]}
        if X_test is not None:
            rec.update({f"test_{k}": v for k, v in
                        classification_metrics(y_test, model.predict(X_test)).items()})
        rounds.append(rec)

    log_round(0, 0)
    tried = 0
    for r, chunk in enumerate(np.array_split(order, n_rounds), start=1):
        for p in chunk:
            tried += 1
            x, t_idx = X_lab[p], int(t_all[p])
            det = model.predict_details(x[None])
            rec = {"round": r, "train_index": int(p), "edit": "shrink"}
            if int(det.cls[0]) == t_idx:
                edits.append({**rec, "edit": "none", "accepted": False, "reason": "already right"})
                continue
            k = shrink_target(model, x, t_idx, det)
            if k < 0:
                edits.append({**rec, "edit": "none", "accepted": False, "reason": "not inside the winning box"})
                continue
            snap = copy.deepcopy(model)
            i = CHOOSERS[chooser](model, x, k, int(det.cls[0]), t_idx, rng,
                                  X_lab=X_lab, background=background)
            info = shrink_box(model, k, x, i, X_lab)
            ok, reason, after = gate.check(before, model)
            edits.append({**rec, "accepted": ok, "reason": reason,
                          "x_correct_after": bool(model.predict(x[None])[0] == y_lab[p]),
                          **{f"shrink_{a}": v for a, v in info.items()}})
            if ok:
                before = after
                if on_accept is not None:
                    on_accept(snap, model, info["box"] if info["removed"] else None)
            else:
                model = snap
        validate_structure(model, X_lab, y_lab)
        log_round(r, tried)
    return model, rounds, edits

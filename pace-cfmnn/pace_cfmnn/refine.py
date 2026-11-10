"""Box edits and the acceptance gate for selective refinement (blueprint 7.2-7.3).

Operations on a labelled point x with true class t:

* A/B, add: ``CFMNN.add_points``, the model's own training rule (absorb into a
  same-class box under the size limit, otherwise a new point box).
* C, explanation-guided shrink: when x is misclassified and lies inside the
  winning wrong-class box k, move one face of k just past x on one feature,
  then add x. C picks the feature with the largest contrastive term e_i of
  "why the wrong class, not t" (``explain.contrastive_pair``).
* C', control shrink: the same edit with the feature chosen at random, or by
  the smallest share of the box's width cut ("volume").
* Exploratory choosers (refinement v2): "fewest" drops the fewest training
  members of the box; "pace_shap" takes the largest PACE-SHAP attribution to
  the wrong prediction. ``active.py`` also has an "oracle" cut that tries every
  feature through the gate (an upper bound, not a method).

After a shrink, box k keeps only the training members still inside it; its
Sum, Count and centroid are recomputed from them, and a box left empty is
removed. Only the main model's fixed centroid rule is supported: with the
paper's rule, Sum and Count do not match the members.

Every edit is checked by ``Gate`` on the acceptance rows and rolled back if
it fails. ``validate_structure`` checks the model's invariants; a failure
there is a bug, not a rejection, so it raises.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cfmnn_core import CFMNN
from .evaluate import classification_metrics
from .explain import contrastive_pair

MARGIN = 1e-3


def _require_fixed(model: CFMNN) -> None:
    if model.centroid_rule != "fixed":
        raise ValueError("refinement needs centroid_rule='fixed' (Sum/Count must match the members)")


def contains(model: CFMNN, k: int, x: np.ndarray) -> bool:
    """True if box k contains x (membership 1)."""
    return bool((model.V_[k] <= x).all() and (x <= model.W_[k]).all())


def shrink_target(model: CFMNN, x: np.ndarray, true_idx: int, det=None) -> int:
    """The winning box if it is of a wrong class and contains x, else -1."""
    det = det if det is not None else model.predict_details(x[None])
    k = int(det.winner[0])
    if int(det.cls[0]) == int(true_idx) or not contains(model, k, x):
        return -1
    return k


def face_cut(model: CFMNN, k: int, x: np.ndarray, i: int, margin: float = MARGIN):
    """Which face of box k to move on feature i, its new value, and the share of width cut.

    The face nearer to x moves just past x, so x falls outside the box. A share
    of 1 (or more, capped at 1) means the box would be cut away entirely.
    """
    v, w, xi = model.V_[k, i], model.W_[k, i], x[i]
    width = w - v
    if xi - v <= w - xi:
        new, side = xi + margin, "low"
        share = (new - v) / width if width > 0 else 1.0
    else:
        new, side = xi - margin, "high"
        share = (w - new) / width if width > 0 else 1.0
    return side, float(new), float(min(share, 1.0))


def shrink_box(model: CFMNN, k: int, x: np.ndarray, feature: int, X_lab: np.ndarray,
               margin: float = MARGIN) -> dict:
    """Move one face of box k past x on ``feature`` (in place). Returns what changed."""
    _require_fixed(model)
    side, new, share = face_cut(model, k, x, feature, margin)
    members = model.members_[k]
    if side == "low":
        gone = new > model.W_[k, feature]
        keep = members[X_lab[members, feature] >= new] if not gone else members[:0]
    else:
        gone = new < model.V_[k, feature]
        keep = members[X_lab[members, feature] <= new] if not gone else members[:0]
    info = {"box": int(k), "feature": int(feature), "side": side, "share_cut": share,
            "n_cut": int(len(members) - len(keep)), "removed": bool(len(keep) == 0)}
    if len(keep) == 0:
        model._remove_box(k)
        return info
    if side == "low":
        model.V_[k, feature] = new
    else:
        model.W_[k, feature] = new
    model.members_[k] = keep
    model.sum_[k] = X_lab[keep].sum(axis=0)
    model.count_[k] = len(keep)
    model.centroid_[k] = model.sum_[k] / len(keep)
    return info


# ------------------------------------------------------------ feature choice
# Every chooser takes (model, x, k, wrong_idx, true_idx, rng) and may use the
# keyword arguments X_lab (the labelled rows that members_ index) and
# background (reference rows for PACE-SHAP); the others ignore them.
def feature_explained(model, x, k, wrong_idx, true_idx, rng=None, **_) -> int:
    """C: the largest contrastive term of 'why the wrong class, not the true one'."""
    c = contrastive_pair(model, x[None], np.array([wrong_idx]), np.array([true_idx]))
    return int(np.argmax(c.e[0]))


def feature_random(model, x, k, wrong_idx, true_idx, rng=None, **_) -> int:
    """C' control: a random feature."""
    return int(rng.integers(model.n_features_in_))


def feature_volume(model, x, k, wrong_idx, true_idx, rng=None, **_) -> int:
    """C' control: the feature where the cut removes the smallest share of the box's width."""
    shares = [face_cut(model, k, x, i)[2] for i in range(model.n_features_in_)]
    return int(np.argmin(shares))


def members_cut(model: CFMNN, k: int, x: np.ndarray, i: int, X_lab: np.ndarray) -> int:
    """How many training members of box k the cut on feature i would drop."""
    side, new, _ = face_cut(model, k, x, i)
    vals = X_lab[model.members_[k], i]
    return int(np.sum(vals < new) if side == "low" else np.sum(vals > new))


def feature_fewest_members(model, x, k, wrong_idx, true_idx, rng=None, X_lab=None, **_) -> int:
    """Data-aware control: the feature whose cut drops the fewest training members of box k.

    Ties go to the smaller share of width cut, then to the lower feature index.
    """
    if X_lab is None:
        raise ValueError("feature_fewest_members needs X_lab")
    d = model.n_features_in_
    key = [(members_cut(model, k, x, i, X_lab), face_cut(model, k, x, i)[2], i) for i in range(d)]
    return int(min(key)[2])


def feature_pace_shap(model, x, k, wrong_idx, true_idx, rng=None, background=None, **_) -> int:
    """C with PACE-SHAP: the feature that pushes the (wrong) prediction's margin up the most.

    PACE-SHAP explains the predicted class's margin over the runner-up, which
    is the wrong class here, so the largest attribution is the feature most
    responsible for the error.
    """
    from .explain import PaceShap   # local import: explain imports nothing from here
    if background is None:
        raise ValueError("feature_pace_shap needs background rows")
    phi = PaceShap(model, background, seed=0).explain(x[None])[0]
    return int(np.argmax(phi))


CHOOSERS = {"explained": feature_explained, "random": feature_random, "volume": feature_volume,
            "fewest": feature_fewest_members, "pace_shap": feature_pace_shap}


# ------------------------------------------------------------------- the gate
@dataclass
class Gate:
    """Accept an edit only if the acceptance rows show no meaningful harm (blueprint 7.3).

    Thresholds are in percentage points: macro-F1 may fall by at most
    ``max_f1_drop`` and no class's recall by more than ``max_recall_drop``,
    each compared with the model before the edit. The box count must stay at
    or below ``box_cap``.
    """

    X: np.ndarray
    y: np.ndarray
    box_cap: int
    max_f1_drop: float = 0.5
    max_recall_drop: float = 1.0

    def measure(self, model: CFMNN) -> dict:
        return classification_metrics(self.y, model.predict(self.X))

    def check(self, before: dict, model: CFMNN):
        """(accepted, reason, metrics after the edit)."""
        if model.n_boxes_ > self.box_cap:
            return False, "box cap", None
        after = self.measure(model)
        if before["macro_f1"] - after["macro_f1"] > self.max_f1_drop + 1e-9:
            return False, "macro-F1", after
        for key, val in before.items():
            if key.startswith("recall_") and val - after[key] > self.max_recall_drop + 1e-9:
                return False, "class recall", after
        return True, "", after


def validate_structure(model: CFMNN, X_lab: np.ndarray, y_lab: np.ndarray) -> None:
    """Raise if any invariant is broken: bounds, Sum/Count/centroid vs members, member classes."""
    _require_fixed(model)
    if not ((model.V_ <= model.W_).all() and (model.V_ >= 0).all() and (model.W_ <= 1).all()):
        raise ValueError("box bounds are invalid")
    if len(model.members_) != model.n_boxes_:
        raise ValueError("members_ does not match the boxes")
    seen = np.concatenate(model.members_) if model.n_boxes_ else np.array([], int)
    if len(np.unique(seen)) != len(seen):
        raise ValueError("a training row belongs to two boxes")
    for j, m in enumerate(model.members_):
        P = X_lab[m]
        if model.count_[j] != len(m) or len(m) == 0:
            raise ValueError(f"box {j}: count does not match its members")
        if not np.allclose(model.sum_[j], P.sum(axis=0), atol=1e-8):
            raise ValueError(f"box {j}: sum does not match its members")
        if not np.allclose(model.centroid_[j], model.sum_[j] / model.count_[j], atol=1e-12):
            raise ValueError(f"box {j}: centroid is stale")
        if not ((P >= model.V_[j]).all() and (P <= model.W_[j]).all()):
            raise ValueError(f"box {j}: a member lies outside the box")
        if not (y_lab[m] == model.classes_[model.box_class_[j]]).all():
            raise ValueError(f"box {j}: a member has another class")
    for c, idx in enumerate(model.class_boxes_):
        if not np.array_equal(idx, np.flatnonzero(model.box_class_ == c)):
            raise ValueError("class_boxes_ is stale")

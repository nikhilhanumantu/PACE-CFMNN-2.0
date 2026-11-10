"""Auditing box edits: which predictions an edit can change, and what did change.

CFMNN predicts with the box of highest membership; boxes within ``tie_tol`` of
the top are tied and the nearest centroid wins (``CFMNN.predict_details``). An
edit changes only some boxes (a shrink changes one box or removes it; an add
grows one box or appends a new one). So the prediction at x can change only if
a changed box is in x's tied top group before the edit or after it: otherwise
the top value, the tied group and the centroids that decide it are all the
same. ``impact_set`` returns that set of rows: a certificate, computed from the
two models alone, that contains every prediction the edit changed.

The tight certificate (``tight=True``) also drops the rows whose top groups,
before and after, hold boxes of one class only: the prediction is that class
both times. It is just as safe and usually much smaller.
"""

from __future__ import annotations

import numpy as np

from .cfmnn_core import CFMNN, feature_terms


def changed_boxes(before: CFMNN, after: CFMNN, removed: int | None = None):
    """Indices of changed boxes, in ``before`` and in ``after``.

    ``removed`` is the index (in ``before``) of a box the edit deleted, if any.
    Boxes keep their order, removal shifts later boxes up by one, and new boxes
    are appended (``CFMNN._remove_box``, ``CFMNN.add_points``).
    """
    kept = [i for i in range(before.n_boxes_) if i != removed]
    if after.n_boxes_ < len(kept):
        raise ValueError("after has fewer boxes than the edit can explain")
    ch_before = [] if removed is None else [int(removed)]
    ch_after = []
    for a, b in enumerate(kept):
        same = (np.array_equal(before.V_[b], after.V_[a]) and np.array_equal(before.W_[b], after.W_[a])
                and np.array_equal(before.centroid_[b], after.centroid_[a])
                and before.box_class_[b] == after.box_class_[a])
        if not same:
            ch_before.append(b)
            ch_after.append(a)
    ch_after += list(range(len(kept), after.n_boxes_))
    return np.array(ch_before, dtype=np.int64), np.array(ch_after, dtype=np.int64)


def top_groups(model: CFMNN, X):
    """The tied top group of every row, shape (n, H), and the predicted class index.

    The same computation as ``CFMNN.predict_details``, in one pass.
    """
    X = model._check_X(X)
    n, H = len(X), model.n_boxes_
    tied = np.zeros((n, H), dtype=bool)
    cls = np.full(n, -1, dtype=np.int64)
    if H == 0:
        return tied, cls
    tol = float(model.tie_tol)
    step = model._chunk_size()
    for s in range(0, n, step):
        Xc = X[s:s + step]
        M = feature_terms(Xc[:, None, :], model.V_, model.W_, model.gamma).mean(axis=2)
        D = ((Xc[:, None, :] - model.centroid_[None, :, :]) ** 2).sum(axis=2)
        t = M >= M.max(axis=1, keepdims=True) - tol
        tied[s:s + step] = t
        cls[s:s + step] = model.box_class_[np.argmin(np.where(t, D, np.inf), axis=1)]
    return tied, cls


def in_top_group(model: CFMNN, X, boxes) -> np.ndarray:
    """True where any of ``boxes`` is within ``tie_tol`` of the highest membership at x."""
    boxes = np.asarray(boxes, dtype=np.int64)
    tied, _ = top_groups(model, X)
    return tied[:, boxes].any(axis=1) if len(boxes) else np.zeros(len(tied), dtype=bool)


def certify(before: CFMNN, after: CFMNN, tied_before, tied_after, removed: int | None = None):
    """(certificate, tight certificate) from the top groups of both models."""
    ch_b, ch_a = changed_boxes(before, after, removed)
    hit = tied_before[:, ch_b].any(axis=1) | tied_after[:, ch_a].any(axis=1)
    classes = np.zeros((len(hit), len(before.classes_)), dtype=bool)
    for c in range(len(before.classes_)):
        classes[:, c] = (tied_before[:, before.box_class_ == c].any(axis=1)
                         | tied_after[:, after.box_class_ == c].any(axis=1))
    return hit, hit & (classes.sum(axis=1) > 1)


def impact_set(before: CFMNN, after: CFMNN, X, removed: int | None = None,
               tight: bool = False) -> np.ndarray:
    """Rows of X whose prediction the edit before -> after can have changed (a superset)."""
    loose, strict = certify(before, after, top_groups(before, X)[0], top_groups(after, X)[0], removed)
    return strict if tight else loose


def prediction_churn(y_true, pred_before, pred_after) -> dict:
    """Share of rows whose prediction changed, and how many were fixed or broken."""
    y_true, pb, pa = (np.asarray(a) for a in (y_true, pred_before, pred_after))
    changed = pb != pa
    return {"changed_share": float(changed.mean()),
            "fixed": int(np.sum((pb != y_true) & (pa == y_true))),
            "broken": int(np.sum((pb == y_true) & (pa != y_true))),
            "changed": int(changed.sum())}


def top_k_sets(scores: np.ndarray, k: int = 3) -> list[frozenset]:
    """The k highest-scoring features of each row (stable order on ties)."""
    order = np.argsort(-np.asarray(scores), axis=1, kind="stable")[:, :k]
    return [frozenset(r.tolist()) for r in order]


def explanation_churn(scores_before: np.ndarray, scores_after: np.ndarray, k: int = 3) -> dict:
    """Share of rows whose attribution changed at all, and whose top-k feature set changed."""
    sb, sa = np.asarray(scores_before), np.asarray(scores_after)
    if len(sb) == 0:
        return {"any_change_share": float("nan"), "topk_change_share": float("nan"), "n": 0}
    any_change = np.abs(sb - sa).max(axis=1) > 1e-12
    topk = np.array([a != b for a, b in zip(top_k_sets(sb, k), top_k_sets(sa, k))])
    return {"any_change_share": float(any_change.mean()), "topk_change_share": float(topk.mean()),
            "n": int(len(sb))}

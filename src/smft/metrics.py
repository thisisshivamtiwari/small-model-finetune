"""Accuracy, macro-F1 and bootstrap intervals. Invalid replies count as wrong and as their own class for F1."""

import numpy as np
from sklearn.metrics import f1_score

from smft.labels import LABELS

INVALID = "<invalid>"


def _arrays(gold: list[str], pred: list[str | None]) -> tuple[np.ndarray, np.ndarray]:
    index = {label: i for i, label in enumerate(LABELS)}
    g = np.array([index[x] for x in gold])
    p = np.array([index.get(x, -1) if x is not None else -1 for x in pred])
    return g, p


def _macro_f1(g: np.ndarray, p: np.ndarray) -> float:
    # Averaged over the 77 real intents only; an invalid reply is a miss for its gold intent.
    return float(f1_score(g, p, labels=list(range(len(LABELS))), average="macro", zero_division=0))


def score(gold: list[str], pred: list[str | None], n_boot: int = 2000, seed: int = 0) -> dict:
    g, p = _arrays(gold, pred)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    acc_boot = (g[idx] == p[idx]).mean(axis=1)
    f1_boot = np.array([_macro_f1(g[i], p[i]) for i in idx[: min(n_boot, 500)]])
    return {
        "n": int(len(g)),
        "accuracy": round(float((g == p).mean()), 4),
        "accuracy_ci95": [round(float(x), 4) for x in np.percentile(acc_boot, [2.5, 97.5])],
        "macro_f1": round(_macro_f1(g, p), 4),
        "macro_f1_ci95": [round(float(x), 4) for x in np.percentile(f1_boot, [2.5, 97.5])],
        "invalid_rate": round(float((p == -1).mean()), 4),
    }


def paired_accuracy_diff(gold: list[str], pred_a: list, pred_b: list, n_boot: int = 2000, seed: int = 0) -> dict:
    """Accuracy of A minus B on the same items, with a paired bootstrap interval and the share of resamples where A wins."""
    g, a = _arrays(gold, pred_a)
    _, b = _arrays(gold, pred_b)
    diff = (g == a).astype(float) - (g == b).astype(float)
    rng = np.random.default_rng(seed)
    boot = diff[rng.integers(0, len(g), size=(n_boot, len(g)))].mean(axis=1)
    return {
        "diff": round(float(diff.mean()), 4),
        "ci95": [round(float(x), 4) for x in np.percentile(boot, [2.5, 97.5])],
        "p_a_better": round(float((boot > 0).mean()), 4),
    }

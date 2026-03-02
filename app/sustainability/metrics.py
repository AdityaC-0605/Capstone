"""Shared evaluation metrics for sustainability experiments."""

import numpy as np


def ks_statistic(y_true, y_prob):
    """
    Kolmogorov-Smirnov statistic for binary classification.

    Measures the maximum separation between the cumulative distributions
    of positive and negative classes ordered by predicted probability.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_prob : array-like
        Predicted probabilities for the positive class.

    Returns
    -------
    float
        KS statistic value in [0, 1].
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    order = np.argsort(y_prob)
    y_sorted = y_true[order]

    positives = y_sorted.sum()
    negatives = len(y_sorted) - positives

    if positives == 0 or negatives == 0:
        return 0.0

    return float(
        np.max(
            np.abs(
                np.cumsum(y_sorted) / positives
                - np.cumsum(1 - y_sorted) / negatives
            )
        )
    )

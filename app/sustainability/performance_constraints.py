def satisfies_constraints(
    metrics,
    reference_metrics,
    auc_drop_max=0.08,
    ks_drop_max=0.12,
    brier_increase_max=0.06
):
    """
    Enforces decision stability relative to a reference model.
    """

    if metrics["auc"] < reference_metrics["auc"] - auc_drop_max:
        return False

    if metrics["ks"] < reference_metrics["ks"] - ks_drop_max:
        return False

    if metrics["brier"] > reference_metrics["brier"] + brier_increase_max:
        return False

    return True

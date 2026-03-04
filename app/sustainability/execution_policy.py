def choose_exit_level(
    carbon_budget_ms,
    estimated_preprocessing_ms,
    exit_latencies_ms
):
    """
    Selects the deepest exit that fits within the carbon budget.
    """

    available_budget = max(0.0, carbon_budget_ms - estimated_preprocessing_ms)
    chosen = min(exit_latencies_ms.keys())

    for level in sorted(exit_latencies_ms.keys()):
        if exit_latencies_ms[level] <= available_budget:
            chosen = level
        else:
            break

    return chosen

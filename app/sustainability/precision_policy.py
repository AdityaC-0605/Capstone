def choose_precision(
    carbon_budget_ms,
    estimated_latency_ms,
    risk_tolerance="medium"
):
    """
    Selects numerical precision based on budget and risk tolerance.
    """

    if risk_tolerance == "high":
        return "fp32"

    if estimated_latency_ms > carbon_budget_ms:
        return "int8"

    if risk_tolerance == "low":
        return "fp16"

    return "fp32"

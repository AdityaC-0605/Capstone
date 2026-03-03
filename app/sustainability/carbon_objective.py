def carbon_cost(
    preprocessing_latency_ms,
    model_latency_ms,
    operation_count,
    precision_multiplier=1.0,
    alpha=1.0,
    beta=0.5,
    gamma=0.1
):
    """
    Carbon-aware cost proxy.

    preprocessing_latency_ms : system overhead
    model_latency_ms         : model execution cost
    operation_count          : approximate FLOPs proxy
    precision_multiplier     : FP32=1.0, FP16<1.0, INT8<<1.0
    """

    latency_cost = alpha * (preprocessing_latency_ms + model_latency_ms)
    compute_cost = beta * operation_count * precision_multiplier

    return latency_cost + compute_cost

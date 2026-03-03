try:
    from .scalable_mlp import ScalableMLP
    from .carbon_objective import carbon_cost
    from .performance_constraints import satisfies_constraints
    from .precision_modes import PRECISION_CONFIG
except ImportError:
    # Allow running as a standalone script from this directory.
    from scalable_mlp import ScalableMLP
    from carbon_objective import carbon_cost
    from performance_constraints import satisfies_constraints
    from precision_modes import PRECISION_CONFIG

import logging


logger = logging.getLogger(__name__)

# -----------------------------
# Architecture Search Space (expanded for better coverage)
# -----------------------------

SEARCH_SPACE = [
    {"hidden_scale": 1.0},
    {"hidden_scale": 0.85},
    {"hidden_scale": 0.75},
    {"hidden_scale": 0.6},
    {"hidden_scale": 0.5},
    {"hidden_scale": 0.35},
]

EXIT_LEVELS = [1, 2, 3]
PRECISION_MODES = ["fp32", "fp16", "int8"]


# -----------------------------
# Utility
# -----------------------------

def estimate_operation_count(hidden_scale):
    """
    Rough FLOP proxy.
    """
    base_ops = 1000
    return base_ops * hidden_scale


# -----------------------------
# NAS Algorithm
# -----------------------------

def carbon_aware_nas(
    X_tensor,
    y_true,
    reference_metrics,
    preprocessing_latency_ms,
    exit_latencies_ms,
    evaluate_model_fn,
    fallback_top_k=15,
    verbose=False,
    dropout=0.3,
):

    pareto_candidates = []
    all_candidates = []

    total_configs = len(SEARCH_SPACE) * len(EXIT_LEVELS) * len(PRECISION_MODES)
    current = 0

    for arch in SEARCH_SPACE:

        hidden_scale = arch["hidden_scale"]

        for exit_level in EXIT_LEVELS:
            for precision in PRECISION_MODES:
                current += 1
                if verbose:
                    logger.info(
                        "[%s/%s] scale=%s exit=%s precision=%s",
                        current,
                        total_configs,
                        hidden_scale,
                        exit_level,
                        precision,
                    )

                model = ScalableMLP(
                    input_dim=X_tensor.shape[1],
                    hidden_scale=hidden_scale,
                    dropout=dropout
                )

                precision_multiplier = PRECISION_CONFIG[precision]["multiplier"]

                probs, metrics = evaluate_model_fn(
                    model=model,
                    X_tensor=X_tensor,
                    exit_level=exit_level,
                    precision=precision
                )

                # --------------------
                # Carbon Cost
                # --------------------
                operation_count = estimate_operation_count(hidden_scale)

                cost = carbon_cost(
                    preprocessing_latency_ms,
                    exit_latencies_ms[exit_level],
                    operation_count,
                    precision_multiplier
                )

                candidate = {
                    "architecture": arch,
                    "exit_level": exit_level,
                    "precision": precision,
                    "metrics": metrics,
                    "carbon_cost": cost,
                }
                candidate["multi_objective_score"] = (
                    (metrics["auc"] * 0.6)
                    + (metrics["ks"] * 0.3)
                    - (metrics["brier"] * 0.1)
                    - (cost * 0.001)
                )

                all_candidates.append(candidate)

                # --------------------
                # Stability Constraints
                # --------------------
                passes = satisfies_constraints(metrics, reference_metrics)
                if passes:
                    pareto_candidates.append(candidate)
                    if verbose:
                        logger.info("AUC=%.4f PASS", metrics["auc"])
                elif verbose:
                    logger.info("AUC=%.4f", metrics["auc"])

    if verbose:
        logger.info("Configurations tested: %s", len(all_candidates))
        logger.info(
            "Configurations passing constraints: %s",
            len(pareto_candidates),
        )

    # Sort by cost
    pareto_candidates.sort(key=lambda x: x["carbon_cost"])
    all_candidates.sort(key=lambda x: x["carbon_cost"])

    if pareto_candidates:
        return pareto_candidates

    # Fallback: still return best candidates for inspection when constraints are too strict.
    fallback = all_candidates[:fallback_top_k]
    for candidate in fallback:
        candidate["constraint_violation"] = True

    return fallback

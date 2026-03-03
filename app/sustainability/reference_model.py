REFERENCE_MODELS = {
    "logistic_regression": {
        "auc": 0.9349,
        "ks": 0.7057,
        "brier": 0.1041
    },
    "mlp_full": {
        "auc": 0.9996,
        "ks": 0.9844,
        "brier": 0.0093
    },
    # German Credit dataset reference (logistic regression baseline)
    # German Credit is a harder, smaller dataset; AUC ~0.75-0.80 is typical
    "german_logistic_regression": {
        "auc": 0.72,
        "ks": 0.30,
        "brier": 0.22
    },
}

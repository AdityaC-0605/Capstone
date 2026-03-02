import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score, brier_score_loss

try:
    from .preprocessing import load_and_preprocess
    from .logistic_regression import train_logistic
    from .mlp import MLP, train_mlp
    from .inference_cost import measure_block_latency
    from .metrics import ks_statistic
except ImportError:
    # Allow running as a standalone script from this directory.
    from preprocessing import load_and_preprocess
    from logistic_regression import train_logistic
    from mlp import MLP, train_mlp
    from inference_cost import measure_block_latency
    from metrics import ks_statistic

DATA_PATH = Path(__file__).resolve().parent / "Bank_data.csv"

def main():
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess(str(DATA_PATH))

    # ---------------- PREPROCESSING COST ----------------
    prep_latency = measure_block_latency(
        fn=lambda x: preprocessor.fit_transform(x),
        X=X_train
    )

    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    if hasattr(X_train_p, "toarray"):
        X_train_p = X_train_p.toarray()
        X_test_p = X_test_p.toarray()

    # ---------------- LOGISTIC REGRESSION ----------------
    log_model = train_logistic(X_train, y_train, preprocessor)
    log_probs = log_model.predict_proba(X_test)[:, 1]

    log_model_latency = measure_block_latency(
        fn=lambda x: log_model.predict_proba(x)[:, 1],
        X=X_test
    )

    # ---------------- MLP ----------------
    mlp_probs = train_mlp(X_train_p, y_train, X_test_p)

    mlp_model = MLP(X_train_p.shape[1])
    mlp_model.eval()
    X_tensor = torch.tensor(X_test_p, dtype=torch.float32)

    mlp_model_latency = measure_block_latency(
        fn=lambda x: mlp_model(x),
        X=X_tensor
    )

    # ---------------- RESULTS ----------------
    print("\n=== Preprocessing Cost ===")
    print(prep_latency)

    print("\n=== Logistic Regression ===")
    print({
        "AUC": roc_auc_score(y_test, log_probs),
        "Brier": brier_score_loss(y_test, log_probs),
        "KS": ks_statistic(y_test.values, log_probs),
        "model_latency": log_model_latency
    })

    print("\n=== MLP ===")
    print({
        "AUC": roc_auc_score(y_test, mlp_probs),
        "Brier": brier_score_loss(y_test, mlp_probs),
        "KS": ks_statistic(y_test.values, mlp_probs),
        "model_latency": mlp_model_latency
    })

if __name__ == "__main__":
    main()

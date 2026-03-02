import torch
import numpy as np
from pathlib import Path

from sklearn.metrics import roc_auc_score, brier_score_loss
from torch.utils.data import DataLoader, TensorDataset

try:
    from .preprocessing_german import load_and_preprocess_german
    from .carbon_aware_nas import carbon_aware_nas
    from .reference_model import REFERENCE_MODELS
    from .plot_pareto import plot_pareto
    from .sustainability_report import sustainability_summary
    from .research_table import generate_research_table
    from .metrics import ks_statistic
except ImportError:
    # Allow running as a standalone script from this directory.
    from preprocessing_german import load_and_preprocess_german
    from carbon_aware_nas import carbon_aware_nas
    from reference_model import REFERENCE_MODELS
    from plot_pareto import plot_pareto
    from sustainability_report import sustainability_summary
    from research_table import generate_research_table
    from metrics import ks_statistic


# -----------------------------
# DATA PATH (German dataset)
# -----------------------------

DATA_PATH = Path(__file__).resolve().parent / "german_data.csv"


# -----------------------------
# MODEL EVALUATION FACTORY
# -----------------------------

def make_evaluate_fn(X_train_t, y_train_t, y_test_series):
    """
    Return an evaluate_model_fn that trains on the captured training data
    and evaluates on the captured test labels.  Avoids fragile global state.
    Tuned for the smaller German Credit dataset.
    """

    def evaluate_model_fn(model, X_tensor, exit_level, precision):
        model.train()

        # Lower learning rate and stronger weight decay for the small dataset
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-3)

        # Class-weighted loss to handle the 70/30 class imbalance
        n_pos = y_train_t.sum().item()
        n_neg = len(y_train_t) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)])
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        best_loss = float('inf')
        patience_counter = 0
        patience = 10

        for epoch in range(100):
            epoch_loss = 0.0
            n_batches = 0

            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                logits = model(x_batch.float(), exit_level=exit_level)
                loss = loss_fn(logits, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches

            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        model.eval()

        with torch.no_grad():
            logits = model(X_tensor.float(), exit_level=exit_level)
            probs = torch.sigmoid(logits).cpu().numpy()

        metrics = {
            "auc": roc_auc_score(y_test_series, probs),
            "ks": ks_statistic(y_test_series.values, probs),
            "brier": brier_score_loss(y_test_series, probs)
        }

        return probs, metrics

    return evaluate_model_fn


# -----------------------------
# MAIN
# -----------------------------

def main():

    print("\nLoading German Credit dataset...")

    X_train_scaled, X_test_scaled, y_train, y_test, scaler = load_and_preprocess_german(str(DATA_PATH))

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    evaluate_fn = make_evaluate_fn(X_train_tensor, y_train_tensor, y_test)

    print("Running Carbon-Aware NAS...\n")

    reference_metrics = REFERENCE_MODELS["german_logistic_regression"]

    preprocessing_latency_ms = 12.0

    exit_latencies_ms = {
        1: 0.10,
        2: 0.20,
        3: 0.25
    }

    results = carbon_aware_nas(
        X_tensor,
        y_test,
        reference_metrics,
        preprocessing_latency_ms,
        exit_latencies_ms,
        evaluate_fn,
        verbose=True,
        dropout=0.15,  # Lower dropout for small German dataset (~800 training samples)
    )

    print("\n=== NAS RESULTS (GERMAN DATASET) ===\n")

    if len(results) == 0:
        print("No configurations satisfied stability constraints.")
        return

    if results[0].get("constraint_violation"):
        print("No configurations met stability constraints; showing lowest-carbon fallback candidates.")

    for r in results[:10]:
        print(r)

    plot_pareto(results)

    print("\n=== SUSTAINABILITY RESULTS ===")

    sustainability_summary(results)

    print("\n=== RESEARCH COMPARISON TABLE ===")

    generate_research_table(results)


if __name__ == "__main__":
    main()

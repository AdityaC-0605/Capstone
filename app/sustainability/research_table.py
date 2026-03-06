import numpy as np


def generate_research_table(results):
    if not results:
        print("No results available for research comparison.")
        return

    # -----------------------------
    # Identify baseline (full fp32 largest model, deepest exit)
    # -----------------------------

    fp32_full = [r for r in results if r["precision"] == "fp32" and r["architecture"]["hidden_scale"] == 1.0 and r["exit_level"] == 3]
    baseline = fp32_full[0] if fp32_full else max(results, key=lambda x: x["carbon_cost"])

    # -----------------------------
    # Combined best (ours) — best efficiency
    # -----------------------------

    combined = max(results, key=lambda x: x["metrics"]["auc"] / x["carbon_cost"])

    # -----------------------------
    # INT8 only baseline (same architecture as baseline but int8)
    # -----------------------------

    int8_candidates = [r for r in results if r["precision"] == "int8"]
    int8_best_auc = max(int8_candidates, key=lambda x: x["metrics"]["auc"]) if int8_candidates else baseline

    # -----------------------------
    # Scaling only baseline (fp32, smallest architecture)
    # -----------------------------

    fp32_candidates = [r for r in results if r["precision"] == "fp32"]
    scaling_only = min(fp32_candidates, key=lambda x: x["carbon_cost"]) if fp32_candidates else baseline

    # -----------------------------
    # Early exit only (full model but exit_level=1, fp32)
    # -----------------------------

    exit_only = [r for r in results if r["precision"] == "fp32" and r["architecture"]["hidden_scale"] == 1.0 and r["exit_level"] == 1]
    early_exit = exit_only[0] if exit_only else baseline

    # -----------------------------
    # Best AUC overall
    # -----------------------------

    best_auc_model = max(results, key=lambda x: x["metrics"]["auc"])

    rows = []

    def build_row(name, model):
        reduction = (
            (baseline["carbon_cost"] - model["carbon_cost"])
            / baseline["carbon_cost"]
        ) * 100

        efficiency = model["metrics"]["auc"] / model["carbon_cost"]

        return [
            name,
            round(model["metrics"]["auc"], 4),
            round(model["metrics"]["ks"], 4),
            round(model["metrics"]["brier"], 4),
            round(model["carbon_cost"], 2),
            round(reduction, 1),
            round(efficiency, 6),
        ]

    rows.append(build_row("Full FP32 Baseline", baseline))
    rows.append(build_row("INT8 Best AUC", int8_best_auc))
    rows.append(build_row("Scaling Only (FP32 small)", scaling_only))
    rows.append(build_row("Early Exit Only (FP32)", early_exit))
    rows.append(build_row("Best AUC Overall", best_auc_model))
    rows.append(build_row("Combined (Ours)", combined))

    # -----------------------------
    # Print Table
    # -----------------------------

    headers = ["Method", "AUC", "KS", "Brier", "Carbon", "Reduct%", "Effic."]

    print("\n=== Research Comparison Table ===\n")

    fmt = "{:<28} {:<8} {:<8} {:<8} {:<8} {:<8} {:<10}"
    print(fmt.format(*headers))
    print("-" * 88)

    for r in rows:
        print(fmt.format(*r))

    # Highlight the key finding
    print(f"\n--- Key Finding ---")
    print(f"Our combined approach achieves {combined['metrics']['auc']:.4f} AUC")
    print(f"at {combined['carbon_cost']:.1f} carbon cost")
    ours_reduction = (baseline["carbon_cost"] - combined["carbon_cost"]) / baseline["carbon_cost"] * 100
    ours_retention = combined["metrics"]["auc"] / baseline["metrics"]["auc"] * 100
    print(f"Carbon Reduction: {ours_reduction:.1f}%")
    print(f"Performance Retention: {ours_retention:.1f}%")

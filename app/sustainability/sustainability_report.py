def sustainability_summary(results):

    # define baseline (full fp32 largest model)
    baseline = max(results, key=lambda x: x["carbon_cost"])

    # best carbon-aware model: best AUC among lowest-cost models
    # Select optimized as lowest carbon that still has decent AUC
    sorted_by_cost = sorted(results, key=lambda x: x["carbon_cost"])
    
    # Pick the model with best efficiency (AUC / carbon_cost)
    optimized = max(results, key=lambda x: x["metrics"]["auc"] / x["carbon_cost"])

    baseline_cost = baseline["carbon_cost"]
    optimized_cost = optimized["carbon_cost"]

    baseline_auc = baseline["metrics"]["auc"]
    optimized_auc = optimized["metrics"]["auc"]

    carbon_reduction = (baseline_cost - optimized_cost) / max(baseline_cost, 1e-12) * 100
    performance_retention = optimized_auc / max(baseline_auc, 1e-12) * 100

    baseline_eff = baseline_auc / baseline_cost
    optimized_eff = optimized_auc / optimized_cost

    print("\n=== Sustainability Comparison ===\n")

    print("Baseline Model:")
    print(baseline)

    print("\nOptimized Carbon-Aware Model:")
    print(optimized)

    print(f"\nCarbon Reduction: {carbon_reduction:.2f}%")
    print(f"Performance Retention: {performance_retention:.2f}%")

    print("\nEfficiency (AUC per carbon):")
    print(f"Baseline: {baseline_eff:.6f}")
    print(f"Optimized: {optimized_eff:.6f}")
    print(f"Efficiency Gain: {(optimized_eff / baseline_eff - 1) * 100:.1f}%")

import matplotlib.pyplot as plt

def compute_pareto_frontier(results):

    sorted_results = sorted(results, key=lambda x: x["carbon_cost"])

    pareto = []
    best_auc = -1

    for r in sorted_results:

        auc = r["metrics"]["auc"]

        if auc > best_auc:
            pareto.append(r)
            best_auc = auc

    return pareto


def plot_pareto(results):

    costs = [r["carbon_cost"] for r in results]
    aucs = [r["metrics"]["auc"] for r in results]

    pareto = compute_pareto_frontier(results)

    pareto_costs = [r["carbon_cost"] for r in pareto]
    pareto_aucs = [r["metrics"]["auc"] for r in pareto]

    plt.figure(figsize=(8,6))

    plt.scatter(costs, aucs, label="NAS candidates")
    plt.plot(pareto_costs, pareto_aucs, color="red", linewidth=2, label="Pareto Frontier")

    plt.xlabel("Carbon Cost")
    plt.ylabel("AUC Performance")
    plt.title("Carbon-Aware NAS Pareto Frontier")

    plt.legend()
    plt.show()

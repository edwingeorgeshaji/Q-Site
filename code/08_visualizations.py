"""
Q-Site | Step 8: Generate visualizations for the technical write-up.

WHAT THIS STEP DOES (in plain terms):
--------------------------------------
Numbers in a table are hard for a judge to absorb quickly. This step turns
our Steps 1-7 results into three figures that make the story visually
obvious:

  Figure 1 - GRID TOPOLOGY: the IEEE 14-bus network diagram, with the AI
             data center load bus and the two QAOA-selected battery sites
             clearly marked. This shows WHERE the problem and solution are.

  Figure 2 - BEFORE/AFTER HEATMAP: for every one of our 15 contingency
             scenarios, how many buses were unsafe BEFORE batteries vs
             AFTER. This shows WHETHER the solution actually helped, and by
             how much, per scenario — not just in aggregate.

  Figure 3 - QAOA SOLUTION LANDSCAPE: every valid 2-battery combination,
             ranked by QUBO cost, with the QAOA-found and classical-optimal
             answer both marked (they're the same point, which IS the
             finding worth showing). This shows quantum actually explored
             the space and landed on the true optimum, not a shortcut.

WHERE THE DATA COMES FROM:
--------------------------------------
Every figure is built directly from the JSON files written by Steps 1-7 —
no new numbers are computed here, only rendering existing verified results.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless rendering, no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

FIG_DIR = "../results/figures"

# ---------------------------------------------------------------------------
# Load all upstream results
# ---------------------------------------------------------------------------

def load_all_data():
    base = "../data"
    with open(f"{base}/grid_graph.json") as f:
        graph_data = json.load(f)
    with open(f"{base}/ai_load_bus.json") as f:
        ai_load_data = json.load(f)
    with open(f"{base}/contingency_scenarios.json") as f:
        before_scenarios = json.load(f)
    with open(f"{base}/validation_with_batteries.json") as f:
        validation = json.load(f)
    with open(f"{base}/classical_baseline_result.json") as f:
        classical = json.load(f)
    with open(f"{base}/qaoa_simulator_result.json") as f:
        qaoa = json.load(f)
    with open(f"{base}/qubo_problem.json") as f:
        qubo_problem = json.load(f)
    return {
        "graph_data": graph_data,
        "ai_load_bus": ai_load_data["ai_load_bus"],
        "before_scenarios": before_scenarios,
        "validation": validation,
        "classical": classical,
        "qaoa": qaoa,
        "qubo_problem": qubo_problem,
    }


# ---------------------------------------------------------------------------
# Figure 1: Grid topology
# ---------------------------------------------------------------------------

def figure_grid_topology(data):
    G = nx.Graph()
    G.add_nodes_from(data["graph_data"]["nodes"])
    G.add_edges_from(data["graph_data"]["edges"])

    ai_bus = data["ai_load_bus"]
    battery_buses = data["validation"]["battery_buses"]
    candidate_buses = data["qubo_problem"]["candidate_buses"]

    pos = nx.spring_layout(G, seed=42, k=0.9)

    fig, ax = plt.subplots(figsize=(9, 7))

    # Draw base edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#999999", width=1.5)

    # Node coloring: default / candidate / battery-selected / AI-load
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        if node == ai_bus:
            node_colors.append("#d62728")   # red = AI load stress point
            node_sizes.append(1000)
        elif node in battery_buses:
            node_colors.append("#2ca02c")   # green = selected battery site
            node_sizes.append(1000)
        elif node in candidate_buses:
            node_colors.append("#ff7f0e")   # orange = was a candidate, not selected
            node_sizes.append(700)
        else:
            node_colors.append("#a6cee3")   # light blue = ordinary bus
            node_sizes.append(600)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                             node_size=node_sizes, edgecolors="black", linewidths=1.2)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight="bold")

    legend_handles = [
        mpatches.Patch(color="#d62728", label=f"AI Data Center Load (Bus {ai_bus})"),
        mpatches.Patch(color="#2ca02c", label=f"Battery Sites Selected by QAOA {battery_buses}"),
        mpatches.Patch(color="#ff7f0e", label="Other Candidate Sites (not selected)"),
        mpatches.Patch(color="#a6cee3", label="Other Grid Buses"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)
    ax.set_title("IEEE 14-Bus Test System — Q-Site Battery Siting Result", fontsize=13, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    out_path = f"{FIG_DIR}/fig1_grid_topology.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


# ---------------------------------------------------------------------------
# Figure 2: Before/after heatmap
# ---------------------------------------------------------------------------

def figure_before_after_heatmap(data):
    before = data["before_scenarios"]
    after = data["validation"]["after_scenarios"]

    labels = []
    before_vals = []
    after_vals = []

    for b, a in zip(before, after):
        labels.append(f"Remove {b['from_bus']}-{b['to_bus']}")
        # Represent "collapse" as a large sentinel value (worse than any real count)
        # so it's visually distinct on the heatmap, and annotate it explicitly.
        before_vals.append(b["num_unsafe_buses"] if b["converged"] else 6)
        after_vals.append(a["num_unsafe_buses"] if a["converged"] else 6)

    matrix = np.array([before_vals, after_vals])

    fig, ax = plt.subplots(figsize=(11, 3.2))
    im = ax.imshow(matrix, cmap="Reds", aspect="auto", vmin=0, vmax=6)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["BEFORE\n(no battery)", "AFTER\n(QAOA siting)"], fontsize=10)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    # Annotate each cell with its value (or "COLLAPSE")
    for row_idx, row_vals in enumerate([before, after]):
        for col_idx, scenario in enumerate(row_vals):
            if not scenario["converged"]:
                text = "COLLAPSE"
                color = "white"
                fontsize = 7.5
            else:
                text = str(scenario["num_unsafe_buses"])
                color = "white" if matrix[row_idx, col_idx] > 3 else "black"
                fontsize = 9
            ax.text(col_idx, row_idx, text, ha="center", va="center",
                     fontsize=fontsize, color=color, fontweight="bold")

    ax.set_title("Unsafe-Bus Count per N-1 Contingency Scenario: Before vs After Battery Siting",
                 fontsize=11, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("Unsafe buses (6 = total voltage collapse)", fontsize=8)

    plt.tight_layout()
    out_path = f"{FIG_DIR}/fig2_before_after_heatmap.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


# ---------------------------------------------------------------------------
# Figure 3: QAOA solution landscape
# ---------------------------------------------------------------------------

def figure_qaoa_landscape(data):
    all_results = data["classical"]["all_results"]  # already sorted by cost, ranked
    qaoa_buses = set(data["qaoa"]["selected_buses"])
    classical_buses = set(data["classical"]["best_buses"])

    labels = [str(r["buses"]) for r in all_results]
    costs = [r["cost"] for r in all_results]

    colors = []
    for r in all_results:
        buses_set = set(r["buses"])
        if buses_set == qaoa_buses and buses_set == classical_buses:
            colors.append("#2ca02c")  # green: both agree here (the actual finding)
        else:
            colors.append("#a6cee3")  # light blue: other candidate combos

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(labels)), costs, color=colors, edgecolor="black", linewidth=0.6)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("QUBO Cost (lower = better)", fontsize=10)
    ax.set_title("All Valid Battery-Site Combinations, Ranked by QUBO Cost\n"
                 "Green = combination found by BOTH classical brute-force AND QAOA",
                 fontsize=11, fontweight="bold")
    ax.axhline(0, color="gray", linewidth=0.8)

    legend_handles = [
        mpatches.Patch(color="#2ca02c", label="QAOA result = Classical optimum (this combination)"),
        mpatches.Patch(color="#a6cee3", label="Other candidate combinations"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)

    plt.tight_layout()
    out_path = f"{FIG_DIR}/fig3_qaoa_landscape.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 8: GENERATING VISUALIZATIONS")
    print("=" * 60)

    data = load_all_data()

    print("\nBuilding Figure 1: Grid topology diagram...")
    p1 = figure_grid_topology(data)
    print(f"  Saved: {p1}")

    print("Building Figure 2: Before/after contingency heatmap...")
    p2 = figure_before_after_heatmap(data)
    print(f"  Saved: {p2}")

    print("Building Figure 3: QAOA solution landscape...")
    p3 = figure_qaoa_landscape(data)
    print(f"  Saved: {p3}")

    print("\nAll figures generated successfully.")

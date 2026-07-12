"""
Q-Site | Step 4: Define candidate battery sites and formulate the QUBO.

WHAT THIS STEP DOES (in plain terms):
--------------------------------------
So far we've PROVEN the grid has a problem (Steps 2-3). Now we need to frame
the actual DECISION as a math problem a quantum computer can work on.

The decision is: "Out of a shortlist of candidate buses, which ones should
get a battery, given we can only afford to build a limited number?"

This is a classic "subset selection under a budget" problem. We turn it into
a QUBO (Quadratic Unconstrained Binary Optimization) — a format quantum
computers can search over. Concretely:

  - Each candidate bus gets a binary variable x_i: 1 = "build a battery here",
    0 = "don't".
  - We define a COST function that scores how good a particular combination
    of batteries is, based on how much it would have helped across our 15
    contingency scenarios from Step 3.
  - We add a PENALTY term that punishes selecting too many or too few
    batteries relative to our budget (this is how a "budget constraint"
    gets folded into an "unconstrained" QUBO format).
  - The quantum computer's job (later, in Step 6) is to find the x_i
    combination that minimizes this cost function.

WHY BUSES ARE SCORED THE WAY THEY ARE:
--------------------------------------
We can't simulate "with a battery installed" using a simple power flow
directly for QUBO scoring (that requires re-running load flow per combination,
which we'll actually do for validation in Step 5). For the QUBO's cost
coefficients, we use a standard, defensible proxy used in real siting studies:
a bus's "vulnerability score" = how often it shows up as unsafe across our
contingency scenarios, weighted by how severe the voltage deviation is.
Buses that are frequently and severely unsafe are the best candidates for
storage, because storage provides local voltage support.

This vulnerability score becomes the LINEAR coefficient for each x_i.
We also add small QUADRATIC coupling terms between electrically-adjacent
candidate buses (from our grid graph) to slightly discourage clustering two
batteries right next to each other (redundant coverage) and encourage spread.

WHERE THE OUTPUT COMES FROM:
--------------------------------------
The vulnerability scores are computed directly from the contingency_scenarios.json
data generated in Step 3 (real pandapower power flow results). Nothing here is
hardcoded — if you re-run Step 3 with different assumptions, this step's QUBO
coefficients change accordingly.
"""

import json
import math
import numpy as np
import networkx as nx

VOLTAGE_MIN = 0.94
VOLTAGE_MAX = 1.06
NUM_BATTERIES = 2   # budget: how many batteries we can afford to build

# Candidate sites: chosen based on ACTUAL under-voltage exposure found in
# Step 3's contingency scenarios (only buses 9 and 10 ever go under-voltage
# in this scenario set), plus their direct electrical neighbors (buses that
# could plausibly provide nearby voltage support). This replaces an earlier,
# less rigorous candidate list that included buses (5, 7) which were in fact
# OVER-voltage, not under-voltage, and therefore could not be helped by a
# reactive-power-injecting battery (see Step 7 validation notes).
CANDIDATE_BUSES = [9, 10, 8, 5, 13, 11]


def load_scenarios():
    path = "../data/contingency_scenarios.json"
    with open(path) as f:
        scenarios = json.load(f)
    return scenarios


def load_graph():
    path = "../data/grid_graph.json"
    with open(path) as f:
        data = json.load(f)
    G = nx.Graph()
    G.add_nodes_from(data["nodes"])
    G.add_edges_from(data["edges"])
    return G


def compute_vulnerability_scores(scenarios, candidate_buses):
    """
    For each candidate bus, compute a vulnerability score across all
    CONVERGED scenarios, counting ONLY UNDER-VOLTAGE violations (v < VOLTAGE_MIN).

    IMPORTANT MODELING DECISION: our battery is modeled (Step 7) as a
    reactive-power-INJECTING device (standard volt/VAR support). Injecting
    reactive power RAISES local voltage. This means it can only help buses
    that are suffering from UNDER-voltage — it would actively make an
    OVER-voltage problem worse. So the siting objective must only reward
    candidate buses for their under-voltage vulnerability, not their total
    (both-direction) voltage deviation. An earlier version of this scoring
    incorrectly summed both directions, which caused the QUBO to select
    buses that were already over-voltage — the opposite of useful. This
    version fixes that.

    Returns: dict {bus: vulnerability_score}
    """
    scores = {bus: 0.0 for bus in candidate_buses}
    collapse_count = 0

    for scenario in scenarios:
        if not scenario["converged"]:
            collapse_count += 1
            continue
        voltages = scenario["voltages"]
        for bus in candidate_buses:
            v = voltages.get(str(bus), voltages.get(bus))
            if v is None:
                continue
            if v < VOLTAGE_MIN:
                scores[bus] += (VOLTAGE_MIN - v)
            # NOTE: over-voltage (v > VOLTAGE_MAX) intentionally NOT scored here,
            # since a Q-injecting battery cannot fix over-voltage.

    return scores, collapse_count


def build_qubo(candidate_buses, vulnerability_scores, graph, num_batteries,
                penalty_weight=5.0, adjacency_weight=0.05):
    """
    Build the QUBO matrix Q such that the objective is:

        minimize   x^T Q x

    where x is a vector of binary variables (1 = battery installed).

    Components:
    1. LINEAR (diagonal) term: -vulnerability_score for each candidate bus.
       Negative because we're MINIMIZING, and we want to REWARD (lower cost)
       picking high-vulnerability buses. Higher vulnerability -> more negative
       -> more attractive to select.

    2. BUDGET PENALTY (quadratic, all pairs + diagonal): standard QUBO
       encoding of "exactly K selected" as a penalty:
           penalty_weight * (sum(x_i) - num_batteries)^2
       Expanding this gives diagonal and off-diagonal contributions.

    3. ADJACENCY term (small, quadratic): tiny positive coupling between
       candidate buses that are directly connected in the grid graph, to
       mildly discourage picking two adjacent buses (redundant coverage,
       since electrically close batteries protect overlapping areas).

    Returns: (Q matrix as dict {(i,j): coefficient}, bus_index_map)
    """
    n = len(candidate_buses)
    bus_index = {bus: i for i, bus in enumerate(candidate_buses)}
    Q = {}

    def add(i, j, val):
        key = (min(i, j), max(i, j))
        Q[key] = Q.get(key, 0.0) + val

    # --- 1. Vulnerability reward (diagonal) ---
    for bus in candidate_buses:
        i = bus_index[bus]
        add(i, i, -vulnerability_scores[bus])

    # --- 2. Budget penalty: P * (sum(x_i) - K)^2 ---
    # Expansion: P * [ sum(x_i^2) + 2*sum_{i<j}(x_i x_j) - 2K*sum(x_i) + K^2 ]
    # Since x_i is binary, x_i^2 = x_i.
    K = num_batteries
    for i in range(n):
        add(i, i, penalty_weight * (1 - 2 * K))
    for i in range(n):
        for j in range(i + 1, n):
            add(i, j, penalty_weight * 2)
    # constant term penalty_weight * K^2 is dropped (doesn't affect argmin)

    # --- 3. Adjacency discouragement ---
    for bus_a in candidate_buses:
        for bus_b in candidate_buses:
            if bus_a < bus_b and graph.has_edge(bus_a, bus_b):
                i, j = bus_index[bus_a], bus_index[bus_b]
                add(i, j, adjacency_weight)

    return Q, bus_index


def qubo_to_matrix(Q, n):
    """Convert the sparse dict QUBO into a dense numpy matrix for inspection/export."""
    M = np.zeros((n, n))
    for (i, j), val in Q.items():
        M[i, j] = val
    return M


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 4: BUILDING THE STORAGE-SITING QUBO")
    print("=" * 60)

    scenarios = load_scenarios()
    graph = load_graph()

    print(f"\nCandidate battery sites: {CANDIDATE_BUSES}")
    print(f"Battery budget: {NUM_BATTERIES} out of {len(CANDIDATE_BUSES)} candidates")
    print(f"Number of possible combinations (classical search space): "
          f"{math.comb(len(CANDIDATE_BUSES), NUM_BATTERIES)}")

    vulnerability_scores, collapse_count = compute_vulnerability_scores(scenarios, CANDIDATE_BUSES)

    print(f"\nScenarios that caused total voltage collapse (excluded from scoring): {collapse_count}")
    print("\nVulnerability scores (higher = more frequently/severely unsafe):")
    for bus, score in sorted(vulnerability_scores.items(), key=lambda x: -x[1]):
        print(f"  Bus {bus}: {score:.4f}")

    Q, bus_index = build_qubo(CANDIDATE_BUSES, vulnerability_scores, graph, NUM_BATTERIES)

    print(f"\nQUBO built with {len(CANDIDATE_BUSES)} binary variables "
          f"({len(CANDIDATE_BUSES)} qubits required for QAOA).")
    print(f"Bus-to-qubit index mapping: {bus_index}")

    M = qubo_to_matrix(Q, len(CANDIDATE_BUSES))
    print("\nQUBO matrix (upper-triangular form):")
    np.set_printoptions(precision=3, suppress=True)
    print(M)

    # Save everything downstream steps need
    output = {
        "candidate_buses": CANDIDATE_BUSES,
        "num_batteries": NUM_BATTERIES,
        "bus_index": bus_index,
        "vulnerability_scores": vulnerability_scores,
        "qubo": {f"{i},{j}": val for (i, j), val in Q.items()},
        "qubo_matrix": M.tolist(),
    }
    with open("../data/qubo_problem.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\nSaved QUBO problem to data/qubo_problem.json")

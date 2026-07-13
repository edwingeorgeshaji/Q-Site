"""
Q-Site | Step 5: Classical baseline solver.

WHAT THIS STEP DOES (in plain terms):
--------------------------------------
Before we let a quantum computer solve our battery-siting QUBO, we need a
classical answer to compare against. The challenge rubric explicitly requires
this ("report a classical baseline" — described as the most commonly missed
requirement). It's also just good practice: if we can't prove quantum did at
least as well as brute force, we have no case for "quantum advantage."

Our problem is small on purpose (6 candidate buses, choose 2 -> only 15
possible combinations), so brute force is not just possible but GUARANTEED
CORRECT — it checks literally every combination and picks the true minimum.
This gives us a gold-standard answer to judge QAOA's output against later.

WHERE THE OUTPUT COMES FROM:
--------------------------------------
We evaluate the exact same QUBO cost function built in Step 4
(data/qubo_problem.json) against every possible combination of 2-out-of-6
candidate buses. No shortcuts, no approximation — this is exhaustive search.
"""

import json
import itertools
import numpy as np
import time


def load_qubo_problem():
    with open("../data/qubo_problem.json") as f:
        return json.load(f)


def evaluate_qubo(x, Q_matrix):
    """
    Compute x^T Q x for a binary vector x and QUBO matrix Q.
    This is the standard QUBO objective evaluation.
    """
    x = np.array(x)
    Q = np.array(Q_matrix)
    return float(x @ Q @ x)


def brute_force_solve(qubo_data):
    candidate_buses = qubo_data["candidate_buses"]
    num_batteries = qubo_data["num_batteries"]
    bus_index = qubo_data["bus_index"]
    Q_matrix = qubo_data["qubo_matrix"]
    n = len(candidate_buses)

    print(f"Searching all combinations of {num_batteries} out of {n} candidate buses...")

    best_cost = float("inf")
    best_combo = None
    all_results = []

    start = time.time()
    # Try every valid combination (respecting the budget exactly, so we can
    # also sanity-check the QUBO penalty is doing its job when we test invalid
    # combos too, further down).
    for combo in itertools.combinations(range(n), num_batteries):
        x = [1 if i in combo else 0 for i in range(n)]
        cost = evaluate_qubo(x, Q_matrix)
        buses_selected = [candidate_buses[i] for i in combo]
        all_results.append({"buses": buses_selected, "cost": cost})
        if cost < best_cost:
            best_cost = cost
            best_combo = buses_selected
    elapsed = time.time() - start

    all_results.sort(key=lambda r: r["cost"])

    return best_combo, best_cost, all_results, elapsed


def sanity_check_penalty(qubo_data):
    """
    Verify the budget penalty actually works: selecting the WRONG number of
    batteries (e.g. 1 or 3 instead of 2) should always score worse (higher
    cost) than the best valid 2-battery combination. If not, the QUBO
    formulation has a bug.
    """
    candidate_buses = qubo_data["candidate_buses"]
    Q_matrix = qubo_data["qubo_matrix"]
    n = len(candidate_buses)
    num_batteries = qubo_data["num_batteries"]

    print("\nSanity check: budget penalty correctness")
    print("-" * 60)

    # Best cost among valid (correct-budget) combinations
    valid_costs = []
    for combo in itertools.combinations(range(n), num_batteries):
        x = [1 if i in combo else 0 for i in range(n)]
        valid_costs.append(evaluate_qubo(x, Q_matrix))
    best_valid_cost = min(valid_costs)

    # Costs for wrong-budget combinations (off by one, both directions)
    problems_found = 0
    for wrong_k in [num_batteries - 1, num_batteries + 1]:
        if wrong_k < 0 or wrong_k > n:
            continue
        wrong_costs = []
        for combo in itertools.combinations(range(n), wrong_k):
            x = [1 if i in combo else 0 for i in range(n)]
            wrong_costs.append(evaluate_qubo(x, Q_matrix))
        best_wrong_cost = min(wrong_costs) if wrong_costs else float("inf")
        ok = best_wrong_cost > best_valid_cost
        print(f"  Budget={wrong_k} best cost = {best_wrong_cost:.3f} "
              f"vs valid budget={num_batteries} best cost = {best_valid_cost:.3f} "
              f"-> {'OK (penalty working)' if ok else '*** PROBLEM: penalty too weak ***'}")
        if not ok:
            problems_found += 1

    return problems_found == 0


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 5: CLASSICAL BASELINE (BRUTE-FORCE) SOLVER")
    print("=" * 60)

    qubo_data = load_qubo_problem()

    penalty_ok = sanity_check_penalty(qubo_data)
    if not penalty_ok:
        print("\nWARNING: penalty weight may be too weak. Consider increasing "
              "penalty_weight in Step 4 and re-running Steps 4-5.")
    else:
        print("\nPenalty formulation verified correct.\n")

    best_combo, best_cost, all_results, elapsed = brute_force_solve(qubo_data)

    print("=" * 60)
    print("BRUTE-FORCE RESULTS (all valid 2-battery combinations, ranked)")
    print("=" * 60)
    for r in all_results:
        marker = " <-- BEST" if r["buses"] == best_combo else ""
        print(f"  Buses {r['buses']}: cost = {r['cost']:.4f}{marker}")

    print()
    print(f"OPTIMAL SOLUTION (classical, exhaustive search):")
    print(f"  Install batteries at buses: {best_combo}")
    print(f"  QUBO cost: {best_cost:.4f}")
    print(f"  Search time: {elapsed*1000:.3f} ms ({len(all_results)} combinations evaluated)")

    output = {
        "method": "classical_brute_force",
        "best_buses": best_combo,
        "best_cost": best_cost,
        "all_results": all_results,
        "runtime_seconds": elapsed,
        "num_combinations_evaluated": len(all_results),
        "penalty_sanity_check_passed": penalty_ok,
    }
    with open("../data/classical_baseline_result.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\nSaved classical baseline result to data/classical_baseline_result.json")

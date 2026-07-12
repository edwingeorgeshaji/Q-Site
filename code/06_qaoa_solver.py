"""
Q-Site | Step 6: Solve the QUBO using QAOA (Quantum Approximate Optimization Algorithm).

WHAT THIS STEP DOES (in plain terms):
--------------------------------------
Steps 1-5 built the problem and solved it classically (brute force = guaranteed
correct answer, since our problem is small). Now we solve the SAME problem
using a quantum algorithm called QAOA, and check whether it finds the same
answer.

WHAT IS QAOA, REALLY?
--------------------------------------
1. We convert our QUBO cost function into a "Hamiltonian" (H) — a sum of
   Pauli-Z operators, which is the standard way to represent a classical
   optimization cost function as something a quantum circuit can encode as
   "energy". Low energy = good solution, exactly mirroring low QUBO cost =
   good battery placement.
2. We build a QAOA circuit: alternating layers of a "cost" unitary
   (derived from H) and a "mixer" unitary (which lets the quantum state
   explore different bitstring combinations). This circuit has tunable
   parameters (beta, gamma).
3. A CLASSICAL optimizer (here: COBYLA, via scipy) sits outside the quantum
   circuit and repeatedly: (a) asks the quantum circuit to run with the
   current parameters, (b) measures the average energy, (c) adjusts the
   parameters to try to reduce that energy, and repeats. This back-and-forth
   is why QAOA is called a HYBRID quantum-classical algorithm.
4. Once optimization finishes, we sample the final circuit many times
   ("shots") and take the MOST FREQUENT bitstring as QAOA's answer.

TWO PLACES THIS CAN RUN:
--------------------------------------
1. SIMULATOR (this script, right now): a classical computer exactly simulates
   the quantum circuit. Fast, free, exact, no queue. This is Qiskit's
   StatevectorSampler. Ideal for development and for verifying correctness.
2. REAL QUANTUM HARDWARE (separate script, run later via qBraid): actual IBM
   quantum chips. Has real noise and a queue. We only do this once this
   simulator version is confirmed correct.

WHERE THE OUTPUT COMES FROM:
--------------------------------------
This uses Qiskit's real QAOAAnsatz circuit builder, a real Pauli-operator
Hamiltonian derived directly from our QUBO matrix, and Qiskit's
StatevectorSampler primitive (exact quantum circuit simulation) combined with
scipy's COBYLA optimizer for the classical parameter search. Nothing here is
precomputed; every energy value is obtained by actually simulating the
circuit.
"""

import json
import time
import warnings
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import SparseEfficiencyWarning

warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorSampler, StatevectorEstimator


def load_qubo_problem():
    with open("../data/qubo_problem.json") as f:
        return json.load(f)


def qubo_to_ising(Q_matrix):
    """
    Convert a QUBO matrix (binary variables x_i in {0,1}) into an Ising
    Hamiltonian (spin variables s_i in {-1,+1}), which is the form quantum
    circuits actually operate on. This is a standard, well-known transform:

        x_i = (1 - s_i) / 2

    Substituting this into x^T Q x and collecting terms gives a Hamiltonian
    of the form:  H = offset + sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j

    Returns: SparsePauliOp representing H, plus the constant offset (energy
    shift that doesn't affect which bitstring is optimal, but matters for
    reporting true QUBO cost values).
    """
    n = Q_matrix.shape[0]
    h = np.zeros(n)          # linear (single Z) coefficients
    J = np.zeros((n, n))     # quadratic (ZZ) coefficients
    offset = 0.0

    for i in range(n):
        offset += Q_matrix[i, i] / 2
        h[i] -= Q_matrix[i, i] / 2
        for j in range(i + 1, n):
            qij = Q_matrix[i, j] + Q_matrix[j, i]  # matrix stored upper-triangular
            offset += qij / 4
            h[i] -= qij / 4
            h[j] -= qij / 4
            J[i, j] += qij / 4

    # Build the SparsePauliOp term list
    pauli_list = []
    for i in range(n):
        if abs(h[i]) > 1e-12:
            label = ["I"] * n
            label[n - 1 - i] = "Z"  # Qiskit uses little-endian qubit ordering
            pauli_list.append(("".join(label), h[i]))
    for i in range(n):
        for j in range(i + 1, n):
            if abs(J[i, j]) > 1e-12:
                label = ["I"] * n
                label[n - 1 - i] = "Z"
                label[n - 1 - j] = "Z"
                pauli_list.append(("".join(label), J[i, j]))

    if not pauli_list:
        pauli_list = [("I" * n, 0.0)]

    H = SparsePauliOp.from_list(pauli_list)
    return H, offset


def run_qaoa(H, offset, reps=2, maxiter=100, seed=42, shots=4096):
    """
    Build and optimize a QAOA circuit for Hamiltonian H.

    reps: QAOA depth (p). p=2 is standard for small problems like ours.
    maxiter: cap on classical optimizer (COBYLA) iterations.
    shots: number of circuit measurements taken for the FINAL sampling step
           (to determine the answer bitstring). During optimization, we use
           the exact statevector expectation (no sampling noise) for a clean,
           reproducible parameter search — a standard simulator-stage choice.
    """
    rng = np.random.default_rng(seed)
    n = H.num_qubits

    ansatz = QAOAAnsatz(cost_operator=H, reps=reps)
    estimator = StatevectorEstimator()
    sampler = StatevectorSampler(seed=seed)

    eval_count = [0]

    def cost_function(params):
        eval_count[0] += 1
        job = estimator.run([(ansatz, H, params)])
        result = job.result()
        energy = result[0].data.evs
        return float(energy)

    # Random initial parameters (standard practice; QAOA landscape is
    # non-convex, so initial point matters, but COBYLA is reasonably robust
    # for small p here).
    x0 = rng.uniform(0, np.pi, size=ansatz.num_parameters)

    print(f"Optimizing {ansatz.num_parameters} QAOA parameters "
          f"(circuit depth p={reps}, {n} qubits)...")

    start = time.time()
    opt_result = minimize(cost_function, x0, method="COBYLA",
                           options={"maxiter": maxiter})
    elapsed = time.time() - start

    print(f"Classical optimizer finished: {eval_count[0]} circuit evaluations, "
          f"{elapsed:.3f}s, final energy = {opt_result.fun:.4f}")

    # Final sampling: bind optimized parameters, measure the circuit `shots`
    # times, and take the most frequent outcome as our answer.
    final_circuit = ansatz.assign_parameters(opt_result.x)
    final_circuit.measure_all()

    sampler_job = sampler.run([final_circuit], shots=shots)
    sampler_result = sampler_job.result()
    counts = sampler_result[0].data.meas.get_counts()

    return counts, opt_result, elapsed, eval_count[0]


def decode_best_bitstring(counts, candidate_buses, Q_matrix):
    """
    Among all measured bitstrings, evaluate the TRUE QUBO cost for each one
    (using the original QUBO matrix, not the Ising Hamiltonian) and return
    the best-scoring one. This is standard QAOA post-processing: sampling
    noise means the most-frequent bitstring isn't always the lowest-cost one,
    so we re-score every distinct sample against the real objective.
    """
    n = len(candidate_buses)
    best_cost = float("inf")
    best_bits = None
    scored = []

    for bitstring, count in counts.items():
        # Qiskit bitstrings are little-endian: rightmost char = qubit 0
        bits = bitstring[::-1]
        x = np.array([int(b) for b in bits])
        cost = float(x @ Q_matrix @ x)
        scored.append({"bitstring": bitstring, "count": count, "cost": cost,
                        "buses": [candidate_buses[i] for i in range(n) if x[i] == 1]})
        if cost < best_cost:
            best_cost = cost
            best_bits = x

    scored.sort(key=lambda r: r["cost"])
    selected_buses = [candidate_buses[i] for i in range(n) if best_bits[i] == 1]
    return selected_buses, best_cost, scored


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 6: QAOA QUANTUM SOLVER (STATEVECTOR SIMULATOR)")
    print("=" * 60)

    qubo_data = load_qubo_problem()
    candidate_buses = qubo_data["candidate_buses"]
    num_batteries = qubo_data["num_batteries"]
    Q_matrix = np.array(qubo_data["qubo_matrix"])

    print(f"\nProblem: choose {num_batteries} of {len(candidate_buses)} candidate buses")
    print(f"Candidate buses: {candidate_buses}")
    print(f"Qubits required: {len(candidate_buses)}")

    print("\nConverting QUBO to Ising Hamiltonian...")
    H, offset = qubo_to_ising(Q_matrix)
    print(f"Hamiltonian has {len(H)} Pauli terms, offset = {offset:.4f}")

    print()
    counts, opt_result, elapsed, n_evals = run_qaoa(H, offset, reps=2, maxiter=100, seed=42, shots=4096)

    print(f"\nFinal circuit sampled {sum(counts.values())} shots, "
          f"{len(counts)} distinct bitstrings observed.")

    selected_buses, best_cost, scored = decode_best_bitstring(counts, candidate_buses, Q_matrix)

    print()
    print("=" * 60)
    print("TOP 5 BITSTRINGS BY QUBO COST (from QAOA sampling)")
    print("=" * 60)
    for r in scored[:5]:
        print(f"  Buses {r['buses']}: cost={r['cost']:.4f}, "
              f"observed {r['count']}/{sum(counts.values())} shots")

    print()
    print("=" * 60)
    print("QAOA FINAL RESULT")
    print("=" * 60)
    print(f"Selected buses (QAOA, best-of-samples): {selected_buses}")
    print(f"QUBO cost:                               {best_cost:.4f}")
    print(f"Total wall-clock runtime:                {elapsed:.3f} seconds")
    print(f"Classical optimizer evaluations:         {n_evals}")
    print(f"Circuit depth (QAOA reps):               2")
    print(f"Qubit count:                              {len(candidate_buses)}")
    print(f"Final shots for sampling:                4096")

    # Compare to classical baseline
    with open("../data/classical_baseline_result.json") as f:
        classical = json.load(f)

    print()
    print("=" * 60)
    print("QAOA vs CLASSICAL BRUTE-FORCE COMPARISON")
    print("=" * 60)
    print(f"Classical optimal buses: {classical['best_buses']} (cost = {classical['best_cost']:.4f})")
    print(f"QAOA buses:              {selected_buses} (cost = {best_cost:.4f})")

    match = set(selected_buses) == set(classical["best_buses"])
    print(f"\nQAOA found the TRUE OPTIMUM: {'YES' if match else 'NO'}")
    gap = best_cost - classical["best_cost"]
    print(f"Optimality gap: {gap:.4f}")

    output = {
        "method": "QAOA_statevector_simulator",
        "reps": 2,
        "qubits": len(candidate_buses),
        "classical_optimizer": "COBYLA",
        "classical_optimizer_evals": n_evals,
        "backend": "Qiskit StatevectorEstimator + StatevectorSampler (exact simulation)",
        "shots_final_sampling": 4096,
        "selected_buses": selected_buses,
        "qubo_cost": best_cost,
        "runtime_seconds": elapsed,
        "matches_classical_optimum": match,
        "optimality_gap": gap,
        "classical_optimal_buses": classical["best_buses"],
        "classical_optimal_cost": classical["best_cost"],
        "top_5_sampled_solutions": scored[:5],
    }
    with open("../data/qaoa_simulator_result.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\nSaved QAOA simulator results to data/qaoa_simulator_result.json")

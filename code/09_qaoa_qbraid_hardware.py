"""
Q-Site | qBraid Hardware Execution Script
==========================================
Run this notebook/script on qBraid Lab, connected to your team's IBM Quantum
account (via the QC Request Form access granted by the GIC organizers).

WHAT THIS SCRIPT DOES:
--------------------------------------
This is the SAME QAOA battery-siting problem solved in Step 6 (simulator),
now configured to run on REAL IBM QUANTUM HARDWARE via qBraid. The QUBO
problem (6 qubits, choose 2-of-6 candidate buses for battery storage) is
embedded directly below — no external files needed, so this runs standalone
on qBraid exactly as submitted.

HOW TO USE THIS SCRIPT (overview — full click-by-click steps given separately):
--------------------------------------
1. Upload this file to qBraid Lab (or copy-paste into a new notebook cell).
2. Run Part A first (SETUP + VERIFY) — this only needs a simulator and
   costs nothing. It re-derives the classical answer, confirming this
   script's QUBO exactly matches Steps 4-5's verified problem.
3. Run Part B (HARDWARE CONNECTION) — this lists available IBM backends
   your account can access. No job is submitted yet, this just checks
   connectivity.
4. Run Part C (REAL HARDWARE EXECUTION) — this submits the actual QAOA
   circuit to a real IBM quantum computer via qBraid/IBM Runtime. This is
   the part that costs quantum credits and may sit in a queue.
5. Run Part D (RESULTS + COMPARISON) — decodes hardware results and
   compares against the classical/simulator answers.

IMPORTANT: Parts A and B are safe to run repeatedly (free, instant). Only
run Part C when you're ready to actually use quantum hardware time.
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
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


# =============================================================================
# EMBEDDED PROBLEM DATA (identical to data/qubo_problem.json from Steps 4-5)
# =============================================================================

CANDIDATE_BUSES = [9, 10, 8, 5, 13, 11]
NUM_BATTERIES = 2
BUS_INDEX = {9: 0, 10: 1, 8: 2, 5: 3, 13: 4, 11: 5}

QUBO_MATRIX = np.array([
    [-15.0931, 10.05,   10.05,   10.0,    10.0,    10.0   ],
    [  0.0,   -15.0596, 10.0,    10.05,   10.0,    10.0   ],
    [  0.0,     0.0,   -15.0,    10.0,    10.05,   10.0   ],
    [  0.0,     0.0,     0.0,   -15.0,    10.0,    10.05  ],
    [  0.0,     0.0,     0.0,     0.0,   -15.0,    10.0   ],
    [  0.0,     0.0,     0.0,     0.0,     0.0,   -15.0   ],
])

# Known correct answer (from Step 5 classical brute-force + Step 6 simulator
# QAOA, both independently verified to agree): buses [9, 10], cost -20.1027
EXPECTED_BUSES = [9, 10]
EXPECTED_COST = -20.1027


# =============================================================================
# SHARED FUNCTIONS (same logic as Step 6, reused here for consistency)
# =============================================================================

def qubo_to_ising(Q_matrix):
    """Convert QUBO matrix to Ising Hamiltonian (SparsePauliOp) + constant offset."""
    n = Q_matrix.shape[0]
    h = np.zeros(n)
    J = np.zeros((n, n))
    offset = 0.0

    for i in range(n):
        offset += Q_matrix[i, i] / 2
        h[i] -= Q_matrix[i, i] / 2
        for j in range(i + 1, n):
            qij = Q_matrix[i, j] + Q_matrix[j, i]
            offset += qij / 4
            h[i] -= qij / 4
            h[j] -= qij / 4
            J[i, j] += qij / 4

    pauli_list = []
    for i in range(n):
        if abs(h[i]) > 1e-12:
            label = ["I"] * n
            label[n - 1 - i] = "Z"
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

    return SparsePauliOp.from_list(pauli_list), offset


def decode_best_bitstring(counts, candidate_buses, Q_matrix, physical_qubit_layout=None):
    """
    Score every measured bitstring against the TRUE QUBO cost, return the best.

    physical_qubit_layout: on REAL HARDWARE, the transpiler maps our small
    N-qubit circuit onto specific physical qubits on a much larger chip (e.g.
    156 qubits on ibm_kingston). The measured bitstring therefore has one bit
    per PHYSICAL qubit on the whole chip, not one bit per our original logical
    variable. This parameter (from isa_ansatz.layout.final_index_layout())
    tells us which physical-qubit position corresponds to each of our
    original logical qubits, so we can extract just the 6 bits that matter.
    On the SIMULATOR (no transpilation to a larger chip), leave this as None
    and the bitstring is used directly, one bit per logical qubit.
    """
    n = len(candidate_buses)
    best_cost = float("inf")
    best_bits = None
    scored = []

    for bitstring, count in counts.items():
        # Qiskit bitstrings are little-endian: rightmost char = qubit 0.
        # Reverse once so index i below = qubit i (physical, if a layout is given).
        bits = bitstring[::-1]

        if physical_qubit_layout is not None:
            # Pull out only the bits at our logical qubits' physical positions,
            # in logical-qubit order (0..n-1).
            x = np.array([int(bits[physical_qubit_layout[i]]) for i in range(n)])
        else:
            x = np.array([int(b) for b in bits[:n]])

        cost = float(x @ Q_matrix @ x)
        scored.append({"bitstring": bitstring, "count": count, "cost": cost,
                        "buses": [candidate_buses[i] for i in range(n) if x[i] == 1]})
        if cost < best_cost:
            best_cost = cost
            best_bits = x

    scored.sort(key=lambda r: r["cost"])
    selected_buses = [candidate_buses[i] for i in range(n) if best_bits[i] == 1]
    return selected_buses, best_cost, scored


# =============================================================================
# PART A: SETUP + VERIFY (runs on simulator, free, instant — run this first)
# =============================================================================

def part_a_verify_problem():
    print("=" * 70)
    print("PART A: VERIFYING QUBO PROBLEM (simulator, no hardware needed)")
    print("=" * 70)

    H, offset = qubo_to_ising(QUBO_MATRIX)
    print(f"Hamiltonian built: {len(H)} Pauli terms, offset={offset:.4f}")

    # Quick simulator QAOA run to confirm this script's problem definition
    # matches Steps 4-5's verified result before spending real hardware time.
    ansatz = QAOAAnsatz(cost_operator=H, reps=2)
    estimator = StatevectorEstimator()
    sampler = StatevectorSampler(seed=42)

    def cost_fn(params):
        job = estimator.run([(ansatz, H, params)])
        return float(job.result()[0].data.evs)

    rng = np.random.default_rng(42)
    x0 = rng.uniform(0, np.pi, size=ansatz.num_parameters)
    opt_result = minimize(cost_fn, x0, method="COBYLA", options={"maxiter": 100})

    final_circuit = ansatz.assign_parameters(opt_result.x)
    final_circuit.measure_all()
    sampler_job = sampler.run([final_circuit], shots=4096)
    counts = sampler_job.result()[0].data.meas.get_counts()

    selected_buses, best_cost, scored = decode_best_bitstring(counts, CANDIDATE_BUSES, QUBO_MATRIX)

    print(f"\nSimulator re-derivation result: buses={selected_buses}, cost={best_cost:.4f}")
    print(f"Expected (from Steps 4-6):      buses={EXPECTED_BUSES}, cost={EXPECTED_COST:.4f}")

    matches = set(selected_buses) == set(EXPECTED_BUSES)
    if matches:
        print("\n[PASS] This script's QUBO matches the verified Steps 4-6 problem.")
        print("Safe to proceed to Part B / Part C.")
    else:
        print("\n[FAIL] MISMATCH — do not proceed to real hardware until this is fixed.")
        print("Check that QUBO_MATRIX above was copied correctly from data/qubo_problem.json.")

    return matches, H, offset


# =============================================================================
# PART B: CHECK HARDWARE CONNECTION (free, instant, no job submitted)
# =============================================================================

def part_b_check_hardware_connection():
    print("=" * 70)
    print("PART B: CHECKING IBM QUANTUM HARDWARE CONNECTION VIA qBraid")
    print("=" * 70)
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError:
        print("[ERROR] qiskit-ibm-runtime is not installed in this qBraid environment.")
        print("Run this in a qBraid Lab cell first:")
        print("    !pip install qiskit-ibm-runtime")
        return None

    try:
        # On qBraid, credentials are typically pre-configured for your account.
        # If this fails, see the "Troubleshooting" section in the instructions.
        service = QiskitRuntimeService()
        backends = service.backends()
        print(f"\nConnected successfully. {len(backends)} backend(s) visible to this account:")
        for b in backends:
            try:
                status = b.status()
                print(f"  - {b.name}: {b.num_qubits} qubits, "
                      f"queue={status.pending_jobs}, operational={status.operational}")
            except Exception:
                print(f"  - {b.name}: {b.num_qubits} qubits")
        return service
    except Exception as e:
        print(f"[ERROR] Could not connect to IBM Quantum service: {e}")
        print("\nCommon causes:")
        print("  1. QC access not yet granted (check email from GIC/qBraid team)")
        print("  2. Not signed into qBraid with the same email used on the QC Request Form")
        print("  3. IBM Quantum account not linked in qBraid account settings")
        return None


# =============================================================================
# PART C: REAL HARDWARE EXECUTION (costs quantum credits — run deliberately)
# =============================================================================

def part_c_run_on_hardware(service, H, min_qubits=6, reps=2, maxiter=12, shots=2048,
                             max_wallclock_seconds=420):
    """
    max_wallclock_seconds: hard safety cutoff (default 420s = 7 minutes), leaving
    a buffer under IBM's free-tier 10-runtime-minute monthly limit. If the
    optimization loop is still running past this, we abort early and use the
    best parameters found so far rather than risk running out of free minutes
    mid-job (which could leave a half-finished, unusable result).
    """
    print("=" * 70)
    print("PART C: RUNNING QAOA ON REAL IBM QUANTUM HARDWARE")
    print("=" * 70)

    from qiskit_ibm_runtime import SamplerV2, EstimatorV2

    print(f"\nSelecting least-busy backend with >= {min_qubits} qubits...")
    backend = service.least_busy(min_num_qubits=min_qubits, operational=True)
    print(f"Selected backend: {backend.name} ({backend.num_qubits} qubits)")

    ansatz = QAOAAnsatz(cost_operator=H, reps=reps)

    # Real hardware requires transpiling to the backend's native gate set
    # and qubit connectivity — this step was NOT needed for the simulator
    # version in Step 6, since the simulator accepts any gate directly.
    print("Transpiling circuit for target hardware...")
    pm = generate_preset_pass_manager(optimization_level=2, backend=backend)
    isa_ansatz = pm.run(ansatz)
    isa_H = H.apply_layout(isa_ansatz.layout)
    print(f"Transpiled circuit depth: {isa_ansatz.depth()}, "
          f"gate count: {sum(isa_ansatz.count_ops().values())}")

    estimator = EstimatorV2(mode=backend)
    sampler = SamplerV2(mode=backend)

    eval_count = [0]
    best_seen = {"params": None, "energy": float("inf")}
    run_start = time.time()

    class TimeoutStop(Exception):
        pass

    def cost_function(params):
        elapsed_so_far = time.time() - run_start
        if elapsed_so_far > max_wallclock_seconds:
            print(f"  [TIMEOUT GUARD] {elapsed_so_far:.0f}s elapsed, exceeds "
                  f"{max_wallclock_seconds}s budget. Stopping early.")
            raise TimeoutStop()

        eval_count[0] += 1
        job = estimator.run([(isa_ansatz, isa_H, params)])
        result = job.result()
        energy = float(result[0].data.evs)
        print(f"  Hardware eval #{eval_count[0]}: energy={energy:.4f} "
              f"(elapsed {elapsed_so_far:.0f}s)")
        if energy < best_seen["energy"]:
            best_seen["energy"] = energy
            best_seen["params"] = params.copy()
        return energy

    rng = np.random.default_rng(42)
    x0 = rng.uniform(0, np.pi, size=isa_ansatz.num_parameters)

    print(f"\nStarting hybrid optimization loop on real hardware "
          f"(maxiter={maxiter}, hard timeout={max_wallclock_seconds}s)...")
    print("Each evaluation below is a REAL quantum circuit execution.\n")

    start = time.time()
    try:
        opt_result = minimize(cost_function, x0, method="COBYLA",
                               options={"maxiter": maxiter})
        final_params = opt_result.x
        final_energy = opt_result.fun
    except TimeoutStop:
        print("  Using best parameters found before timeout.")
        final_params = best_seen["params"] if best_seen["params"] is not None else x0
        final_energy = best_seen["energy"]
    elapsed = time.time() - start

    print(f"\nOptimization phase finished: {eval_count[0]} hardware evaluations, "
          f"{elapsed:.1f}s wall-clock, best energy={final_energy:.4f}")

    print("\nRunning final sampling circuit on hardware...")
    final_circuit = isa_ansatz.assign_parameters(final_params)
    final_circuit.measure_all()
    sampler_job = sampler.run([final_circuit], shots=shots)
    sampler_result = sampler_job.result()
    counts = sampler_result[0].data.meas.get_counts()

    physical_qubit_layout = isa_ansatz.layout.final_index_layout()
    return counts, final_energy, elapsed, eval_count[0], backend.name, physical_qubit_layout


# =============================================================================
# PART D: DECODE + COMPARE RESULTS
# =============================================================================

def part_d_report_results(counts, final_energy, elapsed, n_evals, backend_name, physical_qubit_layout=None):
    print("=" * 70)
    print("PART D: HARDWARE RESULTS")
    print("=" * 70)

    selected_buses, best_cost, scored = decode_best_bitstring(
        counts, CANDIDATE_BUSES, QUBO_MATRIX, physical_qubit_layout=physical_qubit_layout
    )

    print(f"\nBackend used:              {backend_name}")
    print(f"Selected buses (hardware): {selected_buses}")
    print(f"QUBO cost:                 {best_cost:.4f}")
    print(f"Optimizer final energy:    {final_energy:.4f}")
    print(f"Wall-clock runtime:        {elapsed:.1f} seconds")
    print(f"Hardware circuit evals:    {n_evals}")
    print(f"\nExpected (classical/simulator optimum): buses={EXPECTED_BUSES}, cost={EXPECTED_COST:.4f}")

    match = set(selected_buses) == set(EXPECTED_BUSES)
    print(f"Hardware found the true optimum: {'YES' if match else 'NO (expected under real hardware noise)'}")

    print("\nTop 5 measured bitstrings by QUBO cost:")
    for r in scored[:5]:
        print(f"  Buses {r['buses']}: cost={r['cost']:.4f}, "
              f"observed {r['count']}/{sum(counts.values())} shots")

    output = {
        "method": "QAOA_real_hardware",
        "backend": backend_name,
        "selected_buses": selected_buses,
        "qubo_cost": best_cost,
        "runtime_seconds": elapsed,
        "hardware_circuit_evals": n_evals,
        "matches_expected_optimum": match,
        "expected_buses": EXPECTED_BUSES,
        "expected_cost": EXPECTED_COST,
        "top_5_sampled_solutions": scored[:5],
    }
    with open("qaoa_hardware_result.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved results to qaoa_hardware_result.json")
    return output


# =============================================================================
# MAIN — run parts in order
# =============================================================================

if __name__ == "__main__":
    # PART A — always run first (free, instant, verifies problem correctness)
    verified, H, offset = part_a_verify_problem()

    if not verified:
        print("\nStopping here — fix the QUBO mismatch before continuing.")
    else:
        print("\n" + "=" * 70)
        print("Part A passed. Uncomment Part B / Part C below when ready to")
        print("connect to and use real IBM quantum hardware.")
        print("=" * 70)

        # --- Uncomment when ready to check hardware connection (free) ---
        # service = part_b_check_hardware_connection()

        # --- Uncomment when ready to submit a real hardware job (uses credits) ---
        # if service is not None:
        #     counts, final_energy, elapsed, n_evals, backend_name, layout = part_c_run_on_hardware(service, H)
        #     part_d_report_results(counts, final_energy, elapsed, n_evals, backend_name, physical_qubit_layout=layout)

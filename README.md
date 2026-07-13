# Q-Site: Hybrid Quantum Resource Allocation for AI-Loaded Microgrids

> **Team:** kazi</p>
> **Challenge Track:** U.S. Department of Energy Office of Technology Commercialization — Energy Infrastructure</p>
> **Challenge:** Global Industry Challenge 2026, Phase 3</p>

**Team Members:**
- [Edwin George Shaji](https://github.com/edwingeorgeshaji) (POC) — Qiskit Integration & QAOA Execution
- [Jinsa Mariam Thomas](https://github.com/JinsaMariamThomas) — Python Backend & Graph Modelling
- [Mariam Skaria](https://github.com/MariamSkaria) — Documentation & Commercial Strategy
- [Malavika Krishnan](https://github.com/Malavika-Krishnan) — Classical UI & DB Architecture

---

## What This Project Does

Q-Site answers a concrete power-grid planning question: given a power grid stressed by a
large new AI data center load, **which buses should receive battery energy storage
(BESS)** to best preserve voltage stability under a defined set of single-line (N-1)
contingencies, subject to a fixed installation budget?

We formulate this as a Quadratic Unconstrained Binary Optimization (QUBO) problem, solve
it with the Quantum Approximate Optimization Algorithm (QAOA) — both on a simulator and
on real IBM Quantum hardware via qBraid — and validate the resulting battery placement
against real AC power flow physics (not just the QUBO's own score).

**Headline result:** QAOA (both simulator and real hardware) converges to the exact same
answer as exhaustive classical brute-force search — batteries at **Bus 9 and Bus 10** —
and installing storage there produces a measured **10.0% reduction in unsafe-voltage bus
incidents** across our 15-scenario contingency set, verified via pandapower's AC power
flow solver.

---

## Setup Instructions

### Environment / Dependencies

```bash
pip install qiskit qiskit-optimization qiskit-aer qiskit-ibm-runtime \
            pandapower networkx numpy scipy matplotlib --break-system-packages
```

Tested with:
- `qiskit` 2.5.0
- `qiskit-ibm-runtime` 0.47.0
- `pandapower` 3.5.4
- `networkx` 3.6.1
- `Python` 3.11+

No other external configuration, API keys, or environment variables are required to
reproduce the **simulator-based** results (Steps 1-8). Reproducing the **real hardware**
result (Step 9) requires a valid IBM Quantum account (free tier is sufficient) — see
"Reproducing the Real Hardware Result" below.

### Launch on qBraid

This project is qBraid-compatible out of the box. To run it on qBraid Lab:

1. Sign in to lab.qbraid.com
2. Launch a **Python 3 (ipykernel)** environment ("Small" tier is sufficient — this
   workload is not compute-heavy)
3. Upload the contents of the `code/` folder
4. Install dependencies (see above) in a notebook cell
5. Run scripts in order (see "How to Run" below)

[<img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_black.png" width="150">](https://account.qbraid.com?gitHubUrl=<https://github.com/edwingeorgeshaji/Q-Site.git>)

---

## How to Run

The pipeline is organized as 9 sequential scripts. Each writes its output to `data/` as
JSON, which the next script reads — so they must be run in order the first time. All
scripts are plain Python (`python code/NN_script_name.py`), runnable from a terminal or
a Jupyter `%run` cell.

| Step | Script | What it does | Needs hardware? |
|------|--------|---------------|:---:|
| 1 | `01_load_grid.py` | Loads the IEEE 14-bus test system (pandapower) | No |
| 2a | `02a_pick_ai_load_bus.py` | Selects the most electrically exposed bus for a synthetic AI data center load | No |
| 2b | `02b_add_ai_load.py` | Adds the AI load, runs AC power flow, confirms grid stress | No |
| 3 | `03_contingency_scenarios.py` | Runs full N-1 contingency analysis (15 scenarios) | No |
| 4 | `04_build_qubo.py` | Builds the battery-siting QUBO from contingency results | No |
| 5 | `05_classical_baseline.py` | Solves the QUBO via exhaustive classical search (ground truth) | No |
| 6 | `06_qaoa_solver.py` | Solves the same QUBO via QAOA on a local simulator | No |
| 7 | `07_validate_with_batteries.py` | Re-runs all 15 contingencies with batteries installed at the QAOA-selected buses, to validate real resilience improvement | No |
| 8 | `08_visualizations.py` | Generates the three figures used in the write-up | No |
| 9 | `09_qaoa_qbraid_hardware.py` | Re-solves the identical QUBO on **real IBM Quantum hardware** via qBraid | **Yes** |

Run steps 1-8 in order:

```bash
cd code
python 01_load_grid.py
python 02a_pick_ai_load_bus.py
python 02b_add_ai_load.py
python 03_contingency_scenarios.py
python 04_build_qubo.py
python 05_classical_baseline.py
python 06_qaoa_solver.py
python 07_validate_with_batteries.py
python 08_visualizations.py
```

Each script prints its results to stdout and writes JSON to `../data/`. Pre-computed
outputs from our own run are already included in `data/` and `results/figures/`, so you
can inspect results without re-running anything, or re-run to verify reproducibility.

### Reproducing the Real Hardware Result

Script 9 (`09_qaoa_qbraid_hardware.py`) is self-contained — the QUBO problem is embedded
directly in the file, so it does not depend on `data/qubo_problem.json` from Step 4 (this
guarantees it can be copied and run standalone on qBraid without the earlier steps).

1. Obtain a free IBM Quantum Platform account at quantum.cloud.ibm.com and create an
   "Open Plan" instance to get an API token and CRN (Cloud Resource Name).
2. In a qBraid notebook cell, save your credentials **locally in your own environment**
   (never share or commit this token):
   ```python
   from qiskit_ibm_runtime import QiskitRuntimeService
   QiskitRuntimeService.save_account(
       channel="ibm_quantum_platform",
       token="YOUR_TOKEN_HERE",
       instance="YOUR_CRN_HERE",
       overwrite=True
   )
   ```
3. Run the script in three stages (recommended, since Stage 3 uses real hardware time):
   ```python
   # Stage A - free, instant: verifies this script's embedded QUBO matches
   # the verified Steps 4-6 result before using any hardware time
   %run 09_qaoa_qbraid_hardware.py

   # Stage B - free, instant: confirms hardware connectivity
   from qsite_qaoa_qbraid import part_b_check_hardware_connection
   service = part_b_check_hardware_connection()

   # Stage C - uses real quantum hardware time
   from qsite_qaoa_qbraid import part_a_verify_problem, part_c_run_on_hardware, part_d_report_results
   _, H, offset = part_a_verify_problem()
   counts, final_energy, elapsed, n_evals, backend_name, layout = part_c_run_on_hardware(service, H)
   part_d_report_results(counts, final_energy, elapsed, n_evals, backend_name, physical_qubit_layout=layout)
   ```

**Our own hardware run** used `ibm_kingston` (156 qubits), completed 11 real hardware
circuit evaluations within a 420-second safety timeout (free-tier runtime is limited), and
converged to the same optimum as the classical/simulator results. Full results are saved
in `data/qaoa_hardware_result.json`.

---

## Expected Inputs / Outputs

**Inputs:** None required from the user — all grid data (IEEE 14-bus) is loaded
programmatically from `pandapower`'s built-in test case library. No external files, API
keys, or datasets need to be supplied for Steps 1-8.

**Outputs:** Each script writes one JSON file to `data/`, and Step 8 writes three PNG
figures to `results/figures/`. All outputs from our own run are included in this
submission for inspection without re-running.

**Key results at a glance:**
- Optimal battery sites (classical, simulator QAOA, and hardware QAOA — all agree):
  **Bus 9, Bus 10**
- QUBO cost at optimum: **-20.1027**
- Resilience improvement: unsafe-voltage-bus incidents reduced from 30 to 27 across 15
  contingency scenarios (**10.0% reduction**)
- Real hardware backend used: **ibm_kingston** (IBM Quantum, 156 qubits, via qBraid)

---

## Known Limitations and Assumptions

We state these explicitly, as required by the challenge guidelines:

1. **Storage is modeled as reactive-power injection only** (a static generator injecting
   MVAr for local voltage support), not real-power discharge, state-of-charge dynamics, or
   time-series dispatch. This means our siting objective can only address under-voltage
   problems, not over-voltage or thermal/loading problems. An earlier version of our QUBO
   scored both voltage directions symmetrically and produced a non-functional siting
   result (batteries placed at already-over-voltage buses); this was caught during
   internal validation and corrected — see Section 2.2 of the write-up for details.
2. **Two of our 15 contingency scenarios cause total voltage collapse** (loss of line 0-1
   or line 8-9) and are **not resolved** by our storage configuration — 15 MVAr of local
   reactive support at two buses is insufficient for these structurally critical outages.
   We report this honestly rather than omitting it.
3. **Candidate site pool is small by design** (6 buses), chosen for QAOA tractability on
   near-term hardware, not because larger candidate sets aren't relevant to real planning.
4. **N-1 contingencies only** — no N-2 or cascading-failure analysis.
5. **Real hardware run used a reduced optimizer iteration budget** (12 iterations capped,
   11 completed) relative to the simulator (100 iterations), due to free-tier quantum
   runtime limits. That the hardware run still converged to the exact global optimum is a
   positive result but should not be assumed to generalize to larger problem sizes without
   more iterations and/or error mitigation.

---

## Repository Structure

```
.
├── README.md                          (this file)
├── code/
│   ├── 01_load_grid.py
│   ├── 02a_pick_ai_load_bus.py
│   ├── 02b_add_ai_load.py
│   ├── 03_contingency_scenarios.py
│   ├── 04_build_qubo.py
│   ├── 05_classical_baseline.py
│   ├── 06_qaoa_solver.py
│   ├── 07_validate_with_batteries.py
│   ├── 08_visualizations.py
│   └── 09_qaoa_qbraid_hardware.py     (standalone, self-contained hardware script)
├── data/                              (JSON outputs from each step, pre-computed)
└── results/
    └── figures/                       (PNG figures used in the write-up)
```

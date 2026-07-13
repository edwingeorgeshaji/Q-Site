"""
Q-Site | Step 3: Build N-1 contingency scenarios.

WHAT THIS STEP DOES (in plain terms):
--------------------------------------
An "N-1 contingency" is standard power-industry language for: "take the grid,
knock out exactly ONE piece of equipment (one line), and check if the grid
survives." Planners use this because storms, equipment failures, and accidents
routinely take down single lines — the grid MUST be able to handle that
without collapsing.

We already know (from Step 2b) that the grid is stressed just from the AI
load alone. Now we ask a harder question: what happens if a storm ALSO knocks
out one of the lines near Bus 9, at the same time the AI load is active?

For each line in the grid, we:
  1. Temporarily disable that line (simulate it being knocked out)
  2. Re-run the power flow (with the AI load still active)
  3. Record whether the grid survives, and how bad the voltage damage is

This gives us a set of "scenario" grids — each one representing a different
disaster combination. This scenario SET is what the QUBO / quantum optimizer
will need to protect against later: a good battery placement should keep
voltages safe across AS MANY of these scenarios as possible, not just the
default case.

WHERE THE OUTPUT COMES FROM:
--------------------------------------
Every number below comes from pandapower's power flow solver (pp.runpp) —
the same physics engine used in Step 2b. It's not made up or hardcoded; each
scenario re-runs the actual electrical simulation with one line removed.
Nothing here touches a quantum computer yet — this is classical power-systems
simulation, which is the correct place for it: quantum only comes in once we
need to search over the SITING options (which comes next in Step 4).
"""

import pandapower as pp
import pandapower.networks as pn
import json

AI_LOAD_BUS = 9
AI_LOAD_MW = 80.0
AI_LOAD_MVAR = 25.0
VOLTAGE_MIN = 0.94
VOLTAGE_MAX = 1.06

def build_stressed_grid():
    """Recreate the IEEE 14-bus grid with the AI data center load added (from Step 2b)."""
    net = pn.case14()
    pp.create_load(net, bus=AI_LOAD_BUS, p_mw=AI_LOAD_MW, q_mvar=AI_LOAD_MVAR,
                    name="AI_Datacenter_Load")
    return net

def run_contingency(line_idx):
    """
    Take a fresh copy of the grid, disable ONE line (by index), and run
    power flow. Returns whether it converged and the resulting bus voltages.
    """
    net = build_stressed_grid()  # fresh grid each time, so contingencies don't stack
    net.line.at[line_idx, "in_service"] = False

    try:
        pp.runpp(net)
        voltages = net.res_bus["vm_pu"].to_dict()
        unsafe_buses = [
            bus for bus, v in voltages.items()
            if v < VOLTAGE_MIN or v > VOLTAGE_MAX
        ]
        return {
            "line_removed": int(line_idx),
            "from_bus": int(net.line.at[line_idx, "from_bus"]),
            "to_bus": int(net.line.at[line_idx, "to_bus"]),
            "converged": True,
            "voltages": {int(k): round(float(v), 4) for k, v in voltages.items()},
            "unsafe_buses": [int(b) for b in unsafe_buses],
            "num_unsafe_buses": len(unsafe_buses),
        }
    except Exception as e:
        return {
            "line_removed": int(line_idx),
            "from_bus": int(net.line.at[line_idx, "from_bus"]),
            "to_bus": int(net.line.at[line_idx, "to_bus"]),
            "converged": False,
            "error": str(e),
        }

def run_all_contingencies():
    base_net = build_stressed_grid()
    n_lines = len(base_net.line)

    print("=" * 60)
    print(f"RUNNING N-1 CONTINGENCY ANALYSIS ({n_lines} scenarios)")
    print("=" * 60)
    print(f"Base case: AI data center load ({AI_LOAD_MW} MW) active at Bus {AI_LOAD_BUS}")
    print(f"Each scenario removes ONE line and re-runs the power flow.\n")

    scenarios = []
    for line_idx in range(n_lines):
        result = run_contingency(line_idx)
        scenarios.append(result)

        if result["converged"]:
            status = f"{result['num_unsafe_buses']} unsafe buses"
        else:
            status = "DID NOT CONVERGE (voltage collapse)"
        print(f"Scenario {line_idx:2d} | Remove line {result['from_bus']}-{result['to_bus']:<3} | {status}")

    return scenarios

def summarize(scenarios):
    print()
    print("=" * 60)
    print("CONTINGENCY SUMMARY")
    print("=" * 60)

    converged = [s for s in scenarios if s["converged"]]
    failed = [s for s in scenarios if not s["converged"]]

    print(f"Total scenarios:        {len(scenarios)}")
    print(f"Converged (survived):   {len(converged)}")
    print(f"Did not converge (grid collapse): {len(failed)}")

    if converged:
        worst = max(converged, key=lambda s: s["num_unsafe_buses"])
        best = min(converged, key=lambda s: s["num_unsafe_buses"])
        print()
        print(f"WORST scenario: removing line {worst['from_bus']}-{worst['to_bus']} "
              f"-> {worst['num_unsafe_buses']} unsafe buses")
        print(f"BEST scenario:  removing line {best['from_bus']}-{best['to_bus']} "
              f"-> {best['num_unsafe_buses']} unsafe buses")

    if failed:
        print()
        print("Lines whose removal causes total voltage collapse:")
        for s in failed:
            print(f"  Line {s['from_bus']}-{s['to_bus']} (index {s['line_removed']})")

if __name__ == "__main__":
    scenarios = run_all_contingencies()
    summarize(scenarios)

    with open("../data/contingency_scenarios.json", "w") as f:
        json.dump(scenarios, f, indent=2)
    print()
    print("Saved all scenario data to data/contingency_scenarios.json")

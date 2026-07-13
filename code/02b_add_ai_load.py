"""
Q-Site | Step 2b: Add the synthetic AI data center load and check the grid still works.

WHAT THIS STEP DOES (in plain terms):
--------------------------------------
Right now our IEEE 14-bus grid is just the "normal" version — 259 MW of
ordinary household/city load spread across 11 buses. Real grids don't stay
static though: a new AI data center wants to plug in at Bus 9, and it wants
a LOT of power — way more than a normal neighborhood.

We're going to:
  1. Add a big new load (200 MW) at Bus 9 — simulating that data center.
  2. Run a "power flow" calculation. This is the standard physics simulation
     that tells us: given how much power everyone wants, and how the wires
     are connected, does electricity actually flow properly to everyone, or
     does something break (voltage collapse, overloaded line, etc.)?
  3. Check the results. If voltages stay in a safe range (roughly 0.94–1.06
     per unit, which is the industry-standard tolerance) and lines aren't
     overloaded, we call it "solved" — meaning the grid, as-is, can just
     barely handle the new AI load. If not, that PROVES our thesis: the grid
     needs help (i.e. batteries) to handle this new load safely, which is
     exactly the problem Q-Site exists to solve.

Why does this matter for the competition?
--------------------------------------
The judges want us to show real numbers, not just claims. This step gives us
our "before" picture — the stressed grid, without any battery help — which we
compare later against the "after" picture, once QAOA has picked the best
battery locations.
"""

import pandapower as pp
import pandapower.networks as pn
import json

AI_LOAD_BUS = 9          # picked in step 2a: most electrically "exposed" bus
AI_LOAD_MW = 80.0        # within the 50-500 MW range suggested by the challenge doc;
                         # scaled to this grid's ~259 MW base load so contingencies
                         # produce a realistic SPREAD of outcomes rather than total collapse
AI_LOAD_MVAR = 25.0      # reactive power component (standard ~0.3 power factor assumption)

def load_base_grid():
    return pn.case14()

def add_ai_datacenter_load(net, bus, p_mw, q_mvar):
    """
    Add a new load element to the network representing the AI data center.
    pandapower's create_load() just registers a new demand at a bus — it doesn't
    run any physics yet, that happens in the power flow step below.
    """
    pp.create_load(
        net,
        bus=bus,
        p_mw=p_mw,
        q_mvar=q_mvar,
        name="AI_Datacenter_Load"
    )
    return net

def run_power_flow(net):
    """
    Runs pandapower's AC power flow solver (Newton-Raphson method under the hood).
    This answers: "given all the loads and generators as configured, what do
    the actual voltages and line flows look like across the grid?"

    If this fails to converge, it usually means the grid physically cannot
    support the load as configured (voltage collapse) — a strong signal we
    NEED additional resources like storage at that location.
    """
    try:
        pp.runpp(net)
        return True, None
    except Exception as e:
        return False, str(e)

def summarize_stress_test(net, converged):
    print("=" * 60)
    print("POWER FLOW RESULTS — GRID WITH AI DATA CENTER LOAD ADDED")
    print("=" * 60)

    if not converged:
        print("RESULT: Power flow DID NOT CONVERGE.")
        print("This means the grid, as currently configured, cannot physically")
        print("deliver enough power to satisfy all loads — a voltage collapse")
        print("scenario. This is a strong 'Grid-Shock' result on its own.")
        return

    print("RESULT: Power flow converged (grid can technically deliver the power).")
    print()
    print("Bus voltages (per unit — normal safe range is ~0.94 to 1.06):")
    vres = net.res_bus[["vm_pu"]].copy()
    vres["status"] = vres["vm_pu"].apply(
        lambda v: "OK" if 0.94 <= v <= 1.06 else "**OUT OF SAFE RANGE**"
    )
    print(vres.to_string())

    print()
    print("Line loading (% of rated capacity — over 100% = overloaded):")
    lres = net.res_line[["loading_percent"]].copy()
    lres["status"] = lres["loading_percent"].apply(
        lambda l: "OK" if l <= 100 else "**OVERLOADED**"
    )
    print(lres.to_string())

    # Flag problem areas
    bad_voltage_buses = vres[(vres["vm_pu"] < 0.94) | (vres["vm_pu"] > 1.06)].index.tolist()
    overloaded_lines = lres[lres["loading_percent"] > 100].index.tolist()

    print()
    print("-" * 60)
    print("STRESS SUMMARY")
    print("-" * 60)
    print(f"Buses with unsafe voltage: {bad_voltage_buses if bad_voltage_buses else 'None'}")
    print(f"Overloaded lines:          {overloaded_lines if overloaded_lines else 'None'}")

    return {
        "bad_voltage_buses": bad_voltage_buses,
        "overloaded_lines": overloaded_lines,
        "bus_voltages": vres["vm_pu"].to_dict(),
        "line_loading": lres["loading_percent"].to_dict(),
    }

if __name__ == "__main__":
    print(f"Adding AI data center load: {AI_LOAD_MW} MW at Bus {AI_LOAD_BUS}\n")

    net = load_base_grid()
    net = add_ai_datacenter_load(net, AI_LOAD_BUS, AI_LOAD_MW, AI_LOAD_MVAR)

    converged, error = run_power_flow(net)

    if not converged:
        print("Power flow error:", error)
        result_data = {"converged": False, "error": error}
    else:
        result_data = summarize_stress_test(net, converged)
        result_data["converged"] = True

    with open("../data/stress_test_no_storage.json", "w") as f:
        json.dump(result_data, f, indent=2, default=str)

    print()
    print("Saved results to data/stress_test_no_storage.json")

    # Save the loaded network config for reuse in later steps
    pp.to_json(net, "../data/grid_with_ai_load.json")
    print("Saved grid state to data/grid_with_ai_load.json")

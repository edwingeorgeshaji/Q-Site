"""
Q-Site | Step 7: Validate the QAOA battery placement against real physics.

WHAT THIS STEP DOES (in plain terms):
--------------------------------------
Steps 4-6 SELECTED battery locations (Bus 5 and Bus 7) using a proxy
vulnerability score and a quantum algorithm. That selection process never
touched the actual power flow physics directly — it used a scoring shortcut
so the problem could be turned into a QUBO in the first place.

This step closes the loop: we go back to pandapower (the real physics
engine used in Steps 2-3) and ask the ACTUAL question: "if we install
batteries at Bus 5 and Bus 7, and re-run all 15 contingency scenarios,
does the grid actually get safer?"

This is the single most important number in the whole project, because it's
the difference between "quantum picked something" and "quantum picked
something that WORKS."

HOW A BATTERY IS MODELED HERE (stated explicitly, as the challenge doc
requires — "be transparent about assumptions used to simplify physics"):
--------------------------------------
Our Step 2b/3 stress tests showed a VOLTAGE problem, not a power/thermal
overload problem (no lines exceeded their rated capacity — see Step 2b
output). The standard, well-established power-systems mechanism for
localized voltage support is REACTIVE POWER (Q) injection. This is exactly
what real grid-connected battery inverters do in "volt/VAR support" mode,
and what real STATCOMs/capacitor banks do — no real-power discharge needed
to fix a voltage-sag problem.

We model each selected battery as a pandapower "static generator" (sgen)
injecting reactive power at its bus. This is a conservative, standard,
citable simplification (identical in spirit to how the challenge doc
describes DER/storage modeling for planning-level studies). We size the
injection at 15 MVAr per battery — a realistic mid-size grid-scale battery
inverter rating, well within what a 4-8 MWh containerized BESS unit
provides.

WHERE THE OUTPUT COMES FROM:
--------------------------------------
Every voltage number below comes from re-running pandapower's real AC power
flow solver (pp.runpp) — identical methodology and same 15 line-outage
scenarios as Step 3, just with sgen elements added at the QAOA-selected
buses. Nothing is estimated or interpolated.
"""

import json
import pandapower as pp
import pandapower.networks as pn

AI_LOAD_BUS = 9
AI_LOAD_MW = 80.0
AI_LOAD_MVAR = 25.0
VOLTAGE_MIN = 0.94
VOLTAGE_MAX = 1.06
BATTERY_QVAR_MVAR = 15.0   # reactive power injection per battery (voltage support)


def load_battery_sites():
    """Load QAOA's selected battery buses (falls back to classical if QAOA file missing)."""
    with open("../data/qaoa_simulator_result.json") as f:
        qaoa = json.load(f)
    return qaoa["selected_buses"], qaoa


def build_grid_with_batteries(battery_buses):
    """
    Recreate the IEEE 14-bus grid + AI data center load (same as Steps 2-3),
    then add a reactive-power-injecting static generator at each selected
    battery bus, representing grid-connected battery storage in voltage-
    support mode.
    """
    net = pn.case14()
    pp.create_load(net, bus=AI_LOAD_BUS, p_mw=AI_LOAD_MW, q_mvar=AI_LOAD_MVAR,
                    name="AI_Datacenter_Load")
    for bus in battery_buses:
        pp.create_sgen(
            net, bus=bus, p_mw=0.0, q_mvar=BATTERY_QVAR_MVAR,
            name=f"Battery_Storage_Bus{bus}"
        )
    return net


def run_contingency_with_batteries(line_idx, battery_buses):
    """Same methodology as Step 3, but on the grid-with-batteries version."""
    net = build_grid_with_batteries(battery_buses)
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


def run_all_contingencies_with_batteries(battery_buses):
    base_net = build_grid_with_batteries(battery_buses)
    n_lines = len(base_net.line)

    print(f"Re-running all {n_lines} N-1 contingency scenarios WITH batteries "
          f"at buses {battery_buses}...\n")

    results = []
    for line_idx in range(n_lines):
        r = run_contingency_with_batteries(line_idx, battery_buses)
        results.append(r)
        status = (f"{r['num_unsafe_buses']} unsafe buses" if r["converged"]
                   else "DID NOT CONVERGE (voltage collapse)")
        print(f"Scenario {line_idx:2d} | Remove line {r['from_bus']}-{r['to_bus']:<3} | {status}")

    return results


def load_baseline_scenarios():
    """Load the ORIGINAL (no-battery) contingency results from Step 3 for comparison."""
    with open("../data/contingency_scenarios.json") as f:
        return json.load(f)


def compare_before_after(before, after, battery_buses):
    print()
    print("=" * 70)
    print("BEFORE vs AFTER: BATTERY IMPACT ON GRID RESILIENCE")
    print("=" * 70)

    before_converged = sum(1 for s in before if s["converged"])
    after_converged = sum(1 for s in after if s["converged"])
    before_collapse = len(before) - before_converged
    after_collapse = len(after) - after_converged

    before_unsafe_total = sum(s["num_unsafe_buses"] for s in before if s["converged"])
    after_unsafe_total = sum(s["num_unsafe_buses"] for s in after if s["converged"])

    after_label = f"AFTER (Bus {', '.join(str(b) for b in battery_buses)})"
    print(f"{'Metric':<45}{'BEFORE (no battery)':<22}{after_label:<20}")
    print("-" * 87)
    print(f"{'Scenarios that collapse entirely':<45}{before_collapse:<22}{after_collapse:<20}")
    print(f"{'Scenarios that survive':<45}{before_converged:<22}{after_converged:<20}")
    print(f"{'Total unsafe-bus incidents (summed)':<45}{before_unsafe_total:<22}{after_unsafe_total:<20}")

    if before_unsafe_total > 0:
        pct_improvement = 100 * (before_unsafe_total - after_unsafe_total) / before_unsafe_total
        print(f"\nReduction in total unsafe-bus incidents: {pct_improvement:.1f}%")

    collapse_fixed = before_collapse - after_collapse
    print(f"Previously-collapsing scenarios now surviving: {collapse_fixed} "
          f"out of {before_collapse}")

    # Per-scenario detail table
    print()
    print("-" * 70)
    print("PER-SCENARIO DETAIL")
    print("-" * 70)
    print(f"{'Line Removed':<18}{'Before':<25}{'After':<25}")
    for b, a in zip(before, after):
        line_label = f"{b['from_bus']}-{b['to_bus']}"
        b_status = "COLLAPSE" if not b["converged"] else f"{b['num_unsafe_buses']} unsafe"
        a_status = "COLLAPSE" if not a["converged"] else f"{a['num_unsafe_buses']} unsafe"
        flag = "  <-- IMPROVED" if (b["converged"] and a["converged"] and
                                     a["num_unsafe_buses"] < b["num_unsafe_buses"]) or \
                                    (not b["converged"] and a["converged"]) else ""
        print(f"{line_label:<18}{b_status:<25}{a_status:<25}{flag}")

    return {
        "before_collapse_count": before_collapse,
        "after_collapse_count": after_collapse,
        "before_total_unsafe_incidents": before_unsafe_total,
        "after_total_unsafe_incidents": after_unsafe_total,
        "pct_reduction_unsafe_incidents": pct_improvement if before_unsafe_total > 0 else None,
        "collapse_scenarios_fixed": collapse_fixed,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("STEP 7: VALIDATING QAOA'S BATTERY PLACEMENT AGAINST REAL PHYSICS")
    print("=" * 70)

    battery_buses, qaoa_data = load_battery_sites()
    print(f"\nBattery sites selected by QAOA: {battery_buses}")
    print(f"(Matches classical optimum: {qaoa_data['matches_classical_optimum']})")
    print(f"Battery model: {BATTERY_QVAR_MVAR} MVAr reactive power injection per site "
          f"(voltage-support mode, no real-power discharge assumed)\n")

    after_scenarios = run_all_contingencies_with_batteries(battery_buses)
    before_scenarios = load_baseline_scenarios()

    comparison = compare_before_after(before_scenarios, after_scenarios, battery_buses)

    output = {
        "battery_buses": battery_buses,
        "battery_qvar_mvar_each": BATTERY_QVAR_MVAR,
        "after_scenarios": after_scenarios,
        "comparison_summary": comparison,
    }
    with open("../data/validation_with_batteries.json", "w") as f:
        json.dump(output, f, indent=2)

    print()
    print("Saved validation results to data/validation_with_batteries.json")

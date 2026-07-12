"""
Q-Site | Step 2a: Identify the best bus for the synthetic AI data center load.

Rationale: an AI data center load is most stressful to the grid — and most
realistic as a planning problem — if placed at a bus that is:
  1. Electrically "far" from generation (long path = more voltage/flow stress)
  2. Not a highly-connected hub (fewer alternate paths = less redundancy)
  3. Already a load bus (data centers connect at load/distribution points,
     not at generator buses)

We quantify this with NetworkX graph metrics on the IEEE 14-bus topology:
  - Degree centrality (lower = more "exposed")
  - Shortest-path distance to the nearest generator bus (higher = more stressed)
"""

import json
import networkx as nx
import pandapower.networks as pn

def load_graph():
    with open("../data/grid_graph.json") as f:
        data = json.load(f)
    G = nx.Graph()
    G.add_nodes_from(data["nodes"])
    G.add_edges_from(data["edges"])
    return G

def get_generator_buses(net):
    """IEEE 14-bus: bus 0 is the slack/ext_grid, plus gen buses. Pandapower is 0-indexed."""
    gen_buses = set(net.ext_grid["bus"].tolist()) | set(net.gen["bus"].tolist())
    return gen_buses

def get_existing_load_buses(net):
    return set(net.load["bus"].tolist())

def analyze_buses(G, gen_buses, load_buses):
    degree = dict(G.degree())

    # Shortest path distance (in hops) from each bus to nearest generator
    dist_to_gen = {}
    for node in G.nodes():
        dists = [nx.shortest_path_length(G, node, g) for g in gen_buses if nx.has_path(G, node, g)]
        dist_to_gen[node] = min(dists) if dists else float("inf")

    rows = []
    for node in sorted(G.nodes()):
        rows.append({
            "bus": node,
            "degree": degree[node],
            "dist_to_nearest_gen": dist_to_gen[node],
            "is_existing_load_bus": node in load_buses,
            "is_gen_bus": node in gen_buses,
        })
    return rows

def pick_ai_load_bus(rows):
    """
    Score = distance from generation (weighted higher) - degree (want low degree).
    Only consider existing load buses (realistic: data centers tap into
    distribution-level load buses, not generator buses).
    """
    candidates = [r for r in rows if r["is_existing_load_bus"] and not r["is_gen_bus"]]
    for r in candidates:
        r["score"] = 2 * r["dist_to_nearest_gen"] - r["degree"]
    candidates.sort(key=lambda r: r["score"], reverse=True)
    return candidates

if __name__ == "__main__":
    net = pn.case14()
    G = load_graph()
    gen_buses = get_generator_buses(net)
    load_buses = get_existing_load_buses(net)

    print("Generator buses (0-indexed):", sorted(gen_buses))
    print("Existing load buses (0-indexed):", sorted(load_buses))
    print()

    rows = analyze_buses(G, gen_buses, load_buses)
    print(f"{'Bus':<5}{'Degree':<8}{'DistToGen':<12}{'ExistingLoad':<14}{'IsGenBus':<10}")
    for r in rows:
        print(f"{r['bus']:<5}{r['degree']:<8}{r['dist_to_nearest_gen']:<12}{str(r['is_existing_load_bus']):<14}{str(r['is_gen_bus']):<10}")

    print()
    print("=" * 60)
    print("CANDIDATE RANKING FOR AI DATA CENTER PLACEMENT")
    print("=" * 60)
    candidates = pick_ai_load_bus(rows)
    for r in candidates:
        print(f"Bus {r['bus']}: score={r['score']}, degree={r['degree']}, dist_to_gen={r['dist_to_nearest_gen']}")

    best = candidates[0]
    print()
    print(f"SELECTED: Bus {best['bus']} (degree={best['degree']}, "
          f"{best['dist_to_nearest_gen']} hops from nearest generator)")

    with open("../data/ai_load_bus.json", "w") as f:
        json.dump({"ai_load_bus": best["bus"], "rationale": best}, f, indent=2)

"""
Q-Site | Step 1: Load the IEEE 14-bus test system.

This is a standard, publicly available power grid test case used widely in
academic and industry power systems research. We use pandapower's built-in
implementation, which is citable and reproducible.

Reference: IEEE 14-bus system, originally from the IEEE Reliability Test System.
"""

import pandapower as pp
import pandapower.networks as pn
import networkx as nx
import json

def load_ieee14():
    """Load the IEEE 14-bus system and return the pandapower network object."""
    net = pn.case14()
    return net

def summarize_grid(net):
    """Print a summary of the grid structure."""
    print("=" * 60)
    print("IEEE 14-BUS SYSTEM SUMMARY")
    print("=" * 60)
    print(f"Number of buses:        {len(net.bus)}")
    print(f"Number of lines:        {len(net.line)}")
    print(f"Number of transformers: {len(net.trafo)}")
    print(f"Number of generators:   {len(net.gen) + len(net.ext_grid)}")
    print(f"Number of loads:        {len(net.load)}")
    print()
    print("Bus list:")
    print(net.bus[["name", "vn_kv"]].to_string())
    print()
    print("Existing loads (bus, MW):")
    print(net.load[["bus", "p_mw"]].to_string())
    return

def build_graph(net):
    """Convert the pandapower network into a NetworkX graph for QUBO encoding."""
    G = nx.Graph()
    for i in net.bus.index:
        G.add_node(int(i))
    for _, row in net.line.iterrows():
        G.add_edge(int(row["from_bus"]), int(row["to_bus"]), element="line")
    for _, row in net.trafo.iterrows():
        G.add_edge(int(row["hv_bus"]), int(row["lv_bus"]), element="trafo")
    return G

if __name__ == "__main__":
    net = load_ieee14()
    summarize_grid(net)

    G = build_graph(net)
    print()
    print("=" * 60)
    print("GRAPH STRUCTURE (for QUBO encoding)")
    print("=" * 60)
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Edge list: {list(G.edges())}")

    # Save graph structure for downstream steps
    graph_data = {
        "nodes": list(G.nodes()),
        "edges": list(G.edges()),
    }
    with open("../data/grid_graph.json", "w") as f:
        json.dump(graph_data, f, indent=2)
    print()
    print("Saved grid graph to data/grid_graph.json")

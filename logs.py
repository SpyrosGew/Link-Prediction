# logs.py
import networkx as nx


def log_original_dgl_graph(dgl_graph):
    print("Original Graph (DGL)")
    print(" Number of nodes:", dgl_graph.num_nodes())
    print(" Number of edges:", dgl_graph.num_edges())
    print()


def log_networkx_graph(nx_graph, title="NetworkX Graph"):
    print(title)
    print(" Connected:", nx.is_connected(nx_graph))
    print(" Self-loops:", nx.number_of_selfloops(nx_graph))
    print(" Nodes:", nx_graph.number_of_nodes())
    print(" Edges:", nx_graph.number_of_edges())
    print()


def log_train_test_split(
    lcc_graph,
    train_graph,
    test_edges,
    bridges,
    negative_test_edges
):
    print("=== TRAIN / TEST SPLIT INFO ===")

    print("\nOriginal (LCC)")
    print(" Nodes:", lcc_graph.number_of_nodes())
    print(" Edges:", lcc_graph.number_of_edges())
    print(" Connected:", nx.is_connected(lcc_graph))
    print(" Self-loops:", nx.number_of_selfloops(lcc_graph))

    print("\nTrain graph")
    print(" Nodes:", train_graph.number_of_nodes())
    print(" Edges:", train_graph.number_of_edges())
    print(" Connected:", nx.is_connected(train_graph))
    print(" Self-loops:", nx.number_of_selfloops(train_graph))

    print("\nTest set")
    print(" Test edges:", len(test_edges))
    print(" Test ratio:", len(test_edges) / lcc_graph.number_of_edges())

    # Leakage check
    leak = sum(1 for u, v in test_edges if train_graph.has_edge(u, v))
    print(" Test edges still in train graph:", leak)

    # Bridge check
    removed_bridges = sum(1 for e in test_edges if e in bridges)
    print(" Bridges removed:", removed_bridges)

    # Degree sanity
    degrees = dict(train_graph.degree())
    print("\nDegree stats (train graph)")
    print(" Min degree:", min(degrees.values()))
    print(" Max degree:", max(degrees.values()))
    print(" Avg degree:", sum(degrees.values()) / len(degrees))

    print("\nConnected components (train graph):",
          nx.number_connected_components(train_graph))

    print("\nNegative sampling")
    print(" Positive test edges:", len(test_edges))
    print(" Negative test edges:", len(negative_test_edges))

    print("=== END CHECKS ===\n")

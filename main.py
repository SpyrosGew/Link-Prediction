import dgl
from dgl.data import CoraGraphDataset
import networkx as nx
import random

from logs import (
    log_original_dgl_graph,
    log_networkx_graph,
    log_train_test_split
)


dataset = CoraGraphDataset(verbose=False)
g = dataset[0]

log_original_dgl_graph(g)

nx_g = g.to_networkx().to_undirected()
log_networkx_graph(nx_g, "Original graph after NetworkX")

components = sorted(nx.connected_components(nx_g), key=len, reverse=True)
nx_lcc = nx_g.subgraph(components[0]).copy()
log_networkx_graph(nx_lcc, "Largest connected component")

train_graph = nx_lcc.copy()
test_edges = []

edges = list(train_graph.edges())
random.shuffle(edges)

num_test = int(0.1 * train_graph.number_of_edges())

for u, v in edges:
    if len(test_edges) >= num_test:
        break

    train_graph.remove_edge(u, v)

    if nx.is_connected(train_graph):
        test_edges.append((u, v))
    else:
        train_graph.add_edge(u, v)

test_edges_count = len(test_edges)
negative_test_edges = set()
nodes = list(nx_lcc.nodes())

while len(negative_test_edges) < test_edges_count:
    u, v = random.sample(nodes, 2)

    if (
        u != v
        and not nx_lcc.has_edge(u, v)
        and (u, v) not in negative_test_edges
        and (v, u) not in negative_test_edges
    ):
        negative_test_edges.add((u, v))

negative_test_edges = list(negative_test_edges)

log_train_test_split(
    nx_lcc=nx_lcc,
    train_graph=train_graph,
    test_edges=test_edges,
    bridges=[],
    negative_test_edges=negative_test_edges
)


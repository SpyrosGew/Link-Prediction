import dgl
from dgl.data import CoraGraphDataset
import networkx as nx
import random

random.seed(42)  # reproducibility

from logs import (
    log_original_dgl_graph,
    log_networkx_graph,
    log_train_test_split
)

class Dataset:

    def __init__(self):
        dataset = CoraGraphDataset(verbose=False)
        self.dgl_graph = dataset[0]
        log_original_dgl_graph(self.dgl_graph)

        self.nx_graph = self.dgl_graph.to_networkx().to_undirected()
        log_networkx_graph(self.nx_graph, "Original graph after NetworkX")

    def prepare(self):
        self._find_largest_connected_subgraph()
        log_networkx_graph(self.lcc_graph, "Largest connected component")

        self._set_train_test()
        self._set_negative_edges()

        log_train_test_split(
            lcc_graph=self.lcc_graph,
            train_graph=self.train_graph,
            test_edges=self.test_edges,
            bridges=[],
            negative_test_edges=self.negative_test_edges
        )


    def _find_largest_connected_subgraph(self):
        components = sorted(nx.connected_components(self.nx_graph), key=len, reverse=True)
        self.lcc_graph = self.nx_graph.subgraph(components[0]).copy()

    def _set_train_test(self):
        self.train_graph = self.lcc_graph.copy()
        self.test_edges = []

        edges = list(self.train_graph.edges())
        random.shuffle(edges)

        num_test = int(0.1 * self.train_graph.number_of_edges())

        for u, v in edges:
            if len(self.test_edges) >= num_test:
                break

            self.train_graph.remove_edge(u, v)

            if nx.is_connected(self.train_graph):
                self.test_edges.append((u, v))
            else:
                self.train_graph.add_edge(u, v)

    def _set_negative_edges(self):
        test_edges_count = len(self.test_edges)
        self.negative_test_edges = set()
        nodes = list(self.lcc_graph.nodes())

        while len(self.negative_test_edges) < test_edges_count:
            u, v = random.sample(nodes, 2)

            if (
                u != v
                and not self.train_graph.has_edge(u, v)  
                and (u, v) not in self.test_edges         
                and (v, u) not in self.test_edges
                and (u, v) not in self.negative_test_edges
                and (v, u) not in self.negative_test_edges
            ):
                self.negative_test_edges.add((u, v))

        self.negative_test_edges = list(self.negative_test_edges)

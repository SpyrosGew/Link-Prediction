import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv
from sklearn.metrics import roc_auc_score


class GCNEncoder(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super().__init__()

        self.conv1 = GraphConv(in_feats, hidden_feats)
        self.conv2 = GraphConv(hidden_feats, out_feats)

    def forward(self, g, x):
        """
        g: DGLGraph
        x: node features (N x in_feats)
        """
        h = self.conv1(g, x)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h  # Z: node embeddings


def dot_predict(z, edges):
    src = torch.tensor([u for u, v in edges])
    dst = torch.tensor([v for u, v in edges])
    return (z[src] * z[dst]).sum(dim=1)

def sample_negative_edges(graph, num_samples):
    nodes = list(graph.nodes())
    neg = set()
    edges = set(graph.edges()) | {(v, u) for u, v in graph.edges()}

    while len(neg) < num_samples:
        u, v = random.sample(nodes, 2)
        if (u, v) not in edges:
            neg.add((u, v))

    return list(neg)



def train_gcn_link_predictor(dataset, epochs=200, lr=0.01):
    # ------------------------
    # Convert train graph to DGL with node features
    # ------------------------
    # Build features in the same order as sorted node IDs
    sorted_nodes = sorted(dataset.train_graph.nodes())
    feats = np.vstack([dataset.train_graph.nodes[n]['feat'] for n in sorted_nodes])
    
    g = dgl.from_networkx(dataset.train_graph)
    g = dgl.add_self_loop(g)
    x = torch.tensor(feats, dtype=torch.float)

    # Map NetworkX node IDs to 0..N-1 indices (sorted order)
    nx_to_dgl = {n: i for i, n in enumerate(sorted_nodes)}

    # Model & optimizer
    model = GCNEncoder(in_feats=x.shape[1], hidden_feats=64, out_feats=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Map positive edges to DGL indices
    pos_edges = [(nx_to_dgl[u], nx_to_dgl[v]) for u, v in dataset.train_graph.edges()]

    # ------------------------
    # Training loop
    # ------------------------
    for epoch in range(epochs):
        model.train()
        z = model(g, x)

        # Online negative sampling
        neg_edges_nx = sample_negative_edges(dataset.train_graph, len(pos_edges))
        neg_edges = [(nx_to_dgl[u], nx_to_dgl[v]) for u, v in neg_edges_nx]

        # Scores
        pos_scores = dot_predict(z, pos_edges)
        neg_scores = dot_predict(z, neg_edges)

        # BCE loss
        loss = (
            F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores)) +
            F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

    return model


def evaluate_gcn(model, dataset):
    # Features
    sorted_nodes = sorted(dataset.train_graph.nodes())
    feats = np.vstack([dataset.train_graph.nodes[n]['feat'] for n in sorted_nodes])
    g = dgl.from_networkx(dataset.train_graph)
    g = dgl.add_self_loop(g)
    x = torch.tensor(feats, dtype=torch.float)
    nx_to_dgl = {n: i for i, n in enumerate(sorted_nodes)}

    model.eval()
    with torch.no_grad():
        z = model(g, x)

        # Map test edges to DGL indices
        pos_test = [(nx_to_dgl[u], nx_to_dgl[v]) for u, v in dataset.test_edges]
        neg_test = [(nx_to_dgl[u], nx_to_dgl[v]) for u, v in dataset.negative_test_edges]

        scores = torch.cat([
            torch.sigmoid(dot_predict(z, pos_test)),
            torch.sigmoid(dot_predict(z, neg_test))
        ])

        labels = torch.cat([
            torch.ones(len(pos_test)),
            torch.zeros(len(neg_test))
        ])

    return roc_auc_score(labels.numpy(), scores.numpy())
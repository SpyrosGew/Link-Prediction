import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import os


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


def save_gcn_plots(losses, train_aucs, test_aucs, eval_epochs, save_dir='figs/'):
    """Save loss and AUC plots to disk."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot 1: Loss over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(eval_epochs, losses, linewidth=2, color='steelblue', marker='o', markersize=5)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('GCN Training Loss', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gcn_loss.png'), dpi=300)
    plt.close()
    
    # Plot 2: Train and Test AUC over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(eval_epochs, train_aucs, label='Train AUC', linewidth=2, color='green', marker='o', markersize=5)
    plt.plot(eval_epochs, test_aucs, label='Test AUC', linewidth=2, color='red', marker='s', markersize=5)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('AUC', fontsize=12)
    plt.title('GCN Train and Test AUC', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gcn_auc.png'), dpi=300)
    plt.close()
    
    print(f"\nPlots saved to '{save_dir}'")


def train_gcn_link_predictor(dataset, epochs=500, lr=0.01):
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

    # Track metrics (only every 20 epochs)
    losses = []
    train_aucs = []
    test_aucs = []
    eval_epochs = []

    # Pre-compute for evaluation
    pos_test = [(nx_to_dgl[u], nx_to_dgl[v]) for u, v in dataset.test_edges]
    neg_test = [(nx_to_dgl[u], nx_to_dgl[v]) for u, v in dataset.negative_test_edges]
    pos_train = [(nx_to_dgl[u], nx_to_dgl[v]) for u, v in dataset.train_graph.edges()]

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
            losses.append(loss.item())
            eval_epochs.append(epoch)
            
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                z = model(g, x)
                
                # Test AUC
                test_scores = torch.cat([
                    torch.sigmoid(dot_predict(z, pos_test)),
                    torch.sigmoid(dot_predict(z, neg_test))
                ])
                test_labels = torch.cat([
                    torch.ones(len(pos_test)),
                    torch.zeros(len(neg_test))
                ])
                test_auc = roc_auc_score(test_labels.numpy(), test_scores.numpy())
                
                # Train AUC
                neg_train = [(nx_to_dgl[u], nx_to_dgl[v]) for u, v in sample_negative_edges(dataset.train_graph, len(pos_train))]
                train_scores = torch.cat([
                    torch.sigmoid(dot_predict(z, pos_train)),
                    torch.sigmoid(dot_predict(z, neg_train))
                ])
                train_labels = torch.cat([
                    torch.ones(len(pos_train)),
                    torch.zeros(len(neg_train))
                ])
                train_auc = roc_auc_score(train_labels.numpy(), train_scores.numpy())
                
                train_aucs.append(train_auc)
                test_aucs.append(test_auc)
                
                print(f"Train AUC: {train_auc:.4f} | Test AUC: {test_auc:.4f}")

    # Save plots with eval_epochs on x-axis
    save_gcn_plots(losses, train_aucs, test_aucs, eval_epochs)

    return model, losses, train_aucs, test_aucs


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

        # Test set AUC
        pos_test = [(nx_to_dgl[u], nx_to_dgl[v]) for u, v in dataset.test_edges]
        neg_test = [(nx_to_dgl[u], nx_to_dgl[v]) for u, v in dataset.negative_test_edges]

        test_scores = torch.cat([
            torch.sigmoid(dot_predict(z, pos_test)),
            torch.sigmoid(dot_predict(z, neg_test))
        ])
        test_labels = torch.cat([
            torch.ones(len(pos_test)),
            torch.zeros(len(neg_test))
        ])
        test_auc = roc_auc_score(test_labels.numpy(), test_scores.numpy())

        # Train set AUC
        pos_train = [(nx_to_dgl[u], nx_to_dgl[v]) for u, v in dataset.train_graph.edges()]
        neg_train = [(nx_to_dgl[u], nx_to_dgl[v]) for u, v in sample_negative_edges(dataset.train_graph, len(pos_train))]

        train_scores = torch.cat([
            torch.sigmoid(dot_predict(z, pos_train)),
            torch.sigmoid(dot_predict(z, neg_train))
        ])
        train_labels = torch.cat([
            torch.ones(len(pos_train)),
            torch.zeros(len(neg_train))
        ])
        train_auc = roc_auc_score(train_labels.numpy(), train_scores.numpy())

        print(f"Train AUC: {train_auc:.4f} | Test AUC: {test_auc:.4f}")
        return train_auc, test_auc


def evaluate_gcn_silent(model, dataset):
    """Silent evaluation without printing (for tracking during training)."""
    sorted_nodes = sorted(dataset.train_graph.nodes())
    feats = np.vstack([dataset.train_graph.nodes[n]['feat'] for n in sorted_nodes])
    g = dgl.from_networkx(dataset.train_graph)
    g = dgl.add_self_loop(g)
    x = torch.tensor(feats, dtype=torch.float)
    nx_to_dgl = {n: i for i, n in enumerate(sorted_nodes)}

    model.eval()
    with torch.no_grad():
        z = model(g, x)

        # Test set AUC
        pos_test = [(nx_to_dgl[u], nx_to_dgl[v]) for u, v in dataset.test_edges]
        neg_test = [(nx_to_dgl[u], nx_to_dgl[v]) for u, v in dataset.negative_test_edges]

        test_scores = torch.cat([
            torch.sigmoid(dot_predict(z, pos_test)),
            torch.sigmoid(dot_predict(z, neg_test))
        ])
        test_labels = torch.cat([
            torch.ones(len(pos_test)),
            torch.zeros(len(neg_test))
        ])
        test_auc = roc_auc_score(test_labels.numpy(), test_scores.numpy())

        # Train set AUC
        pos_train = [(nx_to_dgl[u], nx_to_dgl[v]) for u, v in dataset.train_graph.edges()]
        neg_train = [(nx_to_dgl[u], nx_to_dgl[v]) for u, v in sample_negative_edges(dataset.train_graph, len(pos_train))]

        train_scores = torch.cat([
            torch.sigmoid(dot_predict(z, pos_train)),
            torch.sigmoid(dot_predict(z, neg_train))
        ])
        train_labels = torch.cat([
            torch.ones(len(pos_train)),
            torch.zeros(len(neg_train))
        ])
        train_auc = roc_auc_score(train_labels.numpy(), train_scores.numpy())

        return train_auc, test_auc
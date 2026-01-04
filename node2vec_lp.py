import random
import numpy as np
from node2vec import Node2Vec
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score


def run_node2vec_link_prediction(dataset):
    # ------------------------
    # 1. Train Node2Vec on TRAIN GRAPH ONLY
    # ------------------------
    node2vec = Node2Vec(
        dataset.train_graph,
        dimensions=64,
        walk_length=40,
        num_walks=10,
        p=1,
        q=1,
        workers=4
    )

    model = node2vec.fit(window=10, min_count=1)

    # Node embeddings
    Z = {int(n): model.wv[str(n)] for n in dataset.train_graph.nodes()}

    def edge_embed(u, v):
        return Z[u] * Z[v]

    # ------------------------
    # 2. TRAIN SET (MLP)
    # ------------------------
    pos_train = list(dataset.train_graph.edges())
    nodes = list(dataset.train_graph.nodes())

    # Forbidden edges (no leakage)
    forbidden = (
        set(dataset.train_graph.edges()) |
        set(dataset.test_edges) |
        set(dataset.negative_test_edges) |
        {(v, u) for u, v in dataset.train_graph.edges()} |
        {(v, u) for u, v in dataset.test_edges} |
        {(v, u) for u, v in dataset.negative_test_edges}
    )

    neg_train = set()
    while len(neg_train) < len(pos_train):
        u, v = random.sample(nodes, 2)
        if (u, v) not in forbidden:
            neg_train.add((u, v))

    neg_train = list(neg_train)

    X_train = np.array([edge_embed(u, v) for u, v in pos_train + neg_train])
    y_train = np.array([1] * len(pos_train) + [0] * len(neg_train))

    # ------------------------
    # 3. TEST SET (fixed by dataset)
    # ------------------------
    pos_test = dataset.test_edges
    neg_test = dataset.negative_test_edges

    X_test = np.array([edge_embed(u, v) for u, v in pos_test + neg_test])
    y_test = np.array([1] * len(pos_test) + [0] * len(neg_test))

    # ------------------------
    # 4. MLP classifier
    # ------------------------
    clf = MLPClassifier(
        hidden_layer_sizes=(64,),
        max_iter=300,
        random_state=42
    )

    clf.fit(X_train, y_train)

    # ------------------------
    # 5. AUC evaluation
    # ------------------------
    scores = clf.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, scores)

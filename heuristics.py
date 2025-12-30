import math
import networkx as nx
from sklearn.metrics import roc_auc_score


def common_neighbors_score(G, u, v):
    return len(list(nx.common_neighbors(G, u, v)))


def jaccard_score(G, u, v):
    nu = set(G.neighbors(u))
    nv = set(G.neighbors(v))
    union = nu | nv
    if len(union) == 0:
        return 0.0
    return len(nu & nv) / len(union)


def adamic_adar_score(G, u, v):
    score = 0.0
    for w in nx.common_neighbors(G, u, v):
        deg = G.degree(w)
        if deg > 1:
            score += 1.0 / math.log(deg)
    return score

def evaluate_auc(G, pos_edges, neg_edges, score_func):
    y_true = []
    y_scores = []

    # positive edges
    for u, v in pos_edges:
        y_true.append(1)
        y_scores.append(score_func(G, u, v))

    # negative edges
    for u, v in neg_edges:
        y_true.append(0)
        y_scores.append(score_func(G, u, v))

    return roc_auc_score(y_true, y_scores)

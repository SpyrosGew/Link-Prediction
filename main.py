from dataset import Dataset
from node2vec_lp import run_node2vec_link_prediction
from gcn import train_gcn_link_predictor, evaluate_gcn

from heuristics import (
    common_neighbors_score,
    jaccard_score,
    adamic_adar_score,
    evaluate_auc
)


def main():
    # ------------------------
    # Dataset
    # ------------------------
    dataset = Dataset()
    dataset.prepare()

    # ------------------------
    # Heuristic baselines
    # ------------------------
    G = dataset.train_graph
    pos = dataset.test_edges
    neg = dataset.negative_test_edges

    auc_cn = evaluate_auc(G, pos, neg, common_neighbors_score)
    auc_jc = evaluate_auc(G, pos, neg, jaccard_score)
    auc_aa = evaluate_auc(G, pos, neg, adamic_adar_score)

    print("=== HEURISTIC BASELINES (AUC) ===")
    print(f"Common Neighbors : {auc_cn:.4f}")
    print(f"Jaccard Coef.    : {auc_jc:.4f}")
    print(f"Adamic-Adar     : {auc_aa:.4f}")

    # ------------------------
    # Node2Vec + MLP
    # ------------------------
    auc_n2v = run_node2vec_link_prediction(dataset)

    print("\n=== NODE2VEC + MLP ===")
    print(f"AUC : {auc_n2v:.4f}")

    # ------------------------
    # GCN end-to-end link prediction
    # ------------------------
    print("\n=== GCN LINK PREDICTION ===")

    gcn_model, losses, train_aucs, test_aucs = train_gcn_link_predictor(dataset)
    train_auc_final, test_auc_final = evaluate_gcn(gcn_model, dataset)

    print(f"\nFinal Test AUC: {test_auc_final:.4f}")


if __name__ == "__main__":
    main()

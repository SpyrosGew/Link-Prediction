from dataset import Dataset
from node2vec_lp import run_node2vec_link_prediction


from heuristics import (
    common_neighbors_score,
    jaccard_score,
    adamic_adar_score,
    evaluate_auc
)

def main():
    dataset = Dataset()
    dataset.prepare()

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

    auc_n2v = run_node2vec_link_prediction(dataset)

    print("\n=== NODE2VEC + MLP ===")
    print(f"AUC : {auc_n2v:.4f}")

if __name__ == "__main__":
    main()

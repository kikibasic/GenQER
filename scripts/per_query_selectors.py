"""The predictive and observational per-query selectors of Table II.

  python scripts/per_query_selectors.py predictive      # needs only results/per_query (scikit-learn)
  python scripts/per_query_selectors.py observational   # needs data/runs/main (Pyserini, trec_eval)

Predictive selector: class-balanced logistic regression over six features of the initial
BM25 retrieval, trained on four datasets and applied to the fifth (leave-one-dataset-out).
The label of a training query is the option with the highest nDCG@10; ties go to the first
option in the order no expansion, then the eight strategies in the order of Table I.

Observational selector: runs all nine options, scores each by the reranking scores of its
top 10 documents, and fuses the top-2 options by RRF. The paper reports the variant that
ranks the options by the maximum cosine similarity (obs_top2_rrf_qpp_max).

The LLM-judgment selector is exploratory and its instruction is not part of the release.
"""
import argparse
import os

import numpy as np
import pandas as pd

import common as C

FEATURES = ["probe_score_max", "probe_score_gap", "probe_hit_count",
            "query_len", "query_specificity", "vocab_overlap"]
# probe_score_max: BM25 score of the top-ranked document; probe_score_gap: score gap between
# ranks 1 and 10; probe_hit_count: number of returned documents (top 10); query_len: number
# of query terms excluding stopwords; query_specificity: mean IDF of the unique query terms;
# vocab_overlap: fraction of those terms that occur in any of the top 10 documents.
SEED = 42


def predictive():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    data = pd.read_csv(os.path.join(C.PER_QUERY, "strategy_validation9_perquery.csv"),
                       dtype={"qid": str}).dropna(subset=FEATURES).reset_index(drop=True)
    data["label"] = data[C.OPTIONS9].idxmax(axis=1)          # first maximum, n0 first
    pred = np.empty(len(data), dtype=object)
    for held_out in C.MAIN_DATASETS:
        train, test = data["dataset"].ne(held_out), data["dataset"].eq(held_out)
        scaler = StandardScaler().fit(data.loc[train, FEATURES])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
        clf.fit(scaler.transform(data.loc[train, FEATURES]), data.loc[train, "label"])
        pred[test] = clf.predict(scaler.transform(data.loc[test, FEATURES]))
    selected = np.array([data.iloc[i][pred[i]] for i in range(len(data))], dtype=float)
    strict = data["strict"].to_numpy()
    best = data[C.OPTIONS9].max(axis=1).to_numpy()

    stored = pd.read_csv(os.path.join(C.PER_QUERY, "router_selection_accuracy9_perquery.csv"),
                         dtype={"qid": str})
    same = (stored["selected_strategy"].to_numpy() == pred).mean()
    print(f"predictive selector: nDCG@10 {selected.mean():.4f}, strict {strict.mean():.4f}, "
          f"difference {selected.mean() - strict.mean():+.4f}, "
          f"p = {C.paired_perm_p(selected, strict):.4f}")
    print(f"  top-1 accuracy (any tied maximum counts) {np.mean(selected == best):.3f}; "
          f"predicts no expansion for {np.mean(pred == 'n0'):.3f} of the queries; "
          f"label is no expansion for {np.mean(data['label'] == 'n0'):.3f}")
    print(f"  same selections as router_selection_accuracy9_perquery.csv: {same:.3f}")


def _estimators(scores):
    s = np.array(scores[:10])
    return {"mean": s.mean(), "max": s[0], "nqc": s.std() / max(abs(s.mean()), 1e-6),
            "gap": (s[0] - s[1]) if len(s) > 1 else 0.0}


def observational(estimator="max"):
    from fuse_runs import RUN_NAME, rrf_fuse

    rows = []
    for ds in C.MAIN_DATASETS:
        runs = [C.read_run(os.path.join(C.DATA, "runs", "main", f"{ds}_{name}.run"))
                for name in RUN_NAME]
        qids = [q for q in C.qids_of(ds) if all(q in r for r in runs)]
        fused = {}
        for q in qids:
            quality = [_estimators([s for _, s in r[q]])[estimator] for r in runs]
            top2 = sorted(range(len(runs)), key=lambda i: -quality[i])[:2]
            fused[q] = rrf_fuse([[d for d, _ in runs[i][q]] for i in top2])
        sc = C.ndcg10_of_ranking(fused, ds, qids)
        rows += [{"dataset": ds, "qid": q, "ndcg10": sc[q]} for q in qids]
    got = pd.DataFrame(rows).set_index(["dataset", "qid"])["ndcg10"]

    stored = pd.read_csv(os.path.join(C.PER_QUERY, "review_postretrieval_perquery.csv"),
                         dtype={"qid": str})
    stored = stored[stored.method == f"obs_top2_rrf_qpp_{estimator}"].set_index(["dataset", "qid"])
    sv = pd.read_csv(os.path.join(C.PER_QUERY, "strategy_validation9_perquery.csv"),
                     dtype={"qid": str}).set_index(["dataset", "qid"])
    got = got.reindex(stored.index)                       # order used for the test in the paper
    strict = sv["strict"].reindex(stored.index)
    print(f"observational selector (top-2 by {estimator} + RRF): nDCG@10 {got.mean():.4f}, "
          f"difference from strict {got.mean() - strict.mean():+.4f}, "
          f"p = {C.paired_perm_p(got.values, strict.values):.4f}")
    print(f"  largest per-query deviation from review_postretrieval_perquery.csv: "
          f"{np.abs(got.values - stored['ndcg10'].values).max():.5f} (n = {len(got)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("selector", choices=["predictive", "observational"])
    ap.add_argument("--estimator", default="max", choices=["mean", "max", "nqc", "gap"],
                    help="quality indicator of the observational selector")
    args = ap.parse_args()
    if args.selector == "predictive":
        predictive()
    else:
        observational(args.estimator)


if __name__ == "__main__":
    main()

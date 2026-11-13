"""Retrieval-result fusion of the nine runs (no expansion + eight strategies): RRF and CombSUM.

Recomputes the per-query nDCG@10 behind the RRF and CombSUM columns of Table III and
compares it with results/per_query/fusion_all9_perquery.csv.

  python scripts/fuse_runs.py            # recompute and compare
  python scripts/fuse_runs.py --out fused_perquery.csv

RRF gives 1/(60 + r) to the document at rank r of each run and sums per document.
CombSUM sums the min-max normalized reranking scores (0 for a document absent from a run).
The fused top 100 are evaluated in the fused order.
"""
import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd

import common as C

RRF_K = 60
TOP = 100
RUN_NAME = ["no_expansion"] + C.STRATEGIES     # same order as C.OPTIONS9


def rrf_fuse(rankings):
    score = defaultdict(float)
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            score[doc] += 1.0 / (RRF_K + rank)
    return [d for d, _ in sorted(score.items(), key=lambda x: -x[1])][:TOP]


def combsum_fuse(score_lists):
    total = defaultdict(float)
    for lst in score_lists:
        sc = np.array([s for _, s in lst])
        lo, hi = sc.min(), sc.max()
        span = max(hi - lo, 1e-9)
        for d, s in lst:
            total[d] += (s - lo) / span
    return [d for d, _ in sorted(total.items(), key=lambda x: -x[1])][:TOP]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="write the recomputed per-query values to this CSV file")
    args = ap.parse_args()

    rows = []
    for ds in C.MAIN_DATASETS:
        qids = C.qids_of(ds)
        runs = [C.read_run(os.path.join(C.DATA, "runs", "main", f"{ds}_{name}.run"))
                for name in RUN_NAME]
        fused = {"rrf_all9": {}, "combsum_all9": {}}
        for q in qids:
            lists = [r[q] for r in runs if r.get(q)]
            fused["rrf_all9"][q] = rrf_fuse([[d for d, _ in lst] for lst in lists])
            fused["combsum_all9"][q] = combsum_fuse(lists)
        for method, ranking in fused.items():
            sc = C.ndcg10_of_ranking(ranking, ds, qids)
            rows += [{"dataset": ds, "qid": q, "method": method, "ndcg10": sc[q]} for q in qids]
        print(f"[{ds}] fused {len(qids)} queries")
    got = pd.DataFrame(rows)
    if args.out:
        got.to_csv(args.out, index=False)

    stored = pd.read_csv(os.path.join(C.PER_QUERY, "fusion_all9_perquery.csv"), dtype={"qid": str})
    m = got.merge(stored, on=["dataset", "qid", "method"], suffixes=("", "_stored"))
    for method, g in m.groupby("method"):
        print(f"{method}: pooled nDCG@10 {g.ndcg10.mean():.4f} (stored {g.ndcg10_stored.mean():.4f}), "
              f"largest per-query deviation {np.abs(g.ndcg10 - g.ndcg10_stored).max():.5f}, n = {len(g)}")


if __name__ == "__main__":
    main()

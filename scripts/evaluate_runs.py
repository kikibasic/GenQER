"""Evaluate TREC run files: per-query nDCG@10 (and other trec_eval metrics).

Two uses:

  # evaluate one run file on the queries used in the paper
  python scripts/evaluate_runs.py data/runs/main/scifact_strict.run --dataset scifact

  # re-evaluate every released run and compare with the per-query files in results/
  python scripts/evaluate_runs.py --check

Requires Pyserini (for the qrels) and trec_eval (a `trec_eval` binary on PATH, or the copy
bundled with Pyserini, which needs Java). Unzip the released runs into data/runs/ first.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

import common as C

RUNS = os.path.join(C.DATA, "runs")


def evaluate_file(path, dataset, metrics, ties="score"):
    qids = C.qids_of(dataset)
    if ties == "rank":
        if tuple(metrics) != ("ndcg_cut.10",):
            raise SystemExit("--ties rank supports nDCG@10 only")
        res = {"ndcg_cut_10": C.per_query_ndcg10(path, dataset, qids, ties="rank")}
    else:
        res = C.per_query_metrics(path, dataset, qids, metrics=metrics)
    df = pd.DataFrame({"dataset": dataset, "qid": qids})
    for name, vals in res.items():
        df[name] = [vals[q] for q in qids]
    return df


def _released(dataset, column, frame):
    g = frame[frame.dataset == dataset].set_index("qid")[column]
    return g


def check():
    """Compare every released run with the stored per-query nDCG@10 (4 decimals)."""
    read = lambda name: pd.read_csv(os.path.join(C.PER_QUERY, name), dtype={"qid": str})
    sv = read("strategy_validation9_perquery.csv")
    cs = read("corpus_strategy_perquery.csv")
    ho = read("heldout_confirmation_perquery.csv")
    fp = read("fusion_pool_perquery.csv")
    cc = read("count_control_perquery.csv")
    sp = read("subset_pool_perquery.csv")

    # (run path, dataset, stored per-query Series indexed by qid, tie handling)
    jobs = []
    for ds in C.MAIN_DATASETS + C.HELDOUT_DATASETS:
        main = ds in C.MAIN_DATASETS
        # heldout_confirmation_perquery.csv (held-out no expansion) keeps the file's rank order
        jobs.append((f"main/{ds}_no_expansion.run", ds, _released(ds, "n0", sv if main else ho),
                     "score" if main else "rank"))
        for s in C.STRATEGIES:
            jobs.append((f"main/{ds}_{s}.run", ds, _released(ds, C.CSV_NAME[s], sv if main else cs), "score"))
        all8 = (_released(ds, "ndcg10", fp[fp.method == "ctxpool_all8"]) if main
                else _released(ds, "all8", cs))
        jobs.append((f"main/{ds}_all8.run", ds, all8, "rank"))
    for ds in C.MAIN_DATASETS:
        for cond in sorted(cc.cond.unique()):
            jobs.append((f"count_control/{ds}_{cond}.run", ds,
                         _released(ds, "ndcg10", cc[cc.cond == cond]), "rank"))
        for path in sorted(glob.glob(os.path.join(RUNS, "subsets", f"{ds}_*.run"))):
            config = os.path.basename(path)[len(ds) + 1:-4]
            jobs.append((f"subsets/{ds}_{config}.run", ds,
                         _released(ds, "ndcg10", sp[sp.config == config]), "rank"))

    worst, n_bad = 0.0, 0
    for rel, ds, stored, ties in jobs:
        qids = C.qids_of(ds)
        got = C.per_query_ndcg10(os.path.join(RUNS, rel), ds, qids, ties=ties)
        diff = np.array([abs(got[q] - stored.get(q, np.nan)) for q in qids])
        dev = float(np.nanmax(diff)) if len(diff) else 0.0
        missing = int(np.isnan(diff).sum())
        worst = max(worst, dev)
        if dev > 5e-5 or missing:
            n_bad += 1
            print(f"MISMATCH {rel}: max |diff| = {dev:.4f}, queries without stored value = {missing}")
    print(f"{len(jobs)} run files re-evaluated; largest per-query deviation from "
          f"results/per_query = {worst:.5f}; files with a mismatch: {n_bad}")

    # Stage-wise analysis (Fig. 1) and the second reranker: compare pooled means.
    ref = C.reference_queries(C.MAIN_DATASETS)
    stage = pd.read_csv(os.path.join(C.TABLES, "stage_ablation_summary.csv"))
    conds = {"strict": "main/{ds}_strict.run",
             "stageA_strictBM25_all8rerank": "stage/{ds}_bm25-strict_rerank-all8.run",
             "stageB_all8BM25_strictrerank": "stage/{ds}_bm25-all8_rerank-strict.run",
             "all8": "main/{ds}_all8.run"}
    metrics = ("ndcg_cut.10", "recall.100", "recip_rank")
    for cond, pattern in conds.items():
        vals = {m.replace(".", "_"): [] for m in metrics}
        for ds in C.MAIN_DATASETS:
            qids = ref[ref.dataset == ds]["qid"].tolist()
            res = C.per_query_metrics(os.path.join(RUNS, pattern.format(ds=ds)), ds, qids, metrics)
            for m in vals:
                vals[m] += [res[m][q] for q in qids]
        for m in vals:
            stored = stage[(stage.metric == m) & (stage.cond == cond)]["mean"].iloc[0]
            flag = "" if abs(np.mean(vals[m]) - stored) < 5e-5 else "  <-- differs"
            print(f"stage {cond:30s} {m:12s} {np.mean(vals[m]):.4f} (stored {stored:.4f}){flag}")
    bge = pd.read_csv(os.path.join(C.TABLES, "reranker_robustness_summary.csv")).iloc[0]
    for cond in ("strict", "all8"):
        vals = []
        for ds in C.MAIN_DATASETS:
            qids = ref[ref.dataset == ds]["qid"].tolist()
            got = C.per_query_ndcg10(os.path.join(RUNS, "bge", f"{ds}_{cond}.run"), ds, qids, ties="rank")
            vals += [got[q] for q in qids]
        flag = "" if abs(np.mean(vals) - bge[cond]) < 5e-5 else "  <-- differs"
        print(f"bge-base-en-v1.5 {cond:6s} nDCG@10 {np.mean(vals):.4f} (stored {bge[cond]:.4f}){flag}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", help="run file to evaluate")
    ap.add_argument("--dataset", choices=C.MAIN_DATASETS + C.HELDOUT_DATASETS)
    ap.add_argument("--metrics", default="ndcg_cut.10", help="comma-separated trec_eval metrics")
    ap.add_argument("--ties", choices=["score", "rank"], default="score",
                    help="order of tied scores: trec_eval's (score) or the run file's ranks (rank)")
    ap.add_argument("--out", help="write the per-query values to this CSV file")
    ap.add_argument("--check", action="store_true", help="re-evaluate all released runs")
    args = ap.parse_args()
    if args.check:
        return check()
    if not args.run or not args.dataset:
        ap.error("give a run file and --dataset, or use --check")
    df = evaluate_file(args.run, args.dataset, tuple(args.metrics.split(",")), ties=args.ties)
    if args.out:
        df.to_csv(args.out, index=False)
    print(df.drop(columns=["dataset", "qid"]).mean().round(4).to_string())
    print(f"n = {len(df)} queries")


if __name__ == "__main__":
    main()

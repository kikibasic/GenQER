"""Shared constants and helpers for the GenQER scripts.

Layout assumed (relative to the repository root):
  data/generations/...   saved generations (from the GitHub Release ZIP)
  data/runs/...          TREC run files      (from the GitHub Release ZIP)
  results/per_query/...  per-query nDCG@10 of every method (in the repository)
  results/tables/...     aggregated numbers behind the paper's tables and figures
"""
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from math import comb

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
PER_QUERY = os.path.join(ROOT, "results", "per_query")
TABLES = os.path.join(ROOT, "results", "tables")

MAIN_DATASETS = ["covid", "nfc", "scifact", "dl19", "dl20"]
HELDOUT_DATASETS = ["news", "robust04"]

# Pyserini prebuilt index, topics, and qrels of each dataset.
PYSERINI_INDEX = {
    "covid": "beir-v1.0.0-trec-covid.flat",
    "nfc": "beir-v1.0.0-nfcorpus.flat",
    "scifact": "beir-v1.0.0-scifact.flat",
    "dl19": "msmarco-v1-passage",
    "dl20": "msmarco-v1-passage",
    "news": "beir-v1.0.0-trec-news.flat",
    "robust04": "beir-v1.0.0-robust04.flat",
}
PYSERINI_TOPICS = {
    "covid": "beir-v1.0.0-trec-covid-test",
    "nfc": "beir-v1.0.0-nfcorpus-test",
    "scifact": "beir-v1.0.0-scifact-test",
    "dl19": "dl19-passage",
    "dl20": "dl20",
    "news": "beir-v1.0.0-trec-news-test",
    "robust04": "beir-v1.0.0-robust04-test",
}
PYSERINI_QRELS = dict(PYSERINI_TOPICS, dl20="dl20-passage")

# The eight generation strategies, in the order of Table I of the paper.
STRATEGIES = ["legacy", "strict", "aspect", "hybrid", "broaden",
              "corpus_strict", "corpus_aspect", "corpus_broaden"]
# Column names used in the result files (corpus_strict is called "grounded" there).
CSV_NAME = {s: s for s in STRATEGIES}
CSV_NAME["corpus_strict"] = "grounded"
OPTIONS9 = ["n0"] + [CSV_NAME[s] for s in STRATEGIES]   # no expansion + eight strategies


# ---------------------------------------------------------------- queries
def reference_queries(datasets=None):
    """(dataset, qid) pairs in the fixed order used for the tests in the paper.

    The five main datasets come from strategy_validation9_perquery.csv (755 queries; its row
    order is the order of the per-query vectors used for the permutation tests). The two
    held-out datasets come from corpus_strategy_perquery.csv.
    """
    main = pd.read_csv(os.path.join(PER_QUERY, "strategy_validation9_perquery.csv"),
                       dtype={"qid": str})[["dataset", "qid"]]
    held = pd.read_csv(os.path.join(PER_QUERY, "corpus_strategy_perquery.csv"),
                       dtype={"qid": str})[["dataset", "qid"]]
    held = held[held.dataset.isin(HELDOUT_DATASETS)]
    ref = pd.concat([main, held], ignore_index=True)
    if datasets is not None:
        ref = ref[ref.dataset.isin(list(datasets))].reset_index(drop=True)
    return ref


def qids_of(dataset):
    ref = reference_queries([dataset])
    return ref["qid"].tolist()


# ---------------------------------------------------------------- generations and runs
def load_generations(dataset, strategy, group=None):
    """{qid: [passage, ...]} of one strategy. group: main / heldout (chosen automatically)."""
    if group is None:
        group = "heldout" if dataset in HELDOUT_DATASETS else "main"
    path = os.path.join(DATA, "generations", group, f"{dataset}_{strategy}.json")
    with open(path, encoding="utf-8") as f:
        return {str(q): refs for q, refs in json.load(f)["generations"].items()}


def load_count_control_generations(dataset):
    """{qid: [up to 40 strict passages]} used in the passage-count control."""
    path = os.path.join(DATA, "generations", "count_control", f"{dataset}_strict_40.json")
    with open(path, encoding="utf-8") as f:
        return {str(q): refs for q, refs in json.load(f)["generations"].items()}


def read_run(path):
    """{qid: [(docid, score), ...]} sorted by score (descending), then by rank in the file."""
    rows = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 6:
                rows[p[0]].append((p[2], float(p[4]), int(p[3])))
    return {q: [(d, s) for d, s, _ in sorted(v, key=lambda x: (-x[1], x[2]))]
            for q, v in rows.items()}


def write_run(path, ranking, tag):
    """ranking: {qid: [(docid, score), ...]} in rank order."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as o:
        for qid, docs in ranking.items():
            for rank, (docid, score) in enumerate(docs, start=1):
                o.write(f"{qid} Q0 {docid} {rank} {score} {tag}\n")


# ---------------------------------------------------------------- evaluation
def _trec_eval_command():
    native = shutil.which("trec_eval")
    if native:
        return [native]
    # Fall back to the trec_eval bundled with Pyserini (needs Java).
    return [sys.executable, "-m", "pyserini.eval.trec_eval"]


def qrels_file(dataset):
    from pyserini.search import get_qrels_file
    return get_qrels_file(PYSERINI_QRELS[dataset])


def per_query_metrics(run_path, dataset, qids, metrics=("ndcg_cut.10",)):
    """{metric: {qid: value}} with trec_eval -q, on the qrels restricted to `qids`.

    Queries of `qids` that are missing from the run get 0.
    """
    qids = [str(q) for q in qids]
    keep = set(qids)
    fd, q_tmp = tempfile.mkstemp(suffix="_qrels.txt")
    with os.fdopen(fd, "w") as o, open(qrels_file(dataset)) as f:
        for line in f:
            p = line.split()
            if len(p) >= 4 and p[0] in keep:
                o.write(line)
    cmd = _trec_eval_command() + ["-q"]
    for m in metrics:
        cmd += ["-m", m]
    try:
        proc = subprocess.run(cmd + [q_tmp, run_path], capture_output=True, text=True)
    finally:
        os.remove(q_tmp)
    if proc.returncode != 0:
        raise RuntimeError(f"trec_eval failed: {proc.stderr.strip()}")
    res = defaultdict(dict)
    for line in proc.stdout.splitlines():
        p = line.split()
        if len(p) == 3 and p[1] != "all":
            res[p[0]][p[1]] = float(p[2])
    return {name: {q: vals.get(q, 0.0) for q in qids} for name, vals in res.items()}


def read_ranking(path):
    """{qid: [docid, ...]} in the rank order written in the run file."""
    rows = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 6:
                rows[p[0]].append((int(p[3]), p[2]))
    return {q: [d for _, d in sorted(v, key=lambda x: x[0])] for q, v in rows.items()}


def ndcg10_of_ranking(ranking, dataset, qids):
    """nDCG@10 of {qid: [docid, ...]} given as rank-ordered lists."""
    fd, run_tmp = tempfile.mkstemp(suffix="_ranked.run")
    with os.fdopen(fd, "w") as o:
        for qid, docs in ranking.items():
            for rank, doc in enumerate(docs, start=1):
                o.write(f"{qid} Q0 {doc} {rank} {1000.0 - rank} ranked\n")
    try:
        return per_query_metrics(run_tmp, dataset, qids)["ndcg_cut_10"]
    finally:
        os.remove(run_tmp)


def per_query_ndcg10(run_path, dataset, qids, ties="score"):
    """Per-query nDCG@10 of a run file.

    Documents with identical text get identical reranking scores, so a run can contain ties.
    ties="score": pass the run to trec_eval as it is (trec_eval orders ties by document id).
    ties="rank":  keep the rank order written in the run file.
    The paper evaluates the single-strategy and no-expansion runs with "score" and the runs
    built from several strategies (integration, count control, subsets) with "rank"; see
    scripts/README.md.
    """
    if ties == "rank":
        return ndcg10_of_ranking(read_ranking(run_path), dataset, qids)
    return per_query_metrics(run_path, dataset, qids)["ndcg_cut_10"]


# ---------------------------------------------------------------- statistics
def paired_perm_p(a, b, n=10000, seed=0):
    """Two-sided paired permutation (sign-flip) test on the mean difference.

    The p-value depends on the order of the per-query vectors in its last digits; the paper
    uses the row order of strategy_validation9_perquery.csv (see reference_queries).
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    obs = d.mean()
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n, len(d)))
    perm = (signs * d).mean(axis=1)
    return (np.sum(np.abs(perm) >= abs(obs)) + 1) / (n + 1)


def boot_ci(d, n=10000, seed=0):
    """95% percentile interval of the mean of d under paired bootstrap resampling."""
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def macro_boot_ci(deltas_by_dataset, n=10000, seed=0):
    """Bootstrap interval of the macro-average: queries are resampled within each dataset."""
    rng = np.random.default_rng(seed)
    acc = np.zeros(n)
    for d in deltas_by_dataset:
        d = np.asarray(d, dtype=float)
        idx = rng.integers(0, len(d), size=(n, len(d)))
        acc += d[idx].mean(axis=1)
    acc /= len(deltas_by_dataset)
    return np.percentile(acc, 2.5), np.percentile(acc, 97.5)


def signflip_p(values):
    """Exact two-sided sign-flip test on the mean of a few values (e.g., five datasets)."""
    values = np.asarray(values, dtype=float)
    obs = abs(values.mean())
    hits = total = 0
    for signs in itertools.product([-1, 1], repeat=len(values)):
        total += 1
        if abs((values * np.array(signs)).mean()) >= obs - 1e-12:
            hits += 1
    return hits / total


def holm(pvalues):
    """Holm-corrected p-values, in the input order."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p, kind="stable")
    out = np.empty(len(p))
    running = 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(p) - i) * p[k]))
        out[k] = running
    return out


def mcnemar_exact(b, c):
    """Two-sided exact McNemar test from the two discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)

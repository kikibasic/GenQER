# Scripts

Everything here runs without calling an LLM. Run the scripts from the repository root.

| Script | What it does | Needs |
|---|---|---|
| `reproduce_tables.py` | Recomputes the differences, confidence intervals, permutation tests, Holm corrections and degradation rates of the paper from `results/per_query/` and compares them with `results/tables/`. | numpy, pandas, scipy |
| `per_query_selectors.py predictive` | Trains and evaluates the predictive selector (logistic regression, leave-one-dataset-out). | + scikit-learn |
| `evaluate_runs.py` | Per-query nDCG@10 (or other trec_eval metrics) of a run file; `--check` re-evaluates all released runs against `results/per_query/`. | + Pyserini, trec_eval, `data/runs/` |
| `fuse_runs.py` | RRF and CombSUM over the nine runs (no expansion + eight strategies). | same |
| `per_query_selectors.py observational` | The observational selector (top-2 options by reranking score, fused with RRF). | same |
| `run_pipeline.py` | Rebuilds a run file from the saved generations: BM25 with the expanded query, then reranking. Covers single strategies, integration of any strategy set, the stage-wise conditions, the passage-count control, and the second reranker. | + torch, transformers, `data/generations/` |
| `common.py` | Dataset tables, file readers, trec_eval wrapper, statistical tests. | |

## Setup

```bash
pip install -r requirements.txt
```

Pyserini needs Java 21. `evaluate_runs.py` uses a `trec_eval` binary if one is on the PATH and
otherwise the copy bundled with Pyserini. The run files and generations come from the
[v1.0 release](https://github.com/kikibasic/GenQER/releases/tag/v1.0); unzip them into `data/`
(see `data/README.md`). The Pyserini indexes, topics and relevance judgments are downloaded
on first use.

## Three levels of reproduction

1. **From the per-query files (seconds).**

   ```bash
   python scripts/reproduce_tables.py
   python scripts/per_query_selectors.py predictive
   ```

2. **From the run files (under a minute).** Re-evaluates the 230 released runs and recomputes
   the fused and selected rankings.

   ```bash
   python scripts/evaluate_runs.py --check
   python scripts/fuse_runs.py
   python scripts/per_query_selectors.py observational
   ```

3. **From the generations (about a minute per 50 queries on a laptop GPU).** Reruns retrieval
   and reranking.

   ```bash
   python scripts/run_pipeline.py --dataset dl19 --passages all8 --out dl19_all8.run
   python scripts/evaluate_runs.py dl19_all8.run --dataset dl19 --ties rank
   ```

   The passage specifications are listed in the header of `run_pipeline.py`. The released
   count-control runs correspond to `strict:10`, `strict:20`, `strict:40`, `pool:5:0`,
   `pool:5:1`, `pool:5:2`, `pool:10:0`, `pool:20:0`, `mix:2:5:0` and `mix:4:5:0`.

## Order of the per-query vectors

The permutation tests use 10,000 random sign flips with a fixed seed, so the last digits of a
p-value depend on the order of the per-query differences. The scripts keep the order used for
the paper (the row order of `strategy_validation9_perquery.csv`, or the sorted order of
dataset and query id where the original analysis sorted).

## Tied scores

Documents with identical text receive identical reranking scores, and every collection
contains such duplicates, so the run files contain ties. Two orderings of tied documents
were used:

- the no-expansion and single-strategy runs were passed to trec_eval as they are, and
  trec_eval orders tied documents by document id (`--ties score`);
- the runs that combine several strategies (all-strategy integration, passage-count control,
  strategy subsets, the second reranker), all fused rankings, and the three held-out runs in
  `heldout_confirmation_perquery.csv` were evaluated in the rank order written in the run
  file (`--ties rank`).

The two orderings differ on about ten of the 755 queries. `evaluate_runs.py --check` applies
to each run the ordering that produced the stored value. With one ordering on both sides, the
pooled difference between all-strategy integration and strict is +0.0180 (trec_eval order) or
+0.0181 (file order) instead of the reported +0.0182, and the permutation test gives
p = 0.0002 in both cases. The stage-wise analysis (Fig. 1, `stage_ablation_summary.csv`) uses
the trec_eval order for all four conditions, which is why all-strategy integration appears
there as 0.6171 rather than 0.6174.

## Not included

- Calling the LLM. The prompts are in `prompts/` and the settings in
  `config/generation.yaml`; the saved generations replace the calls.
- The LLM-judgment selector, which the paper reports as exploratory only.
- Per-dataset selection with development queries (five-fold and the learning curve): the
  per-query values of every candidate configuration are in
  `results/per_query/subset_pool_perquery.csv` and the outcomes in
  `results/tables/review_per_dataset_learning_curve*.csv`.

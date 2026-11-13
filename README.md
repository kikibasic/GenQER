# GenQER: Strategy Selection versus Integration for Generative Query Expansion and Reranking

Materials for the paper

> Aira Hayashi, Miki Enoki, Masaaki Tsuchida, Yukio Uematsu, and Masato Oguchi.
> "GenQER: Strategy Selection versus Integration for Generative Query Expansion and Reranking."
> *Proceedings of the 21st International Conference on Ubiquitous Information Management and Communication (IMCOM 2027)*, Taipei, January 2027. (DOI to be added.)

The paper asks whether, for query expansion with LLM-generated pseudo-reference passages, it is better to pick one generation strategy per query or to integrate passages from several strategies. Eight generation strategies are compared on five IR datasets (755 queries), with two further held-out datasets.

## What is here

| Folder | Contents | Status |
|---|---|---|
| `prompts/` | The full text of every prompt used by the eight generation strategies, exported verbatim from the experiment code. | complete |
| `config/generation.yaml` | Generation conditions: model, temperature, number of passages, dates, API settings. | complete |
| `results/tables/` | The aggregated numbers behind every table and figure in the paper (CSV). | complete |
| `results/per_query/` | Per-query nDCG@10 for every compared method, from which all tests in the paper can be recomputed (CSV). | complete |
| `data/` | The saved generations (passages) and TREC run files. | distributed as two ZIP archives attached to the [v1.0 release](https://github.com/kikibasic/GenQER/releases/tag/v1.0); see `data/README.md` |
| `scripts/` | Retrieval and reranking from the saved generations, evaluation of run files, fusion, selectors, and the statistical tests. | complete; see `scripts/README.md` |

## Mapping to the paper

| Paper | File(s) |
|---|---|
| Table I (strategies) | `prompts/` |
| Table II (per-query selection) | `results/tables/router_selection_accuracy9_summary.csv`, `postretrieval_selection_summary.csv`, `review_selector_diagnostics.csv` |
| Table III (per-dataset differences, RRF, CombSUM) | `results/tables/review_per_dataset.csv`, `fusion_all9_summary.csv`, `fusion_pool_summary.csv` |
| Direct selection-vs-integration test | `results/tables/review_selection_vs_integration.csv` |
| Table IV (passage-count control) | `results/tables/count_control_summary.csv`, `count_control_samecount.csv` |
| Table V (cost) | `results/tables/review_cost_methods.csv`, `review_cost.csv` |
| Fig. 1 (stage-wise analysis) | `results/tables/stage_ablation_summary.csv` |
| Fig. 2 (per-dataset strategy characteristics) | `results/tables/corpus_strategy_oracle_distribution.csv`, `review_median_comparison.csv` |
| Per-dataset selection and learning curve | `results/tables/review_per_dataset_learning_curve.csv` |
| Per-dataset selection with five-fold cross-validation | `results/tables/subset_pool_summary.csv`, `subset_pool_choices.csv` |
| Held-out datasets (TREC-NEWS, Robust04) | `results/tables/heldout_confirmation_summary.csv` |
| Generation variability (three trials on DL19/DL20) | `results/tables/oracle_noise_summary.csv` |
| LLM-judgment selector (exploratory) | `results/tables/llm_router_v4_9_summary.csv` |
| Second reranker (bge-base) | `results/tables/reranker_robustness_summary.csv` |
| Holm-corrected p-values | `results/tables/holm_corrected_pooled.csv` |

## Experimental conditions in brief

- Generator: OpenAI `gpt-4o` (snapshot `gpt-4o-2024-08-06`, the alias's default snapshot per OpenAI's model documentation; the code requested the alias), temperature 0.7, other sampling parameters at API defaults, five passages per strategy and query. Main generations: June 26 to July 26, 2026. Additional strict passages for the passage-count control (up to 40 per query): August 30 and September 10, 2026. See `config/generation.yaml`.
- First stage: Pyserini prebuilt BM25 indexes, k1 = 0.9, b = 0.4, top 100.
- Second stage: `all-mpnet-base-v2`; the query representation is the re-normalized mean of the L2-normalized embeddings of "original query + each passage".
- Datasets: TREC-COVID (50 queries), NFCorpus (308), SciFact (300), TREC DL19 (43), DL20 (54); held-out: TREC-NEWS (57), Robust04 (249).
- NFCorpus: BEIR provides 323 test queries. For 15 single-word queries (e.g., *okra*, *Yale*, *Zoloft*), BM25 with the original query returns no document, so the no-expansion baseline is undefined; all conditions are evaluated on the 308 queries for which every condition produces a ranking.
- Statistical tests: paired two-sided permutation test with 10,000 resamples (the exact p-value depends on the order of the per-query vector; the per-query files are stored in the order used for the paper), McNemar test for degradation rates, Holm correction within the stated families, paired bootstrap (10,000 resamples) for confidence intervals.

## Reproducing the numbers

No step below calls the LLM. Details and requirements are in `scripts/README.md`.

1. From the per-query files in `results/per_query/` (numpy, pandas, scipy only):

   ```bash
   python scripts/reproduce_tables.py
   ```

   recomputes the differences, confidence intervals, permutation tests, Holm corrections and degradation rates and compares them with `results/tables/`.

2. From the run files (unzip the release archives into `data/` first):

   ```bash
   python scripts/evaluate_runs.py --check
   ```

   re-evaluates every released run with trec_eval and compares the per-query nDCG@10 with `results/per_query/`. `scripts/fuse_runs.py` and `scripts/per_query_selectors.py` recompute the fused rankings and the selectors.

3. From the saved generations:

   ```bash
   python scripts/run_pipeline.py --dataset dl19 --passages all8 --out dl19_all8.run
   ```

   reruns BM25 and reranking for a strategy, a set of strategies, or a passage-count condition.

Regenerating the passages with the API is possible with the prompts in `prompts/` and the settings in `config/generation.yaml`; note that the gpt-4o snapshots are scheduled for shutdown by OpenAI on January 14, 2027, and that sampling at temperature 0.7 is not deterministic. Use the saved generations in `data/` for exact reproduction.

## License

Code: MIT (see `LICENSE`). Prompts, result tables, and released generations: CC BY 4.0. The datasets themselves are distributed by their original providers (BEIR, TREC) under their own terms.

## Questions

Please open a GitHub issue.

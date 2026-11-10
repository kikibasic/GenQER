# Results

All numbers are nDCG@10 unless a column says otherwise. Dataset codes: `covid` = TREC-COVID, `nfc` = NFCorpus, `scifact` = SciFact, `dl19`/`dl20` = TREC DL 2019/2020, `news` = TREC-NEWS, `robust04` = Robust04. `POOLED` = all 755 queries of the five main datasets as one set.

Strategy codes used in column names: `n0` = no expansion; `legacy`, `strict`, `aspect`, `hybrid`, `broaden`, `grounded`, `corpus_aspect`, `corpus_broaden` = the eight strategies (`grounded` in the files is the strategy called corpus_strict in the paper). In `subset_pool_perquery.csv` the short codes are `leg`, `str`, `asp`, `hyb`, `bro`, `gro` (= corpus_strict), `casp`, `cbro`; `drop_X` = all strategies except X; `all6` = all except corpus_aspect and corpus_broaden; `all8` = all-strategy integration.

## per_query/

| File | Content |
|---|---|
| `strategy_validation9_perquery.csv` | nDCG@10 of no expansion and the eight single strategies per query, plus the six selector features. The row order of this file is the order used for the permutation tests in the paper. |
| `fusion_pool_perquery.csv` | passage-level integration variants (`ctxpool_all8` = all-strategy integration, `ctxpool_all6` = six-strategy integration, oracle/selector top-2 variants) |
| `fusion_all9_perquery.csv` | retrieval-result fusion of the nine runs with RRF (`rrf_all9`) and CombSUM (`combsum_all9`), and of the seven LLM-only runs (`*_all7`) |
| `count_control_perquery.csv` | passage-count control conditions (`single_strict_10/20/40`, `pool_5_{seed}`, `pool_10_0`, `pool_20_0`, `mix_2_5_0`, `mix_4_5_0`) |
| `router_selection_accuracy9_perquery.csv` | predictive selector (logistic regression, leave-one-dataset-out): selected option and its nDCG@10 |
| `review_postretrieval_perquery.csv` | observational selectors (`obs_top1_*`, `obs_top2_rrf_*`; the paper reports `obs_top2_rrf_qpp_max`) |
| `corpus_strategy_perquery.csv` | the eight strategies and all-strategy integration on all seven datasets |
| `heldout_confirmation_perquery.csv` | TREC-NEWS and Robust04: no expansion, strict, all-strategy integration |
| `subset_pool_perquery.csv` | the 18 candidates (and greedy pairs/triples) of per-dataset strategy-set selection |

## tables/

| File | Used for |
|---|---|
| `review_per_dataset.csv` | Table III: per-dataset differences with bootstrap CIs and permutation p-values; pooled and macro-average rows |
| `review_selection_vs_integration.csv` | direct tests of all-strategy integration against the observational and predictive selectors |
| `review_cost.csv`, `review_cost_methods.csv` | Table V: calls, tokens, searches, embeddings per query |
| `count_control_summary.csv`, `count_control_samecount.csv` | Table IV and the same-count tests (with Holm correction) |
| `review_median_comparison.csv`, `review_median_comparison_corr.csv` | integration versus the median/mean/best single strategy per dataset, and the correlations with the spread of strategy means |
| `review_per_dataset_learning_curve.csv`, `review_per_dataset_learning_curve_choices.csv` | per-dataset selection with 10/20/40/80 % development queries |
| `review_selector_diagnostics.csv`, `review_selector_prediction_dist.csv` | regret, prediction distribution, and the selector with WIG/NQC added |
| `router_selection_accuracy9_summary.csv`, `postretrieval_selection_summary.csv` | Table II |
| `fusion_pool_summary.csv`, `fusion_all9_summary.csv` | integration and fusion summaries |
| `stage_ablation_summary.csv` | Fig. 1 |
| `corpus_strategy_variation_summary.csv`, `corpus_strategy_oracle_distribution.csv` | Fig. 2 |
| `heldout_confirmation_summary.csv` | held-out datasets |
| `reranker_robustness_summary.csv` | bge-base-en-v1.5 check |
| `holm_corrected_pooled.csv` | Holm-corrected p-values of the main family |

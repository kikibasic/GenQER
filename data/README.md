# Data

The saved generations and run files are distributed as ZIP archives attached to a GitHub Release of this repository (they are too large to keep in the repository itself).

Planned contents:

- `generations/main/`: for each of the five main datasets and eight strategies, one JSON file with the five passages per query (and, for the aspect-type strategies, the Stage 1 decomposition), about 35 MB in total.
- `generations/count_control/`: the additional strict passages (up to 40 per query) used in the passage-count control.
- `runs/`: TREC-format run files (top 100 after reranking) for no expansion, each single strategy, and all-strategy integration.

Until the Release is published, the per-query nDCG@10 of every method is available in `../results/per_query/`.

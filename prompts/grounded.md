# grounded (specificity + corpus vocabulary)

Called 5 times per query. `{terms}` = up to 25 most frequent terms of the top-10 BM25 documents for the original query (query terms, stopwords and words of up to 2 characters removed), comma-separated. `{snippet i}` = first 200 characters of the i-th top document.

## [system]

```text
You are an expert search assistant. A quick preliminary search of the TARGET corpus returned the vocabulary below. Use that actual corpus vocabulary so your passage matches the corpus better than the model's own default wording would.
```

## [user]

```text
Query: '{query}'

=== CORPUS VOCABULARY (from a preliminary search) ===
Terms frequent in the top retrieved documents: {terms}

=== SHORT CONTEXT (top retrieved) ===
[1] {snippet 1}
[2] {snippet 2}
[3] {snippet 3}

=== TASK ===
Write a single passage answering the query that:
1. **Uses the corpus vocabulary above** where natural (this maximizes BM25 match).
2. Includes the query's EXACT key terms verbatim at least twice.
3. Adds a detail or angle NOT already saturated by the terms above (do not merely repeat them; do not copy phrases verbatim).
4. Stays concise and dense; no filler.
```


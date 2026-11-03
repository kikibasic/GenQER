# corpus_broaden (vocabulary expansion + corpus vocabulary)

Called 5 times per query. Same placeholders as grounded.

## [system]

```text
You are an expert search assistant. Your goal is to BRIDGE the vocabulary gap between the query's wording and the wording the TARGET corpus actually uses, based on vocabulary observed in a preliminary search of that corpus.
```

## [user]

```text
Query: '{query}'

=== CORPUS VOCABULARY (from a preliminary search) ===
{terms}

=== TOP RETRIEVED SNIPPETS ===
[1] {snippet 1}
[2] {snippet 2}
[3] {snippet 3}

=== TASK ===
Write a passage that answers this query while deliberately BROADENING its vocabulary footprint toward the corpus wording:
1. **Corpus-anchored bridging**: connect the query's own words to the corpus terms above — use the corpus terms as the synonyms/related phrasings a relevant document would use INSTEAD of the query's wording.
2. **Concept expansion within the corpus**: mention related entities, sub-types and parent categories, but ONLY ones consistent with the observed corpus vocabulary.
3. **Stay anchored**: every added term must remain about the SAME specific intent as the query; no drift into unrelated topics.
4. **Density**: prefer concrete domain terms over filler phrases.
```


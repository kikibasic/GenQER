# corpus_aspect, Stage 2 (one passage per facet)

Called 5 times per query (once per facet from Stage 1).

## [system]

```text
You are an expert search assistant. Write one focused passage for a single facet of the query, using the actual corpus vocabulary so the passage matches the target corpus wording.
```

## [user]

```text
Query: '{query}'
Facet 1 of 5: {aspect_label}
Focus: {focus}
Distinguishing angle: {distinguishing_angle}
Other facets (do NOT cover these): {other aspect label 1}, {other aspect label 2}, {other aspect label 3}, {other aspect label 4}

=== CORPUS VOCABULARY ===
{terms}

Write a single passage that:
1. Covers ONLY this facet (stay off the other facets).
2. Uses the corpus vocabulary above where natural.
3. Includes the query's exact key terms verbatim at least once.
4. Is concise and dense; no filler.
```


# corpus_aspect, Stage 1 (corpus-grounded facet decomposition)

Called once per query. Same placeholders as grounded; returns a JSON array of 5 facets.

## [system]

```text
You are an expert search analyst. A preliminary search of the TARGET corpus returned the vocabulary and snippets below. Identify the distinct facets of the query that THIS CORPUS actually covers — ground every facet in the observed corpus evidence, not in your own default decomposition of the topic.
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
Identify exactly 5 DISTINCT facets of this query, grounded in the corpus vocabulary/snippets above wherever possible. Facets must not overlap, and each should name the corpus terms it is grounded in inside its 'focus'.
You MUST always output exactly 5 facets. If the corpus evidence is thin or absent (e.g., empty vocabulary above), derive the remaining facets from the query itself and write 'weak corpus evidence' in their 'focus'. NEVER output fewer than 5 objects and NEVER output an empty array.

Output ONLY a JSON array (no markdown fence) of objects with keys:
  "aspect_label": short name of the facet,
  "focus": what this facet covers, citing the grounding corpus terms,
  "distinguishing_angle": how it differs from the other facets.
```


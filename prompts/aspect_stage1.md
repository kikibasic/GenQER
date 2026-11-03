# aspect, Stage 1 (decompose the query into 5 aspects)

Called once per query. The model returns a JSON array of 5 objects with keys `aspect_label`, `focus`, `distinguishing_angle`. Parsing is retried up to 3 times.

## [system]

```text
You are an expert in information retrieval and query analysis. Your task is to decompose a user query into N complementary aspects that together provide non-overlapping coverage of what the query asks for. The aspects will be used to generate N distinct pseudo-reference documents for query expansion.
```

## [user]

```text
Query: '{query}'

Decompose this query into exactly 5 aspects. The 5 aspects together must satisfy:

1. **Anchored**: every aspect addresses the SPECIFIC query above. Do NOT generalize the topic beyond the query's scope.
2. **Non-overlapping**: each aspect covers content that the other 4 aspects do NOT cover.
3. **Constraint-preserving**: if the query contains modifiers (e.g., 'right' vs 'left', 'in main Paris', specific numbers, dates), numerical conditions, location/direction qualifiers, abbreviations, or comparison words ('vs', 'difference between'), EVERY aspect must respect them.
4. **Independently answerable**: each aspect can be answered by a self-contained 150-200 word passage.
5. **Semantically distinct**: aspects must differ in WHAT they cover (e.g., definition vs mechanism vs causal factor vs example vs counter-case), not just rephrase the same content under different headings.

Output a JSON array of exactly 5 objects with this exact schema:

[
  {
    "aspect_label": "<short noun phrase, 3-8 words, identifying the sub-topic>",
    "focus": "<one sentence describing what this aspect specifically addresses>",
    "distinguishing_angle": "<one short phrase explaining how this aspect differs from the other 4>"
  },
  ...
]

Output the JSON array only. No preamble, no explanation, no markdown code fences.
```


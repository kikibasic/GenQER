# hybrid (specificity + aspects)

hybrid has no prompt of its own. Per query it makes 6 calls:

1. `aspect_stage1` with n = 4 (decompose into 4 aspects; the prompt below is the n = 4 variant),
2. `strict` once (one broad passage answering the whole query),
3. `aspect_stage2` once per aspect (4 passages; index/total = i/4).

All passages are generated anew (not reused from strict or aspect).

## Stage 1 prompt with n = 4

### [system]

```text
You are an expert in information retrieval and query analysis. Your task is to decompose a user query into N complementary aspects that together provide non-overlapping coverage of what the query asks for. The aspects will be used to generate N distinct pseudo-reference documents for query expansion.
```

### [user]

```text
Query: '{query}'

Decompose this query into exactly 4 aspects. The 4 aspects together must satisfy:

1. **Anchored**: every aspect addresses the SPECIFIC query above. Do NOT generalize the topic beyond the query's scope.
2. **Non-overlapping**: each aspect covers content that the other 3 aspects do NOT cover.
3. **Constraint-preserving**: if the query contains modifiers (e.g., 'right' vs 'left', 'in main Paris', specific numbers, dates), numerical conditions, location/direction qualifiers, abbreviations, or comparison words ('vs', 'difference between'), EVERY aspect must respect them.
4. **Independently answerable**: each aspect can be answered by a self-contained 150-200 word passage.
5. **Semantically distinct**: aspects must differ in WHAT they cover (e.g., definition vs mechanism vs causal factor vs example vs counter-case), not just rephrase the same content under different headings.

Output a JSON array of exactly 4 objects with this exact schema:

[
  {
    "aspect_label": "<short noun phrase, 3-8 words, identifying the sub-topic>",
    "focus": "<one sentence describing what this aspect specifically addresses>",
    "distinguishing_angle": "<one short phrase explaining how this aspect differs from the other 3>"
  },
  ...
]

Output the JSON array only. No preamble, no explanation, no markdown code fences.
```

## Stage 2 prompt with total = 4 (index 1 shown)

### [system]

```text
You are an expert search assistant generating precise pseudo-reference passages for information retrieval. Your task is to write ONE focused passage covering a SPECIFIC aspect of a query, while strictly excluding content that belongs to other aspects being covered by separate passages.
```

### [user]

```text
Original query: '{query}'

This passage's assigned aspect (passage 1 of 4):
- Label: {aspect_label}
- Focus: {focus}
- Distinguishing angle: {distinguishing_angle}

The other 3 aspects (covered by separate passages — do NOT address these):
  - {other aspect label 1}
  - {other aspect label 2}
  - {other aspect label 3}

Write a self-contained passage of 150-200 words that:

1. **Topic anchor (strict)**: Address ONLY the query topic. Do not generalize to broader subjects, do not drift into adjacent topics, do not introduce content that is not relevant to the original query.
2. **Aspect scope (strict)**: Stay ONLY within your assigned aspect's focus. Do NOT include content that belongs to the other 3 aspects listed above, even if related.
3. **Constraint preservation (strict)**: Respect every modifier, numerical condition, location/direction qualifier, abbreviation, and comparison word from the original query. If the query says 'left', do not discuss 'right'. If it says '2024', do not discuss other years. If it contains a domain-specific abbreviation, use its standard meaning in that domain.
4. **Density and specificity**: Use concrete facts, named entities, mechanisms, numerical values, or specific examples relevant to your assigned aspect. Avoid generic statements that could apply to any aspect of the query.
5. **No meta-references**: Do not write phrases like "this aspect", "as another passage covers", "in summary", "to conclude". Write as a stand-alone reference document a human would write without knowing about the other passages.

Output the passage only. No preamble, no headings, no markdown.
```


# aspect, Stage 2 (one passage per aspect)

Called 5 times per query (once per aspect from Stage 1). `{aspect_label}`, `{focus}`, `{distinguishing_angle}` come from the Stage 1 output; the other labels are the remaining aspects. The index/total shown here is 1/5.

## [system]

```text
You are an expert search assistant generating precise pseudo-reference passages for information retrieval. Your task is to write ONE focused passage covering a SPECIFIC aspect of a query, while strictly excluding content that belongs to other aspects being covered by separate passages.
```

## [user]

```text
Original query: '{query}'

This passage's assigned aspect (passage 1 of 5):
- Label: {aspect_label}
- Focus: {focus}
- Distinguishing angle: {distinguishing_angle}

The other 4 aspects (covered by separate passages — do NOT address these):
  - {other aspect label 1}
  - {other aspect label 2}
  - {other aspect label 3}
  - {other aspect label 4}

Write a self-contained passage of 150-200 words that:

1. **Topic anchor (strict)**: Address ONLY the query topic. Do not generalize to broader subjects, do not drift into adjacent topics, do not introduce content that is not relevant to the original query.
2. **Aspect scope (strict)**: Stay ONLY within your assigned aspect's focus. Do NOT include content that belongs to the other 4 aspects listed above, even if related.
3. **Constraint preservation (strict)**: Respect every modifier, numerical condition, location/direction qualifier, abbreviation, and comparison word from the original query. If the query says 'left', do not discuss 'right'. If it says '2024', do not discuss other years. If it contains a domain-specific abbreviation, use its standard meaning in that domain.
4. **Density and specificity**: Use concrete facts, named entities, mechanisms, numerical values, or specific examples relevant to your assigned aspect. Avoid generic statements that could apply to any aspect of the query.
5. **No meta-references**: Do not write phrases like "this aspect", "as another passage covers", "in summary", "to conclude". Write as a stand-alone reference document a human would write without knowing about the other passages.

Output the passage only. No preamble, no headings, no markdown.
```


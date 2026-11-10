# Prompts

Each file holds the exact system and user messages sent to `gpt-4o` for one generation step, exported verbatim from the experiment code. Parts in curly braces are filled in at run time:

| Placeholder | Filled with |
|---|---|
| `{query}` | the original query text |
| `{terms}` | up to 25 corpus terms from the probe retrieval, comma-separated (corpus_strict, corpus_aspect, corpus_broaden) |
| `{snippet 1}` .. `{snippet 3}` | the first 200 characters of the top three documents of the probe retrieval |
| `{aspect_label}`, `{focus}`, `{distinguishing_angle}` | one aspect returned by the Stage 1 decomposition (JSON) |
| `{other aspect label i}` | the labels of the remaining aspects |

| Strategy | Files | Calls per query |
|---|---|---|
| legacy | `legacy.md` | 5 |
| strict | `strict.md` | 5 |
| aspect | `aspect_stage1.md`, `aspect_stage2.md` | 1 + 5 |
| hybrid | `hybrid.md` (Stage 1 with 4 aspects, then strict once and Stage 2 four times) | 1 + 1 + 4 |
| broaden | `broaden.md` | 5 |
| corpus_strict | `corpus_strict.md` | 5 (+ 1 probe retrieval) |
| corpus_aspect | `corpus_aspect_stage1.md`, `corpus_aspect_stage2.md` | 1 + 5 (+ 1 probe retrieval) |
| corpus_broaden | `corpus_broaden.md` | 5 (+ 1 probe retrieval) |

The strategy corpus_strict is called `corpus_grounded` in the experiment code and in the column names of the result files (and `grounded` in earlier drafts of the paper).

Every call requests one passage; the Stage 1 calls request a JSON array of aspects instead and are retried up to three times if the JSON cannot be parsed.

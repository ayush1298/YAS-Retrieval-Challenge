# YAS Retrieval Challenge: Solution and Evaluation

This is my solution to the YAS retrieval challenge; the brief is in [README.md](README.md). It is a layout-aware hybrid retrieval system over nine Arabic pages from two December 1954 issues of *Al-Kuwait Al-Yawm*, Kuwait's official newspaper.

For an Arabic query it returns one of:
- ranked passage IDs;
- page IDs, for "everything about …" and "everything in issue …" queries;
- an abstention, when the pages don't answer the query.

Everything runs locally: two open models ([BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3), [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)) plus rule-based Arabic text and layout processing. The code is in `solution/`.

## Run

**Requirements.**

- [uv](https://docs.astral.sh/uv/) is the only thing to install. It sets up the pinned Python (3.12) and all dependencies from `uv.lock`.
- The first run downloads the two models (~4.5 GB, pinned to fixed revisions) into the Hugging Face cache; later runs work offline.
- On Linux, PyTorch from PyPI also pulls several GB of CUDA libraries on the first `uv run`. It is slow, but automatic.
- It runs on CPU, Apple MPS or CUDA, and CPU and GPU give identical tables.
- The optional verifier (below) needs [Ollama](https://ollama.com).

```bash
uv run solution evaluate                  # every table below: 4 configurations × 3 query sets (~1 min on an M3 Max GPU, ~2.5 min on CPU)
uv run solution evaluate --details        # plus the ranking of each query
uv run solution run held_out.jsonl        # one JSON line per query
uv run solution search "قانون البلدية المادة ٢٥"
uv run pytest                             # layout, query-parsing, search-intent and metric tests
```

Each result has:
- `mode`: `passages`, `pages` (exhaustive and issue-date queries) or `abstain` (empty lists);
- the ranked `passages` and `pages`;
- `evidence`: the best answer evidence found, from 0 to 1. It is `null` when the decision came from structure alone: an issue-date filter, or a cited article or issue that does not exist.

**Optional answerability verifier.** Pull the model once with `ollama pull qwen3.5:4b`, keep Ollama running, and put `--verifier qwen3.5:4b` before the command (e.g. `uv run solution --verifier qwen3.5:4b run held_out.jsonl`). When the reranker is unsure, the local LLM decides whether the top passages really answer the query. It leaves the provided and dev results unchanged and fixes most wrong answers to unanswerable questions; see "Choosing the answerability check".

## What I built

1. **Arabic matching** (`solution/text.py`): normalisation (alef/hamza, ya, ta marbuta, Persian `ک ی`, Arabic-Indic digits, `عبد ال…`), Light10 stemming, and character 3–5-grams so `كارسياكا` still matches `کارسیا کا`.
2. **Layout recovery** (`solution/corpus.py`). These are rules over the transcripts, not per-page annotation:
   - short unpunctuated title lines open sections, and notice bullets and tables start new ones;
   - law articles are `مادة N` lines and top-level `N-` regulations, and a passage belongs to every article it spans;
   - a paragraph ending in `:` is the lead-in of the marker-prefixed items that follow it;
   - table cells are labelled with their column header and caption. When every cell splits into the same number of values, rows are rebuilt by position (G2-p13 → 12 rows such as `اليوم: الاثنين | التاريخ: 3/1/1955 | الارقام الفائزة في القرعة: 41 - 50`).

   Each passage is indexed with its heading, article or list lead-in, or table header, plus one extra unit per rebuilt row.
3. **Retrieval** (`solution/search.py`):
   - BM25 on stems, BM25 on character n-grams and BGE-M3 dense scores are min-max fused.
   - The top 30 passages are rescored by bge-reranker-v2-m3, blended 50/50 with the fused score.
   - That score is blended 50/50 with a section-level score, so whole sections and articles rank together.
4. **Query intent**, read from the query text alone:
   - A cited article (`المادة ٢٥`, `المادة الثانية`) is returned first, as a block.
   - A publication cue with a date (`كل ما نشر في عدد ٢٥ ديسمبر ١٩٥٤`) filters pages by issue date.
   - `كل ما …` / `جميع ما …` returns pages.
   - The system abstains when no passage carries answer evidence, or when the cited article or issue does not exist.
5. **Optional verifier** (`--verifier`): a cascade. Reranker evidence of 0.5 or more is accepted as is. Below that, a local LLM reads each of the top three passages, with their context and rebuilt rows. The query is answered only if the LLM says one of them contains the answer.

## Results at a glance

R@5 / R@10 / MRR over all queries. Per-tier and per-type breakdowns follow.

| Configuration | Provided (12) | Dev (34) | Stress (26) |
|---|---|---|---|
| BM25 (normalised, stemmed) | 0.59 / 0.69 / 0.67 | 0.75 / 0.77 / 0.72 | 0.33 / 0.37 / 0.31 |
| Hybrid: + char n-grams + BGE-M3 | 0.68 / 0.77 / 0.72 | 0.75 / 0.78 / 0.72 | 0.37 / 0.38 / 0.32 |
| + layout recovery and query intent | 0.85 / 0.89 / 0.88 | 0.91 / 0.93 / 0.89 | 0.44 / 0.44 / 0.42 |
| + reranking and abstention (default) | **0.93 / 0.99 / 1.00** | **0.97 / 0.99 / 0.98** | 0.73 / 0.73 / 0.71 |
| + qwen3.5:4b verifier (`--verifier`) | 0.93 / 0.99 / 1.00 | 0.97 / 0.99 / 0.98 | **0.88 / 0.88 / 0.86** |

On the provided set the default system reaches the metric ceiling. q02 and q06 have 7 and 11 relevant passages, so R@5 cannot exceed 0.93 and R@10 cannot exceed 0.99. The stress set is adversarial by construction, so its numbers are not an estimate of held-out performance (see Setup).

## Setup

**Query sets.**

- `data/queries.jsonl`: the 12 provided queries.
- `eval/dev_queries.jsonl`: 34 queries I wrote and labelled from the transcripts, spread over all tiers and types.
- `eval/hard_queries.jsonl`: the stress set, 26 queries for answerability.
  - 16 are unanswerable: absent names, facts the pages don't give, false premises, a non-existent issue.
  - 10 are answerable but easy to wrongly abstain on: counting, vague, keyword-only or paraphrased questions.

I wrote the dev queries before building the retrieval stack and then used them during development, so they are not a blind test. I wrote the stress set later, after probing the system for weaknesses. It is deliberately hard for the system and measures abstention, not typical performance.

**Configurations.**

| Name | What it adds |
|---|---|
| `bm25` | BM25 over normalised, Light10-stemmed words of the raw passage text |
| `hybrid` | BM25 over character 3–5-grams and BGE-M3 cosine similarity, min-max fused with the above |
| `hybrid+layout` | Passages indexed with their recovered context and rebuilt table rows; section-level score blended in; query intent (article citations, issue-date filter, page-level answers, abstaining on a missing article or issue) |
| `hybrid+layout+rerank` | bge-reranker-v2-m3 over the top 30 passages, and evidence-based abstention. This is the default. |
| `…+verifier` | qwen3.5:4b through Ollama decides answerability when reranker evidence is below 0.5. Opt-in with `--verifier`. |

**Metrics.**

- **Recall@k:** |relevant ∩ top k| / |relevant|.
- **MRR:** 1 / rank of the first relevant result, or 0 if none is returned.
- **What gets ranked:** passage queries are scored on the passage list. Page queries (`relevant_pages`) are scored on the returned pages; a configuration that answers them with passages instead (`bm25`, `hybrid`) is scored on the pages of its top 10 passages and returns no page set.
- **Abstain queries:** score 1 on every metric when the system abstains and 0 otherwise.
- **Answerable queries:** score 0 when the system abstains, so false abstentions count in every table below.
- **Ceilings:** some queries have more than 5 or 10 relevant passages, which caps their recall.
  - q02 has 7 relevant passages, so R@5 ≤ 0.71.
  - q06 has 11, so R@5 ≤ 0.45 and R@10 ≤ 0.91.
  - d10 has 16, so R@5 ≤ 0.31 and R@10 ≤ 0.63.

## Results: provided queries

Cells: R@5 / R@10 / MRR

| Tier | n | bm25 | hybrid | hybrid+layout | hybrid+layout+rerank |
|---|---|---|---|---|---|
| Bronze | 3 | 0.76 / 0.81 / 0.83 | 0.76 / 0.81 / 1.00 | 0.90 / 0.90 / 1.00 | 0.90 / 1.00 / 1.00 |
| Silver | 3 | 0.70 / 0.70 / 1.00 | 0.70 / 0.70 / 1.00 | 0.82 / 0.97 / 1.00 | 0.82 / 0.97 / 1.00 |
| Gold | 3 | 0.33 / 0.67 / 0.39 | 0.67 / 1.00 / 0.43 | 1.00 / 1.00 / 0.83 | 1.00 / 1.00 / 1.00 |
| Cross-tier | 3 | 0.58 / 0.58 / 0.44 | 0.58 / 0.58 / 0.44 | 0.67 / 0.67 / 0.67 | 1.00 / 1.00 / 1.00 |
| All | 12 | 0.59 / 0.69 / 0.67 | 0.68 / 0.77 / 0.72 | 0.85 / 0.89 / 0.88 | 0.93 / 0.99 / 1.00 |

| Type | n | bm25 | hybrid | hybrid+layout | hybrid+layout+rerank |
|---|---|---|---|---|---|
| fact | 4 | 1.00 / 1.00 / 0.88 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| topic | 1 | 0.29 / 0.43 / 1.00 | 0.29 / 0.43 / 1.00 | 0.71 / 0.71 / 1.00 | 0.71 / 1.00 / 1.00 |
| exact_citation | 1 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| list | 1 | 0.09 / 0.09 / 1.00 | 0.09 / 0.09 / 1.00 | 0.45 / 0.91 / 1.00 | 0.45 / 0.91 / 1.00 |
| table_row | 2 | 0.00 / 0.50 / 0.09 | 0.50 / 1.00 / 0.15 | 1.00 / 1.00 / 0.75 | 1.00 / 1.00 / 1.00 |
| exhaustive | 1 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| date_range | 1 | 0.75 / 0.75 / 0.33 | 0.75 / 0.75 / 0.33 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| abstain | 1 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 1.00 / 1.00 / 1.00 |

## Results: dev queries

| Tier | n | bm25 | hybrid | hybrid+layout | hybrid+layout+rerank |
|---|---|---|---|---|---|
| Bronze | 7 | 0.82 / 0.83 / 0.83 | 0.82 / 0.87 / 0.90 | 0.94 / 1.00 / 0.90 | 0.94 / 1.00 / 0.90 |
| Silver | 8 | 0.76 / 0.80 / 0.75 | 0.76 / 0.81 / 0.74 | 0.91 / 0.95 / 0.88 | 0.91 / 0.95 / 1.00 |
| Gold | 11 | 1.00 / 1.00 / 0.84 | 1.00 / 1.00 / 0.79 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| Cross-tier | 8 | 0.36 / 0.36 / 0.44 | 0.36 / 0.36 / 0.44 | 0.75 / 0.75 / 0.75 | 1.00 / 1.00 / 1.00 |
| All | 34 | 0.75 / 0.77 / 0.72 | 0.75 / 0.78 / 0.72 | 0.91 / 0.93 / 0.89 | 0.97 / 0.99 / 0.98 |

| Type | n | bm25 | hybrid | hybrid+layout | hybrid+layout+rerank |
|---|---|---|---|---|---|
| fact | 12 | 1.00 / 1.00 / 0.79 | 1.00 / 1.00 / 0.83 | 1.00 / 1.00 / 0.90 | 1.00 / 1.00 / 0.94 |
| name | 3 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| topic | 1 | 0.22 / 0.33 / 1.00 | 0.22 / 0.33 / 1.00 | 0.56 / 1.00 / 1.00 | 0.56 / 1.00 / 1.00 |
| list | 3 | 0.43 / 0.54 / 0.83 | 0.43 / 0.67 / 0.83 | 0.77 / 0.88 / 0.83 | 0.77 / 0.88 / 1.00 |
| exact_citation | 2 | 0.62 / 0.62 / 0.67 | 0.62 / 0.62 / 0.60 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| table_row | 5 | 1.00 / 1.00 / 0.75 | 1.00 / 1.00 / 0.67 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| exhaustive | 2 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| date_range | 2 | 0.45 / 0.45 / 0.75 | 0.45 / 0.45 / 0.75 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| abstain | 4 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.50 / 0.50 / 0.50 | 1.00 / 1.00 / 1.00 |

With the verifier, the provided and dev tables are identical to the default column.

**What each stage fixes.**

- **Layout recovery** carries most of the Silver and Gold gains.
  - List items such as `د - منع التسول والتشرد` inherit their article lead-in, so q06 goes from 1 to 10 of 11 items in the top 10.
  - Header-labelled cells let `حمولة … وكيلها` match row G3-p23, and the character n-grams bridge `كارسياكا` / `کارسیا کا`. q08 moves from rank 107 (BM25) to rank 1.
  - The rebuilt row `41 - 50` separates G2-p13 from the announcement text around it.
- **The section score** keeps a heading-less section together. The Belgian part of B3 continues from the previous issue and has no heading on this page, yet q02 gets all 7 of its passages in the top 7.
- **The reranker** fixes the remaining rank-2 answers (q07, d13) and makes score-based abstention possible.

**Sensitivity.** With the section weight anywhere in 0.3–0.5 and the rerank weight in 0.3–0.7, the dev metrics do not change (0.97 / 0.99 / 0.98) and provided MRR stays at 1.00. Only q02's recall moves (provided R@5 0.91–0.93, R@10 0.96–0.99). At a section weight of 0.6, provided MRR drops to 0.96. Both weights default to 0.5.

## Results: stress queries

| Tier | n | bm25 | hybrid | hybrid+layout | hybrid+layout+rerank | …+verifier |
|---|---|---|---|---|---|---|
| Bronze | 2 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 0.67 | 1.00 / 1.00 / 0.67 |
| Silver | 1 | 1.00 / 1.00 / 0.50 | 1.00 / 1.00 / 0.50 | 1.00 / 1.00 / 0.50 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| Gold | 7 | 0.79 / 0.93 / 0.81 | 0.93 / 1.00 / 0.83 | 0.93 / 0.93 / 0.90 | 0.71 / 0.71 / 0.71 | 0.71 / 0.71 / 0.71 |
| Cross-tier (unanswerable) | 16 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.12 / 0.12 / 0.12 | 0.69 / 0.69 / 0.69 | 0.94 / 0.94 / 0.94 |
| All | 26 | 0.33 / 0.37 / 0.31 | 0.37 / 0.38 / 0.32 | 0.44 / 0.44 / 0.42 | 0.73 / 0.73 / 0.71 | 0.88 / 0.88 / 0.86 |

The configurations without a reranker never abstain on evidence, so they answer every answerable question here but fail almost every unanswerable one. The reranker raises correct abstentions from 2 to 11 of 16, at the cost of two wrong ones (h17, h18), which is why Gold drops. With the verifier:

- 15 of 16 unanswerable queries are abstained on correctly. The only miss is h04, "cases heard in **1950**": the reranker is confident (0.99), because B1-p03 gives case counts, just for 1953 and 1954. So the verifier is never asked.
- 2 of 10 answerable queries are still wrongly abstained on:
  - h18 `ماذا حدث يوم ١٣ ديسمبر ١٩٥٤؟`: a vague question, and the date is written `١٣ - ١٢ - ١٩٥٤` in the passage;
  - h23 "when does the shoe distribution start?": the answer is only `قريبا` ("soon").
- h26 (`من هو الشيخ عبدالله الجابر الصباح؟`) ranks G2-p01 first, which names a different sheikh (`عبدالله المبارك الصباح`), with the right passage third (MRR 0.33).

## Exhaustive, date-range and abstain queries

**Exhaustive.**

- **Trigger:** `كل ما …` or `جميع ما …` ("everything that …").
- **What the system does:** it removes the cue words and searches the remaining topic like any other query. A page is returned when at least one of its passages carries answer evidence (reranker probability ≥ 0.1, or every topic term in one passage; the optional verifier is not used for pages), and pages are ranked by their best passage.
- **Scoring:** the ranked page list gets R@k / MRR, and the returned set gets precision / recall.
- **Results:** q10 (audit office) returns B3, B2; d28 (shipping) returns G3. d27 (municipality law) returns S1 and S3, although S3 never contains the phrase `قانون البلدية`: its passages match through their recovered heading `اختصاصات المجلس البلدي` and the article 23 lead-in. Precision and recall are 1.00 / 1.00 on all three.

**Date range.**

- **Trigger:** a publication cue (forms of `نشر` such as `نشره`, `المنشورة`, or `عدد` directly followed by a date) plus a date. Other wording around the date (`تم`, `لغاية`, weekday names, `المقالات`) is not treated as a topic.
- **Parsed forms:** day-month-year, month-year, year alone, numeric dates (`/`, `-` or `–`, two- or four-digit year), and ranges (`بين ١٨ و ٢٥ ديسمبر ١٩٥٤`, `١٨–٢٥ ديسمبر`, `من … الى …`, `لغاية`), including reversed ranges and ranges that cross a year. A date without a year takes the nearest year in the query, else the latest publication year in the collection. An impossible date (`٣١ نوفمبر`) is moved to the nearest real one, so it matches no issue instead of failing.
- **What the system does:** it filters pages on `publication_date`, and reranks only passages from those pages.
  - With no other topic words, it returns every page in the span, ordered by date.
  - With topic words left, it ranks that topic inside the span, as for exhaustive queries.
  - An empty span abstains (d34, 1 January 1955).
- **Not triggered by:** `عدد` meaning "number of" (`كم كان عدد سكان …`), which stays an ordinary question.
- **Results:** q11, d29 and d30 all have precision and recall of 1.00 / 1.00.

**Abstain.** The system returns `mode: "abstain"` with empty lists in three cases.

1. **No answer evidence.** A passage has evidence if the reranker gives it probability ≥ 0.1, or if it contains every content term of the query. The lexical clause is there because the cross-encoder scores bare keyword queries low even when they match exactly: `ساوثمبتن` gets 0.02 and `التسول` gets 0.07.
   - On the provided and dev sets, answerable passage queries reach at least 0.63, and unanswerable ones at most 0.044.
   - q12 (the flying school's aircraft) peaks at 0.044. The flying-school passages are on topic, but none says how many aircraft it owns.
2. **A cited article exists nowhere**, neither as a header nor inline. d32 asks for article 10, but the pages hold articles 1–6 and 23–26. `المادة 116 من الدستور البلجيكي` is not abstained, because B3-p01 cites article 116 inline.
3. **A date span contains no issue.**

The default system abstains on 5 of 5 abstain queries and on none of the 41 answerable ones in the provided and dev sets.

### Choosing the answerability check

The default check is the weakest part on the stress set: it abstains correctly on 11 of 16 and wrongly on 2 of 10. The bge cross-encoder scores topical match, not whether the answer is present, and no threshold separates the two there. h04 is unanswerable but scores 0.99; h17 is answerable but scores 0.075.

I compared four deciders on top of the same ranking, timed on an Apple M3 Max:

| Decider | Provided (R@5 / R@10 / MRR) | Dev | Stress: correct abstentions | Stress: wrong abstentions | Time per query |
|---|---|---|---|---|---|
| A. bge-reranker evidence ≥ 0.1 (default) | 0.93 / 0.99 / 1.00 | 0.97 / 0.99 / 0.98 | 11/16 | 2/10 | 0.5 s |
| B. Qwen3-Reranker-0.6B with an answerability instruction, as reranker and decider (yes-probability ≥ 0.5) | 0.85 / 0.91 / 0.92; **misses q12** | 0.97 / 0.99 / 0.99 | 13/16 | 1/10 | 4.7 s |
| C. qwen3.5:4b verifies the top 3 for every query | 0.93 / 0.99 / 1.00 | 0.94 / 0.96 / 0.95 | 16/16 | 2/10 | slowest: up to 3 LLM calls on every query |
| **D. Cascade: A when evidence ≥ 0.5, otherwise C** (`--verifier`) | **0.93 / 0.99 / 1.00** | **0.97 / 0.99 / 0.98** | **15/16** | 2/10 | 1.1 s |

- **B** helps on the stress set but loses the one provided abstain query and hurts provided ranking.
- **C** also rejects some correct answers (d03, which asks when the author met the Emir, is wrongly abstained).
- **D** asks the LLM only when the reranker is unsure. It keeps both labelled sets exactly as they were and adds four correct abstentions.

Three caveats:
- The stress set has 16 unanswerable queries, so a difference of one or two queries is within noise.
- Rows B and C were measured before two later parsing fixes (month names inside words, date handling). Those fixes left rows A and D unchanged, so I did not re-run B and C.
- The verifier needs Ollama, so it stays opt-in and the default run keeps a single command with no extra services.

## Three failures

**1. The answer refers back through a pronoun (d04, MRR 0.33).**

- **Query:** `من يتولى تعيين أعضاء ديوان المحاسبة في بلجيكا؟` ("Who appoints the members of the Belgian audit office?").
- **The answer:** B3-p02, `ورغبة في استقلال الديوان قرر أن يتولى تعيينهم مجلس النواب` ("…the Chamber of Representatives appoints *them*"). It names neither Belgium nor the members; `تعيينهم` points back to B3-p01.
- **What wins instead:** B3-p01 contains every concept in the query. It states that article 116 of the Belgian constitution sets how members are appointed, but not by whom. The reranker scores it 0.96 against 0.45 for p02, and B3-p07 (Belgium + audit office) comes second.
- **Why layout doesn't help:** the Belgian part of B3 has no heading on this page, so p02 inherits no context. The section score keeps p02 in the top 3 but cannot reorder passages inside the section.
- **Fix:** index continuation paragraphs (those opening with `و`/`ف` or a pronoun) together with the preceding paragraph as an extra unit, or add LLM-written per-passage context. Alternatively, prefer passages that contain an answer-type span (the body that appoints) over passages that restate the question.

**2. A ship that is not in the table is not abstained (h01).**

- **Query:** `متى تصل الباخرة تيتانيك؟` ("When does the ship Titanic arrive?"). It should abstain, but the default system returns G3 table rows.
- **Why:** the reranker gives 0.16 to the arrivals header row G3-p17 (`البواخر المنتظر وصولها … اسم الباخرة | … | تاريخ وصولها`) and 0.15 to the row G3-p22. The table's schema answers "when does a ship arrive", so the cross-encoder rates it relevant without checking that the named ship exists. The lexical clause can only rescue a match; it never vetoes one.
- **Why the threshold wasn't raised:** 0.16 sits between the unanswerable maximum (0.044) and the answerable minimum (0.63) on the provided and dev sets. But short keyword and topic queries also score in that band (`رصيف الشويخ` 0.33; the exhaustive topic `حركة البواخر الميناء` 0.25), so raising the threshold would trade this error for others.
- **Fix:** check named entities. When a name-like query term never occurs in the corpus, match it with character n-grams against table cells and names; if it is still absent, abstain.
- **With `--verifier`: fixed.** The evidence (0.16) is below 0.5, so qwen3.5:4b reads the top rows, finds no Titanic, and the system abstains.

**3. An aggregation question is abstained (h17).**

- **Query:** `كم عدد الطيارين الذين تخرجوا؟` ("How many pilots graduated?"). It can be answered by counting the names in G2-p02, which G2-p01 introduces as the graduates who received certificates.
- **A transcription gap:** the transcript lists seven names, but the scan (`images/G2.png`) numbers eight. `٥ — عبد الله السمحان` is missing from G2-p02. Retrieval still points at the right passage, but a count taken from the text would be wrong.
- **Retrieval is right:** G2-p01, G2-p02 and G2-p03 are the top 3.
- **Why it abstains:** the best reranker probability is 0.075 (G2-p02). No passage states a number of graduates, and a cross-encoder does not count list items. The lexical clause cannot fire either: `تخرجوا` appears nowhere (the text says `خريجي`). The query therefore scores 0 under the rules above.
- **Same mechanism elsewhere:** `ماذا حدث يوم ١٣ ديسمبر ١٩٥٤؟` (h18) scores 0.012, because G1-p03 writes that date only as `١٣ - ١٢ - ١٩٥٤`.
- **Fix:** let a local LLM read the top passages to decide answerability and do the counting.
- **With `--verifier`: the counting question is fixed.** The LLM confirms that G2-p02 answers it. The date question (h18) still abstains.

## With another week

- Catch near-misses the reranker is confident about, like h04 (the right kind of number, for the wrong year). One way is to check that dates and numbers in the query appear in the passage, or to send confident answers to the verifier when the query contains a year or number.
- Give continuation paragraphs coreference-aware context, so a paragraph that says "appoints *them*" carries its subject (failure 1).
- Tell apart near-identical names such as `عبدالله الجابر` / `عبدالله المبارك` (h26) with exact-phrase matching on person names.
- Test the larger Qwen3-Reranker-4B, and make the verifier the default where Ollama can be assumed.
- Recover layout from `images/` (column and box detection) to confirm the text heuristics and the table reconstruction. Checking the G2 scan by hand already found a name missing from its transcript.
- Parse Hijri dates and issue numbers (`العدد ٢`), add labelled queries with a second annotator, and set thresholds by cross-validation instead of inspection.

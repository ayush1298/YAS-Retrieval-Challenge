> **Solution:** the code is in [`solution/`](solution/), and its approach, results and how to run it are in [REPORT.md](REPORT.md).

# Yas Co retrieval challenge (48 hours)

Build and test a basic retrieval and evaluation workflow over historical Arabic government gazette pages.

You don't need to read Arabic. Every query has an English gloss, and every answer is given as an ID, so you can score your system without reading the text.

## The data

Nine real pages from two December 1954 issues, as **human-verified transcripts**: the text has already been read and checked, so there is no OCR noise to fight. What makes a page hard here is its **layout**, and so how the text is structured once it's written down.

| Tier | Folder | Pages | What makes it that hard |
|---|---|---|---|
| 🥉 **Bronze** | `bronze/` | B1–B3 | Plain two-column running prose, one article per page. |
| 🥈 **Silver** | `silver/` | S1–S3 | Headings and sub-headings over law and regulation articles, with long numbered lists. Answers are spread across many short passages. |
| 🥇 **Gold** | `gold/` | G1–G3 | A front page, a mixed news page and a report built from tables. Short notices sit next to each other, and tables are kept as the transcriber laid them out. Some store a whole column in a single cell, so rows have to be lined up by position. |

Each tier folder holds three page files. Each file has a short header (tier, publication date, layout), then the page's text split into numbered **passages**:

```
[G3-p23] (table row)
کارسیا کا | ٧٥٠٠ ( اسمنت) | يعقوب القطامي | رومانيا | ٢٢- ١٢
```

A passage is one paragraph of the page, with any short heading folded into the paragraph that follows it. Table rows are one passage each, with cells separated by ` | `. The page scans are in `images/`.

The same content, machine-readable, is in `data/`:

| File | Contents |
|---|---|
| `pages.jsonl` | `page_id`, `tier`, `publication_date`, `layout`, `image`, `passages` |
| `passages.jsonl` | `passage_id` (e.g. `S3-p17`), `page_id`, `tier`, `publication_date`, `position` on the page, `is_table_row`, `text`. This is the unit you retrieve. |
| `queries.jsonl` | 12 queries (see below) |

In `queries.jsonl`:

| Field | Meaning |
|---|---|
| `tier` | `B`, `S` or `G` for the tier the answer lives in; `X` when it spans tiers |
| `query` | The question in Arabic, as a user would type it |
| `gloss` | English meaning |
| `type` | `fact`, `topic`, `name`, `list`, `exact_citation`, `table_row`, `exhaustive`, `date_range` or `abstain` |
| `relevant_passages` | The passage IDs that answer it; all of them count |
| `relevant_pages` | Used instead, for page-level questions |
| `expected: "abstain"` | The pages do not answer this one. The right output is "not enough evidence". |

## What to build

A workflow that takes a query and returns ranked passage IDs. Where a query calls for it, it may return page IDs or abstain instead.

It must run **entirely on your own machine**: no paid or hosted APIs. Open models on CPU or a consumer GPU are fine.

## How it's judged

- **Bronze:** it finds the right passage in plain running text.
- **Silver:** it recovers answers split across many short, list-like passages under headings, and handles exact article numbers.
- **Gold:** it gets answers out of tables and crowded mixed pages, including a table whose rows have to be reconstructed.

## What to report

- Recall@5, Recall@10 and MRR, **per tier and per query type**, for at least two configurations (for example BM25 alone vs hybrid)
- What your system does on the `exhaustive`, `date_range` and `abstain` queries, and how you scored them
- Three failures, each explained in detail

We'll also run your system on a set of held-out queries you haven't seen, over the same pages.

## What to send back

- A repository that runs with one command
- A README of at most one page: what you built, the results table, and what you'd do with another week

You have 48 hours from when you receive this.

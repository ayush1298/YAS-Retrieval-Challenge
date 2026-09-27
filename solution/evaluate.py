"""Recall@k / MRR per tier and query type, plus set metrics for page-level and abstain queries."""

import json
from collections import defaultdict
from pathlib import Path

from .search import Result

TIERS = {"B": "Bronze", "S": "Silver", "G": "Gold", "X": "Cross-tier"}
METRICS = ("R@5", "R@10", "MRR")


def load_queries(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _page(passage_id: str) -> str:
    return passage_id.rsplit("-", 1)[0]


def score(query: dict, result: Result) -> dict[str, float]:
    """Abstain queries score 1 on every metric when the system abstains and 0 otherwise;
    answerable queries score 0 when it abstains. Page-level queries are scored on page IDs: the
    returned pages, or the pages of the top 10 passages when the system answered with passages."""
    if query.get("expected") == "abstain":
        hit = float(result.mode == "abstain")
        return dict.fromkeys(METRICS, hit)
    relevant = set(query.get("relevant_pages") or query["relevant_passages"])
    if not query.get("relevant_pages"):
        ranked = result.passages
    elif result.mode == "pages":
        ranked = result.pages
    else:
        ranked = list(dict.fromkeys(_page(p) for p in result.passages[:10]))
    first = next((rank for rank, doc in enumerate(ranked, 1) if doc in relevant), None)
    return {
        "R@5": len(relevant & set(ranked[:5])) / len(relevant),
        "R@10": len(relevant & set(ranked[:10])) / len(relevant),
        "MRR": 1 / first if first else 0.0,
    }


def set_metrics(query: dict, result: Result) -> tuple[float, float]:
    """Precision and recall of the returned page set, for page-level queries."""
    relevant, returned = set(query["relevant_pages"]), set(result.pages if result.mode == "pages" else [])
    hits = len(relevant & returned)
    return (hits / len(returned) if returned else 0.0), hits / len(relevant)


def report(queries: list[dict], runs: dict[str, dict[str, Result]]) -> str:
    """R@5 / R@10 / MRR by tier and by query type, one column per configuration."""
    scores = {name: [score(q, results[q["query_id"]]) for q in queries] for name, results in runs.items()}
    by_tier, by_type = defaultdict(list), defaultdict(list)
    for i, q in enumerate(queries):
        by_tier[TIERS[q["tier"]]].append(i)
        by_type[q["type"]].append(i)
    tiers = {t: by_tier[t] for t in TIERS.values() if t in by_tier} | {"All": list(range(len(queries)))}

    tables = []
    for label, groups in (("Tier", tiers), ("Type", by_type)):
        lines = [f"| {label} | n | " + " | ".join(runs) + " |", "|---|---|" + "---|" * len(runs)]
        for group, idx in groups.items():
            cells = [
                " / ".join(f"{sum(scores[name][i][m] for i in idx) / len(idx):.2f}" for m in METRICS)
                for name in runs
            ]
            lines.append(f"| {group} | {len(idx)} | " + " | ".join(cells) + " |")
        tables.append("\n".join(lines))
    return "Cells: R@5 / R@10 / MRR\n\n" + "\n\n".join(tables)


def details(queries: list[dict], results: dict[str, Result]) -> str:
    lines = ["| id | type | mode | top 5 | R@10 | MRR | set P/R |", "|---|---|---|---|---|---|---|"]
    for q in queries:
        result = results[q["query_id"]]
        s = score(q, result)
        top = result.pages if q.get("relevant_pages") else result.passages
        pr = "{:.2f}/{:.2f}".format(*set_metrics(q, result)) if q.get("relevant_pages") else ""
        lines.append(
            f"| {q['query_id']} | {q['type']} | {result.mode} | {' '.join(top[:5])} | {s['R@10']:.2f} | {s['MRR']:.2f} | {pr} |"
        )
    return "\n".join(lines)

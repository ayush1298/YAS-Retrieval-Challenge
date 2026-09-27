"""Command line: search one query, run a query file, or evaluate every configuration."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from .evaluate import details, load_queries, report
from .search import CONFIGS, Retriever

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
QUERY_SETS = (DATA / "queries.jsonl", ROOT / "eval" / "dev_queries.jsonl", ROOT / "eval" / "hard_queries.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(prog="solution")
    parser.add_argument("--verifier", metavar="MODEL", help="Ollama model that checks answerability, e.g. qwen3.5:4b")
    commands = parser.add_subparsers(dest="command", required=True)
    search = commands.add_parser("search", help="rank passages for one query")
    search.add_argument("query")
    search.add_argument("-k", type=int, default=10)
    run = commands.add_parser("run", help="write ranked results for a query file as JSONL")
    run.add_argument("queries", type=Path)
    run.add_argument("-k", type=int, default=10)
    evaluate = commands.add_parser("evaluate", help="score every configuration on labelled query files")
    evaluate.add_argument("queries", type=Path, nargs="*", default=list(QUERY_SETS))
    evaluate.add_argument("--details", action="store_true", help="print per-query results")
    args = parser.parse_args()

    default = replace(CONFIGS[-1], verifier=args.verifier)
    if args.command == "search":
        result = Retriever(DATA, default).search(args.query)
        print(json.dumps({**asdict(result), "passages": result.passages[: args.k]}, ensure_ascii=False, indent=2))
    elif args.command == "run":
        retriever = Retriever(DATA, default)
        for q in load_queries(args.queries):
            result = retriever.search(q["query"])
            record = {"query_id": q["query_id"], **asdict(result), "passages": result.passages[: args.k]}
            print(json.dumps(record, ensure_ascii=False))
    else:
        configs = CONFIGS + ((replace(default, name=f"{default.name}+verifier"),) if args.verifier else ())
        query_sets = {path: load_queries(path) for path in args.queries}
        runs = {path: {} for path in query_sets}
        for config in configs:
            retriever = Retriever(DATA, config)
            for path, queries in query_sets.items():
                runs[path][config.name] = {q["query_id"]: retriever.search(q["query"]) for q in queries}
        for path, queries in query_sets.items():
            print(f"## {path.name}\n\n{report(queries, runs[path])}\n")
            if args.details:
                for name, results in runs[path].items():
                    print(f"### {name}\n\n{details(queries, results)}\n")


if __name__ == "__main__":
    main()

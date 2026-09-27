"""Corpus loading and layout recovery: sections, law articles, lists and tables."""

import json
import re
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path

from .text import normalize

_END_PUNCT = tuple(".:؛،,!؟?…-–")
_LETTER_ITEM = re.compile(r"^[ء-ي](?:\s*[-–)]|\s)")
_ITEM = re.compile(
    r"^(?:(?P<letter>[ء-ي](?:\s*[-–)]|\s))|(?P<number>\d+\s*[-–)])"
    r"|(?P<ordinal>(?:ال)?(?:اول|ثاني|ثالث|رابع|خامس|سادس)[اه]?\s*[-–]))"
)
_ARTICLE = re.compile(r"^(?:ال)?ماده\s*(\d+)")
_NUMBERED = re.compile(r"^(\d+)\s*[-–]\s*")
_BULLET = re.compile(r"^[*•]")
_DATES = re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}")
_RANGES = re.compile(r"(\d+)\s*(?:[-–]|الي(?:\s+رقم)?)\s*(\d+)")


@dataclass
class Passage:
    id: str
    page: str
    position: int
    is_table_row: bool
    text: str
    section: int = 0
    context: str = ""
    body: str = ""
    articles: set[int] = field(default_factory=set)
    rows: list[str] = field(default_factory=list)

    @property
    def units(self) -> list[str]:
        """Texts indexed for this passage: the passage in its context, plus any reconstructed table rows."""
        return [f"{self.context}\n{text}".strip() for text in [self.body, *self.rows]]


@dataclass
class Page:
    id: str
    date: str


def _is_heading(line: str, allow_digits: bool = False) -> bool:
    line = normalize(line)
    return (
        0 < len(line.split()) <= 15
        and not line.endswith(_END_PUNCT)
        and "|" not in line
        and not _ARTICLE.match(line)
        and not _LETTER_ITEM.match(line)
        and (allow_digits or not re.search(r"\d", _NUMBERED.sub("", line)))
    )


def _heading_lines(lines: list[str], page_start: bool) -> list[str]:
    """Short title lines folded into the top of a passage; a lone title line counts as a heading passage."""
    if len(lines) == 1:
        single = lines[0]
        return lines if _is_heading(single, allow_digits=page_start) and not _ITEM.match(normalize(single)) else []
    count = 0
    while count < len(lines) and _is_heading(lines[count]):
        count += 1
    return lines[:count]


def _split_column(cell: str) -> list[str]:
    """Split a cell that holds a whole column into its values."""
    norm = normalize(cell)
    if len(dates := _DATES.findall(norm)) > 1:
        return dates
    if len(ranges := _RANGES.findall(norm)) > 1:
        return [f"{a} - {b}" for a, b in ranges]
    return norm.split()


def _label(header: list[str], cells: list[str]) -> str:
    if len(header) != len(cells):
        return " | ".join(cells)
    return " | ".join(f"{h}: {c}" for h, c in zip(header, cells))


def _layout_tables(passages: list[Passage]) -> None:
    """Label cells with their column header and rebuild rows from cells that hold a whole column."""
    header, caption, previous, in_table = [], "", "", False
    for p in passages:
        if not p.is_table_row:
            previous, in_table = p.text, False
            continue
        lines = p.text.split("\n")
        titles = "\n".join(line for line in lines if "|" not in line)
        cells = [c.strip() for line in lines if "|" in line for c in line.split("|")]
        if titles or not in_table:
            header, caption, in_table = cells, titles or previous, True
            p.context, p.body = caption, " | ".join(cells)
            continue
        p.context, p.body = caption, _label(header, cells)
        columns = [_split_column(c) for c in cells]
        if len({len(c) for c in columns}) == 1 and len(columns[0]) >= 3:
            p.rows = [_label(header, list(row)) for row in zip(*columns)]


def _marker(line: str) -> str | None:
    match = _ITEM.match(line)
    return match.lastgroup if match else None


def _article(line: str, numbered: bool) -> int | None:
    """Article number opening a line: 'مادة 25 -', or a bare '3-' for top-level numbered regulations."""
    match = _ARTICLE.match(line) or (_NUMBERED.match(line) if numbered else None)
    return int(match.group(1)) if match else None


def _contains(outer: str, inner: str) -> bool:
    return f"\n{inner}\n" in f"\n{outer}\n"


def _context(parts: list[str], text: str) -> str:
    """Join inherited context, dropping parts already in the passage or inside another part."""
    parts = list(dict.fromkeys(c for c in parts if c and not _contains(text, c)))
    return "\n".join(c for c in parts if not any(c != o and _contains(o, c) for o in parts))


def _layout_prose(passages: list[Passage], count: int) -> int:
    """Assign sections (split at headings, notice bullets and tables) and give each passage the
    heading, law article and list lead-in it sits under. Prose after a table resumes the section
    the table interrupted."""
    count += 1
    section, heading = count, ""
    article_lead = list_lead = ""
    article, was_table, resume = None, False, None
    for i, p in enumerate(passages):
        lines = [line.strip() for line in p.text.split("\n") if line.strip()]
        titles = _heading_lines(lines, page_start=i == 0)
        table_starts = p.is_table_row and not was_table
        if titles or _BULLET.match(lines[0]) or table_starts:
            if table_starts:
                resume = section, heading
            count += 1
            section, heading = count, "\n".join(titles)
            article_lead, list_lead, article = "", "", None
        elif was_table and not p.is_table_row:
            section, heading = resume
        was_table, p.section = p.is_table_row, section
        if p.is_table_row:
            continue

        norm_lines = [normalize(line) for line in lines]
        in_list = bool(list_lead) and _marker(norm_lines[0]) not in (None, _marker(normalize(list_lead)))
        headers = [_article(norm, numbered=j == 0 and not titles and not in_list) for j, norm in enumerate(norm_lines)]
        continues_article = article is not None and headers[0] is None
        if continues_article:
            p.articles.add(article)
        inherited = [heading, article_lead if continues_article else "", list_lead if in_list else ""]
        if not in_list:
            list_lead = ""
        for line, number in zip(lines, headers):
            if number is not None:
                article, article_lead, list_lead = number, line, ""
                p.articles.add(article)

        p.context, p.body = _context(inherited, p.text), p.text
        if norm_lines[-1].rstrip("-").endswith(":"):
            list_lead = p.text
    return count


def load(data_dir: Path) -> tuple[list[Page], list[Passage]]:
    pages = [Page(r["page_id"], r["publication_date"]) for r in _jsonl(data_dir / "pages.jsonl")]
    passages = [
        Passage(r["passage_id"], r["page_id"], r["position"], r["is_table_row"], r["text"])
        for r in _jsonl(data_dir / "passages.jsonl")
    ]
    passages.sort(key=lambda p: (p.page, p.position))
    section = 0
    for _, group in groupby(passages, key=lambda p: p.page):
        page_passages = list(group)
        section = _layout_prose(page_passages, section)
        _layout_tables(page_passages)
    return pages, passages


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

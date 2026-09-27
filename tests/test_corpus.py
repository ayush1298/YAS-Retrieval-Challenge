from datetime import date
from pathlib import Path

import pytest

from solution.corpus import load

DATA = Path(__file__).resolve().parent.parent / "data"
WEEKDAYS = {"الاثنين": 0, "الثلاثاء": 1, "الاربعاء": 2, "السبت": 5, "الاحد": 6}


@pytest.fixture(scope="module")
def passages():
    _, passages = load(DATA)
    return {p.id: p for p in passages}


def _with_article(passages, number, page):
    return {pid for pid, p in passages.items() if p.page == page and number in p.articles}


def test_articles_span_their_items(passages):
    assert _with_article(passages, 2, "S1") == {f"S1-p{i:02d}" for i in range(10, 21)}
    assert _with_article(passages, 23, "S3") == {f"S3-p{i:02d}" for i in range(1, 17)}
    assert _with_article(passages, 26, "S3") == {f"S3-p{i:02d}" for i in range(18, 22)}
    assert passages["S3-p16"].articles == {23, 24}
    assert passages["S3-p17"].articles == {25}


def test_numbered_regulations_are_articles_but_list_items_and_headings_are_not(passages):
    assert passages["S2-p04"].articles == {2}
    assert passages["S2-p07"].articles == {3}
    assert not passages["B2-p22"].articles
    assert not passages["B3-p08"].articles


def test_list_items_inherit_their_lead_in(passages):
    assert passages["S1-p12"].context.endswith("مادة ٢ - اختصاصات البلدية هي :")
    assert passages["B2-p21"].context.endswith("وعليه ان يتحقق قبل التأثير بالصرف مما يأتي :")
    assert passages["S2-p07"].context == "التعليمات والقواعد المتعلقة بحق ممارسة العمل في الكويت"
    assert "مما يأتي" not in passages["B2-p19"].context


def test_sections_split_at_headings_and_notices(passages):
    assert passages["B3-p07"].section != passages["B3-p08"].section
    assert passages["B3-p13"].section == passages["B3-p16"].section
    assert len({passages[f"G2-p{i:02d}"].section for i in (8, 9, 10)}) == 3
    assert passages["G2-p14"].section == passages["G2-p11"].section != passages["G2-p13"].section


def test_table_cells_carry_header_and_caption(passages):
    row = passages["G3-p23"]
    assert row.context == "البواخر المنتظر وصولها من ١٨ - ١٢ لغاية ٢٤ – ١٢"
    assert "اسم الوكيل: يعقوب القطامي" in row.body
    assert "حمولتها بالأطنان: ٧٥٠٠ ( اسمنت)" in row.body


def test_column_packed_table_is_rebuilt_in_scan_order(passages):
    rows = [dict(cell.split(": ") for cell in row.split(" | ")) for row in passages["G2-p13"].rows]
    assert len(rows) == 12
    assert rows[4] == {"اليوم": "الاثنين", "التاريخ": "3/1/1955", "الارقام الفائزة في القرعة": "41 - 50"}
    for i, row in enumerate(rows):
        day, month, year = map(int, row["التاريخ"].split("/"))
        assert date(year, month, day).weekday() == WEEKDAYS[row["اليوم"]]
        assert int(row["الارقام الفائزة في القرعة"].split(" - ")[0]) == 10 * i + 1

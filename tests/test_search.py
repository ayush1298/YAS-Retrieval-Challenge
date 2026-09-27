from pathlib import Path

import pytest

from solution.search import Config, Retriever

DATA = Path(__file__).resolve().parent.parent / "data"
ISSUE_25_DEC = ["B3", "G2", "S2", "S3"]


@pytest.fixture(scope="module")
def retriever():
    return Retriever(DATA, Config("lexical+layout", channels=("word", "char"), rerank=False))


@pytest.mark.parametrize(
    "query, pages",
    [
        ("كل ما نشر في عدد ٢٥ ديسمبر ١٩٥٤", ISSUE_25_DEC),
        ("كل ما تم نشره في عدد ٢٥ ديسمبر", ISSUE_25_DEC),
        ("كل المقالات المنشورة في عدد 25/12/1954", ISSUE_25_DEC),
        ("كل ما نشر يوم السبت ٢٥ ديسمبر ١٩٥٤", ISSUE_25_DEC),
        ("كل ما نشر من ١٨ ديسمبر لغاية ٢٥ ديسمبر ١٩٥٤", ["B1", "B2", "B3", "G1", "G2", "G3", "S1", "S2", "S3"]),
    ],
)
def test_issue_date_queries_return_every_page_of_the_issue(retriever, query, pages):
    result = retriever.search(query)
    assert result.mode == "pages"
    assert sorted(result.pages) == pages


@pytest.mark.parametrize(
    "query",
    ["كل ما نشر في عدد ١ يناير ١٩٥٥", "كل ما نشر في عدد ٣١ نوفمبر ١٩٥٤", "ما نص المادة العاشرة من قانون البلدية؟"],
)
def test_missing_issue_or_article_abstains(retriever, query):
    assert retriever.search(query).mode == "abstain"


@pytest.mark.parametrize(
    "query, first",
    [
        ("قانون البلدية المادة ٢٥", ["S3-p17"]),
        ("المادة الخامسة و العشرون من قانون البلدية", ["S3-p17"]),
        ("المادة ٢٦ من قانون البلدية", ["S3-p18", "S3-p19", "S3-p20", "S3-p21"]),
    ],
)
def test_cited_article_is_returned_first(retriever, query, first):
    assert retriever.search(query).passages[: len(first)] == first

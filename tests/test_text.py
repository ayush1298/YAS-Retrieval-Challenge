from datetime import date

import pytest

from solution.text import article_number, char_terms, date_span, normalize, word_terms


@pytest.mark.parametrize(
    "query, number",
    [
        ("قانون البلدية المادة ٢٥", 25),
        ("اختصاصات البلدية حسب المادة الثانية", 2),
        ("نص المادة الأولى", 1),
        ("المادة الحادية عشرة", 11),
        ("المادة الخامسة والعشرون", 25),
        ("المادة رقم 116 من الدستور", 116),
        ("في المادة السابقة", None),
        ("متى أسست أول محكمة مستقلة في الكويت؟", None),
    ],
)
def test_article_number(query, number):
    assert article_number(query) == number


@pytest.mark.parametrize(
    "query, span",
    [
        ("كل ما نشر في عدد ٢٥ ديسمبر ١٩٥٤", (date(1954, 12, 25), date(1954, 12, 25))),
        ("ما نشر بين ١٨ و ٢٥ ديسمبر ١٩٥٤", (date(1954, 12, 18), date(1954, 12, 25))),
        ("من ١٨ ديسمبر الى ٢٤ ديسمبر ١٩٥٤", (date(1954, 12, 18), date(1954, 12, 24))),
        ("كل ما نشر في ديسمبر ١٩٥٤", (date(1954, 12, 1), date(1954, 12, 31))),
        ("عدد 18/12/1954", (date(1954, 12, 18), date(1954, 12, 18))),
        ("ما نشر في عام ١٩٥٥", (date(1955, 1, 1), date(1955, 12, 31))),
        ("متى أسست أول محكمة مستقلة في الكويت؟", None),
    ],
)
def test_date_span(query, span):
    assert date_span(query) == span


def test_normalize_unifies_orthography():
    assert normalize("کارسیا کا") == normalize("كارسيا كا")
    assert normalize("١٩٥٤ ۸۱۷") == "1954 817"
    assert normalize("عبد الله") == normalize("عبدالله")
    assert normalize("مُحَمَّد") == "محمد"


def test_stemming_matches_inflected_forms():
    assert word_terms("للأرقام الفائزة") == word_terms("الارقام فائزة")


def test_char_ngrams_bridge_split_words():
    shared = set(char_terms("كارسياكا")) & set(char_terms("کارسیا کا"))
    assert {" كار", "كارس", "ارسي", "رسيا"} <= shared

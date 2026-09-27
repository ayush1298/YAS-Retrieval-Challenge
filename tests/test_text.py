from datetime import date

import pytest

from solution.text import article_number, char_terms, date_span, expand_months, normalize, word_terms


@pytest.mark.parametrize(
    "query, number",
    [
        ("قانون البلدية المادة ٢٥", 25),
        ("اختصاصات البلدية حسب المادة الثانية", 2),
        ("نص المادة الأولى", 1),
        ("المادة الحادية عشرة", 11),
        ("المادة الخامسة والعشرون", 25),
        ("المادة الخامسة و العشرون", 25),
        ("المادة رقم 116 من الدستور", 116),
        ("ما النص الكامل للمادة العاشرة", 10),
        ("بالمادة ٢٤ من القانون", 24),
        ("المادة٢٥", 25),
        ("المادة (٢٥)", 25),
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
        ("ما نشر بين ٢٥ و ١٨ ديسمبر ١٩٥٤", (date(1954, 12, 18), date(1954, 12, 25))),
        ("كل ما نشر ١٨–٢٥ ديسمبر ١٩٥٤", (date(1954, 12, 18), date(1954, 12, 25))),
        ("من ١٨ ديسمبر الى ٢٤ ديسمبر ١٩٥٤", (date(1954, 12, 18), date(1954, 12, 24))),
        ("كل ما نشر من ٢٠ ديسمبر الى ٥ يناير ١٩٥٥", (date(1954, 12, 20), date(1955, 1, 5))),
        ("كل ما نشر من ٢٠ ديسمبر ١٩٥٤ الى ٥ يناير", (date(1954, 12, 20), date(1955, 1, 5))),
        ("كل ما نشر في ديسمبر ١٩٥٤", (date(1954, 12, 1), date(1954, 12, 31))),
        ("كل ما نشر في عدد ٢٥ ديسمبر", (date(1954, 12, 25), date(1954, 12, 25))),
        ("عدد 18/12/1954", (date(1954, 12, 18), date(1954, 12, 18))),
        ("عدد 25–12–1954", (date(1954, 12, 25), date(1954, 12, 25))),
        ("كل ما نشر في عدد 25/12/54", (date(1954, 12, 25), date(1954, 12, 25))),
        ("كل ما نشر في عدد 12/25/1954", (date(1954, 12, 25), date(1954, 12, 25))),
        ("ما نشر في عام ١٩٥٥", (date(1955, 1, 1), date(1955, 12, 31))),
        ("كل ما نشر في عام ٢٠٢٤", (date(2024, 1, 1), date(2024, 12, 31))),
        ("متى أسست أول محكمة مستقلة في الكويت؟", None),
    ],
)
def test_date_span(query, span):
    assert date_span(query, default_year=1954) == span


@pytest.mark.parametrize(
    "query", ["كل ما نشر في عدد ٣١ نوفمبر ١٩٥٤", "ما نشر في ٣١ فبراير ١٩٥٤", "عدد 0 ديسمبر 1954", "ديسمبر 9999"]
)
def test_impossible_dates_do_not_crash(query):
    start, end = date_span(query, default_year=1954)
    assert start <= end


def test_month_names_match_whole_words_only():
    assert expand_months("رخصة حق ممارسة العمل") == "رخصة حق ممارسة العمل"
    assert expand_months("عدد ٢٥ ديسمبر").endswith(" 12")


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

"""Arabic normalisation, tokenisation and number/date parsing."""

import re
from datetime import date, timedelta

_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")
_CHARS = str.maketrans(
    "أإآٱىیئةکؤ٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "اااايييهكو01234567890123456789",
)
_TOKEN = re.compile(r"\d+|[^\W\d_]+")

STOPWORDS = frozenset(
    "في من الي علي عن ما ماذا متي كم كيف اين هل لماذا اي هو هي هم هذا هذه ذلك تلك "
    "التي الذي الذين ان او ام ثم قد لا لم لن كل جميع مع بعد قبل بين حتي اذا عند كان "
    "كانت يكون تكون و ف ب ل ك له لها به بها فيه فيها منه منها عليه عليها".split()
)

_PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال")
_SUFFIXES = ("ها", "ان", "ات", "ون", "ين", "يه", "ه", "ي")


def normalize(text: str) -> str:
    text = _DIACRITICS.sub("", text).translate(_CHARS).lower()
    return re.sub(r"\bعبد\s+ال", "عبدال", text)


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(normalize(text)) if t not in STOPWORDS]


def stem(token: str) -> str:
    """Light10 stemmer (Larkey et al., 2007)."""
    if token.isdigit() or not "ء" <= token[0] <= "ي":
        return token
    if token.startswith("و") and len(token) > 3:
        token = token[1:]
    for prefix in _PREFIXES:
        if token.startswith(prefix) and len(token) - len(prefix) > 1:
            token = token[len(prefix):]
            break
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) > 1:
            token = token[: -len(suffix)]
    return token


def word_terms(text: str) -> list[str]:
    return [stem(t) for t in tokens(text)]


def char_terms(text: str, sizes=(3, 4, 5)) -> list[str]:
    grams = []
    for token in tokens(text):
        padded = f" {token} "
        grams += [padded[i : i + n] for n in sizes for i in range(len(padded) - n + 1)]
    return grams


_UNITS = "اول|اولي|حادي|حاديه ثاني|ثانيه ثالث|ثالثه رابع|رابعه خامس|خامسه سادس|سادسه سابع|سابعه ثامن|ثامنه تاسع|تاسعه عاشر|عاشره"
ORDINALS = {w: i + 1 for i, group in enumerate(_UNITS.split()) for w in group.split("|")}
TENS = {w: 10 * (i + 2) for i, pair in enumerate(
    "عشرون|عشرين ثلاثون|ثلاثين اربعون|اربعين خمسون|خمسين ستون|ستين سبعون|سبعين ثمانون|ثمانين تسعون|تسعين".split()
) for w in pair.split("|")}
_ARTICLE_REF = re.compile(r"\b(?:ال)?ماده\s+(?:رقم\s+)?(\d+|\S+(?:\s+\S+)?)")


def article_number(query: str) -> int | None:
    """Article cited in a query: 'المادة ٢٥', 'المادة الثانية', 'المادة الخامسة والعشرون'."""
    match = _ARTICLE_REF.search(normalize(query))
    if not match:
        return None
    words = [w.removeprefix("ال").removeprefix("وال") for w in match.group(1).split()]
    if words[0].isdigit():
        return int(words[0])
    if words[0] in TENS:
        return TENS[words[0]]
    if words[0] not in ORDINALS:
        return None
    number = ORDINALS[words[0]]
    if len(words) > 1 and words[1] in ("عشر", "عشره"):
        number += 10
    elif len(words) > 1 and words[1] in TENS:
        number += TENS[words[1]]
    return number


MONTHS = {
    "يناير": 1, "فبراير": 2, "مارس": 3, "ابريل": 4, "مايو": 5, "يونيو": 6, "يونيه": 6,
    "يوليو": 7, "يوليه": 7, "اغسطس": 8, "سبتمبر": 9, "اكتوبر": 10, "نوفمبر": 11, "ديسمبر": 12,
}
_MONTH = "|".join(MONTHS)
_DAY_MONTH_YEAR = re.compile(rf"(?:\b(\d{{1,2}})\s*(?:و|الي|حتي|-)\s*)?\b(\d{{1,2}})?\s*({_MONTH})\s*(\d{{4}})?")
_NUMERIC_DATE = re.compile(r"\b(\d{1,2})\s*[/-]\s*(\d{1,2})\s*[/-]\s*(\d{4})\b")
_YEAR = re.compile(r"\b(1[89]\d\d)\b")


def _month_end(year: int, month: int) -> date:
    return date(year + month // 12, month % 12 + 1, 1) - timedelta(days=1)


def date_span(query: str) -> tuple[date, date] | None:
    """Calendar span mentioned in a query: a day, a month, a year or a range between two of them."""
    text = normalize(query)
    years = _YEAR.findall(text)
    spans = []
    for first, day, month_name, year in _DAY_MONTH_YEAR.findall(text):
        if not (year or years):
            continue
        year, month = int(year or years[-1]), MONTHS[month_name]
        if day:
            spans.append((date(year, month, int(first or day)), date(year, month, int(day))))
        else:
            spans.append((date(year, month, 1), _month_end(year, month)))
    for day, month, year in _NUMERIC_DATE.findall(text):
        spans.append((date(int(year), int(month), int(day)),) * 2)
    if not spans:
        spans = [(date(int(y), 1, 1), date(int(y), 12, 31)) for y in _YEAR.findall(text)]
    if not spans:
        return None
    return min(s for s, _ in spans), max(e for _, e in spans)


def expand_months(query: str) -> str:
    """Append month numbers so 'ديسمبر' also matches dates written as '12/1954'."""
    months = [str(MONTHS[m]) for m in re.findall(_MONTH, normalize(query))]
    return " ".join([query, *months])

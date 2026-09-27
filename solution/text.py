"""Arabic normalisation, tokenisation and number/date parsing."""

import re
from calendar import monthrange
from datetime import date

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
_ARTICLE_REF = re.compile(r"(?:^|\W)[وفبك]?(?:ال|لل)?ماده")


def article_number(query: str) -> int | None:
    """Article cited in a query: 'المادة ٢٥', 'للمادة (٢٥)', 'المادة الثانية', 'المادة الخامسة و العشرون'."""
    text = normalize(query)
    match = _ARTICLE_REF.search(text)
    if not match:
        return None
    following = _TOKEN.findall(text[match.end():])[:4]
    words = [w.removeprefix("و").removeprefix("ال") for w in following if w not in ("و", "رقم")] or [""]
    first, second = words[0], words[1] if len(words) > 1 else ""
    if first.isdigit():
        return int(first)
    if first in TENS:
        return TENS[first]
    if first not in ORDINALS:
        return None
    return ORDINALS[first] + (10 if second in ("عشر", "عشره") else TENS.get(second, 0))


MONTHS = {
    "يناير": 1, "فبراير": 2, "مارس": 3, "ابريل": 4, "مايو": 5, "يونيو": 6, "يونيه": 6,
    "يوليو": 7, "يوليه": 7, "اغسطس": 8, "سبتمبر": 9, "اكتوبر": 10, "نوفمبر": 11, "ديسمبر": 12,
}
_MONTH = "|".join(MONTHS)
_DAY_MONTH_YEAR = re.compile(
    rf"(?:\b(\d{{1,2}})\s*(?:و|الي|حتي|لغايه|-|–)\s*)?\b(\d{{1,2}})?\s*({_MONTH})\s*(\d{{4}})?"
)
_NUMERIC_DATE = re.compile(r"\b(\d{1,2})\s*[/\-–]\s*(\d{1,2})\s*[/\-–]\s*(\d{4}|\d{2})\b")
_YEAR = re.compile(r"\b(1[89]\d\d|20\d\d)\b")


def _valid_date(year: int, month: int, day: int) -> date:
    """Nearest real date, so '31 November' or a month/day swap never raises."""
    if month > 12 >= day:
        month, day = day, month
    year, month = max(year, 1), min(max(month, 1), 12)
    return date(year, month, min(max(day, 1), monthrange(year, month)[1]))


def _shift(day: date, years: int) -> date:
    return _valid_date(day.year + years, day.month, day.day)


def date_span(query: str, default_year: int | None = None) -> tuple[date, date] | None:
    """Calendar span mentioned in a query: a day, a month, a year or a range between them. A date
    without a year takes the nearest year written in the query, else `default_year`; a range that
    would run backwards is read as crossing into the next year."""
    text = normalize(query)
    years = [(m.start(), int(m.group(1))) for m in _YEAR.finditer(text)]
    mentions = []  # (start, end, year was inferred)
    for m in _DAY_MONTH_YEAR.finditer(text):
        first, day, month_name, year = m.groups()
        later = [y for pos, y in years if pos > m.start()]
        earlier = [y for pos, y in years if pos < m.start()]
        inferred_year = later[0] if later else earlier[-1] if earlier else default_year
        if not (year or inferred_year):
            continue
        year, month = int(year or inferred_year), MONTHS[month_name]
        days = sorted(int(d) for d in (first, day) if d) or [1, monthrange(year, month)[1]]
        mentions.append((_valid_date(year, month, days[0]), _valid_date(year, month, days[-1]), not m.group(4)))
    for i in range(1, len(mentions)):
        (start0, end0, inferred0), (start1, end1, inferred1) = mentions[i - 1], mentions[i]
        if start1 < end0 and inferred1:
            mentions[i] = (_shift(start1, 1), _shift(end1, 1), True)
        elif start1 < end0 and inferred0:
            mentions[i - 1] = (_shift(start0, -1), _shift(end0, -1), True)
    century = (default_year or 1900) // 100 * 100
    for day, month, year in _NUMERIC_DATE.findall(text):
        full_year = int(year) + (century if len(year) == 2 else 0)
        mentions.append((_valid_date(full_year, int(month), int(day)),) * 2 + (False,))
    if not mentions:
        mentions = [(date(y, 1, 1), date(y, 12, 31), False) for _, y in years]
    if not mentions:
        return None
    return min(m[0] for m in mentions), max(m[1] for m in mentions)


def expand_months(query: str) -> str:
    """Append month numbers so 'ديسمبر' also matches dates written as '12/1954'."""
    months = [str(MONTHS[m]) for m in re.findall(rf"\b(?:{_MONTH})\b", normalize(query))]
    return " ".join([query, *months])

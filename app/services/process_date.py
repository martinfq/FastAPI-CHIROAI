from difflib import SequenceMatcher
import re
import unicodedata
from datetime import date, datetime

_MONTHS: dict[str, int] = {
    "enero": 1, "febrero": 2, "marzo": 3,
    "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9,
    "octubre": 10, "noviembre": 11, "diciembre": 12,
}

_DATE_RE = re.compile(
    r"""
    \b
    (?:el\s*)?
    (\d{1,2})
    (?:\s*de\s*|\s+)
    ([a-záéíóúñ0-9]{3,12})
    (?:\s*de\s*|\s+)
    (\d{2,4})
    \b
    """,
    re.VERBOSE | re.IGNORECASE,
)


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    without_accents = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", without_accents).strip().lower()


def find_month(text: str) -> int | None:
    """Find month number by exact match or length filter + similarity scoring."""
    normalized = normalize_text(text)
    if normalized in _MONTHS:
        return _MONTHS[normalized]

    best_month: int | None = None
    best_score = 0.0

    for month_name, month_num in _MONTHS.items():
        # Filter out months with incompatible lengths (tolerance of +-2 chars)
        if abs(len(month_name) - len(normalized)) > 2:
            continue

        score = SequenceMatcher(None, normalized, month_name).ratio()
        if score > best_score:
            best_score = score
            best_month = month_num

    if best_score >= 0.70:
        return best_month

    return None


def parse_date(text: str) -> date | None:
    normalized = normalize_text(text)
    match = _DATE_RE.search(normalized)
    if not match:
        return None
    day_str, month_name, year_str = match.groups()
    month = find_month(month_name)
    if not month:
        return None

    try:
        day = int(day_str)
        year = int(year_str)
        if year < 100:
            year += 2000
        return datetime(year, month, day).date()
    except ValueError:
        return None


def extract_date(texts: list[str]) -> date | None:
    for text in texts:
        if any(char.isdigit() for char in text):
            parsed = parse_date(text)
            if parsed is not None:
                return parsed
    return None

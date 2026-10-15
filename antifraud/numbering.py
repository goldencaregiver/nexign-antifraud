"""Справочник нумерации: регион, часовой пояс и оператор по номеру.

В реальной сети это реестр Россвязи (DEF-коды) и база переносимых номеров.
Здесь — детерминированная модель, чтобы генератор и признаки сходились.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache

N_REGIONS = 85
MOSCOW = 77
SPB = 78

MOBILE_OPERATORS = ("MTS", "Beeline", "MegaFon", "T2")

LANDLINE_CODES = {
    "495": MOSCOW,
    "499": MOSCOW,
    "812": SPB,
    "343": 66,
    "383": 54,
    "843": 16,
    "861": 23,
    "846": 63,
    "4212": 27,
    "423": 25,
}

EMERGENCY_SHORT = frozenset({"101", "102", "103", "104", "112"})
SERVICE_SHORT_RE = re.compile(r"^(?:\d{3,4}|8800\d{7}|\+78800\d{7})$")

# Примерные TAC (первые 8 цифр IMEI) моделей GSM-шлюзов и SIM-банков.
GSM_GATEWAY_TACS = frozenset({"86206700", "35226305", "86778402", "35909300"})


def normalize(number: str) -> str:
    raw = re.sub(r"[^\d+]", "", number or "")
    if raw.startswith("8") and len(raw) == 11:
        raw = "+7" + raw[1:]
    if raw.startswith("7") and len(raw) == 11 and not raw.startswith("+"):
        raw = "+" + raw
    return raw


def is_short(number: str) -> bool:
    return bool(SERVICE_SHORT_RE.match(normalize(number))) and not normalize(number).startswith("+7")


def is_service_number(number: str) -> bool:
    n = normalize(number)
    return is_short(n) or n.startswith("+78800") or n.startswith("8800")


def is_emergency_short(number: str) -> bool:
    return normalize(number) in EMERGENCY_SHORT


def number_country(number: str) -> str:
    """Страна по плану нумерации. +76/+77 — Казахстан, хотя код тоже +7."""
    n = normalize(number)
    if is_short(n):
        return "RU"
    if n.startswith("+76") or n.startswith("+77"):
        return "KZ"
    if n.startswith("+7"):
        return "RU"
    return "XX"


def is_landline(number: str) -> bool:
    n = normalize(number)
    return (
        number_country(n) == "RU"
        and len(n) == 12
        and n[2] in "348"
        and not n.startswith("+7800")
    )


def _stable_int(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)


@lru_cache(maxsize=200_000)
def region_of(number: str) -> int:
    """Код региона 1..85; для коротких и зарубежных номеров 0."""
    n = normalize(number)
    if number_country(n) != "RU" or len(n) != 12:
        return 0
    body = n[2:]
    for code, region in LANDLINE_CODES.items():
        if body.startswith(code):
            return region
    return 1 + _stable_int(body[:4]) % N_REGIONS


@lru_cache(maxsize=None)
def tz_offset_msk(region: int) -> int:
    """Сдвиг часового пояса региона относительно Москвы."""
    if region in (0, MOSCOW, SPB):
        return 0
    offsets = (0, 0, 0, 0, 1, 2, 2, 3, 4, 4, 5, 6, 7, 8, 9, -1)
    return offsets[_stable_int(f"tz{region}") % len(offsets)]


def operator_of(number: str) -> str:
    n = normalize(number)
    if is_short(n):
        return "service"
    if number_country(n) != "RU":
        return "foreign"
    if is_landline(n):
        return "Rostelecom"
    return MOBILE_OPERATORS[_stable_int(n[2:6] + "op") % len(MOBILE_OPERATORS)]


@lru_cache(maxsize=None)
def mobile_prefixes_by_region() -> dict[int, list[str]]:
    """Префиксы вида +79XXX, сгруппированные по региону — для генерации номеров."""
    out: dict[int, list[str]] = {}
    for block in range(900, 1000):
        for d in range(10):
            prefix = f"+7{block}{d}"
            out.setdefault(region_of(prefix + "000000"), []).append(prefix)
    return out


def mask(number: str) -> str:
    n = normalize(number)
    if n.startswith("+7") and len(n) == 12:
        return f"+7 {n[2:5]} ***-**-{n[-2:]}"
    return n

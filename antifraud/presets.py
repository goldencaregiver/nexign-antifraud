"""Готовые истории для демо: по одному вызову из формы оконные признаки пусты.
Каждая история строится вокруг номеров и времени из формы, поэтому поля можно
менять, а накопленный «фон» номера останется.
"""
from __future__ import annotations
from datetime import datetime, timedelta
import numpy as np
from . import numbering as nb
from .features import Call, FeatureStore

PRESETS = {
    "fraud": {
        "title": "Мошенник",
        "call": {
            "a_number": "+37120123456",  # Латвийский номер (иностранец)
            "b_number": "+79169876583",  # Российский получатель
            "ts": "2026-10-14T11:42:07",
            "ingress_country": "LV",
            "connection_type": "voip",
            "duration_sec": 6,
            "outcome": "answered"
        },
    },
    "normal": {
        "title": "Обычный",
        "call": {
            "a_number": "+79161112233",  # Российский номер
            "b_number": "+79035556677",  # Российский получатель
            "ts": "2026-10-14T19:05:00",
            "ingress_country": "RU",
            "connection_type": "mobile",
            "duration_sec": 240,
            "outcome": "answered"
        },
    },
    "ambulance": {
        "title": "Экстренная служба",
        "call": {
            "a_number": "+74956201103",  # Городской номер скорой (Москва)
            "b_number": "+79165554433",  # Российский получатель
            "ts": "2026-10-14T03:12:00",
            "ingress_country": "RU",
            "connection_type": "landline",
            "duration_sec": 45,
            "outcome": "answered"
        },
    },
}

def _random_mobile(rng: np.random.Generator, region: int | None = None) -> str:
    prefixes = nb.mobile_prefixes_by_region()
    regions = sorted(r for r in prefixes if r)
    region = region if region in prefixes else int(rng.choice(regions))
    p = prefixes[region][rng.integers(len(prefixes[region]))]
    return p + f"{rng.integers(0, 10**6):06d}"

def _fraud(store: FeatureStore, a: str, b: str, ts: datetime, country: str, rng) -> None:
    """Номеру 2 дня, за последние 4 минуты — 312 вызовов в ~41 регион через зарубежный шлюз."""
    trunk = f"INTL-{country}-7"
    regions = list(rng.choice(sorted(r for r in nb.mobile_prefixes_by_region() if r), size=41, replace=False))
    events = [Call(ts - timedelta(days=2, hours=1), a, _random_mobile(rng), country, trunk, "voip",
                   outcome="no_answer")]
    for i in range(312):
        t = ts - timedelta(seconds=240) + timedelta(seconds=240 * i / 312)
        outcome = str(rng.choice(["answered", "no_answer", "rejected", "ring_drop"], p=[0.3, 0.45, 0.2, 0.05]))
        events.append(Call(t, a, _random_mobile(rng, int(regions[i % 41])), country, trunk, "voip",
                           outcome=outcome, duration_sec=float(rng.uniform(1, 12))))
    for i in range(40):
        events.append(Call(ts - timedelta(seconds=rng.uniform(5, 3000)), _random_mobile(rng),
                           _random_mobile(rng), country, trunk, "voip", outcome="no_answer"))
    for c in sorted(events, key=lambda c: c.t):
        store.update(c)

def _normal(store: FeatureStore, a: str, b: str, ts: datetime, rng) -> None:
    """Абонент полгода в сети, с получателем созванивается регулярно в обе стороны."""
    region = nb.region_of(a)
    friends = [b] + [_random_mobile(rng, region) for _ in range(8)]
    events = [Call(ts - timedelta(days=180), a, friends[1], outcome="answered", duration_sec=120)]
    for d in range(0, 15):
        for _ in range(int(rng.integers(3, 8))):
            f = friends[int(rng.integers(len(friends)))]
            t = ts - timedelta(days=d, hours=float(rng.uniform(3, 12)))
            outcome = "answered" if rng.random() < 0.85 else "no_answer"
            events.append(Call(t, a, f, outcome=outcome, duration_sec=float(rng.lognormal(4.3, 0.8))))
            if rng.random() < 0.6:
                events.append(Call(t + timedelta(hours=2), f, a, outcome="answered",
                                   duration_sec=float(rng.lognormal(4.3, 0.8))))
    for c in sorted(events, key=lambda c: c.t):
        store.update(c)

def _ambulance(store: FeatureStore, a: str, b: str, ts: datetime, rng) -> None:
    """Получатель 15 минут назад звонил в 103, подстанция ночью перезванивает разным пациентам."""
    events = [Call(ts - timedelta(minutes=15), b, "103", outcome="answered", duration_sec=95)]
    for _ in range(40):
        events.append(Call(ts - timedelta(minutes=float(rng.uniform(20, 60 * 24 * 30))), a, _random_mobile(rng),
                           connection_type="landline", outcome="answered",
                           duration_sec=float(rng.uniform(20, 90))))
    for c in sorted(events, key=lambda c: c.t):
        store.update(c)

def build_store(name: str, call: Call, seed: int = 7) -> FeatureStore:
    rng = np.random.default_rng(seed)
    store = FeatureStore()
    if name == "fraud":
        _fraud(store, call.a_number, call.b_number, call.ts,
               call.ingress_country if call.ingress_country != "RU" else "LV", rng)
    elif name == "normal":
        _normal(store, call.a_number, call.b_number, call.ts, rng)
    elif name == "ambulance":
        _ambulance(store, call.a_number, call.b_number, call.ts, rng)
    else:
        raise KeyError(name)
    return store
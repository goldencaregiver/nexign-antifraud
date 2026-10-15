"""Признаки вызова по «конверту» и накопленной истории.

`FeatureStore` обрабатывает вызовы строго по времени: сначала `features()` по
состоянию до вызова, потом `update()` после его завершения. Обучение и API
используют один и тот же класс, поэтому расчёт признаков совпадает.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from . import numbering as nb

MIN = 60.0
HOUR = 3600.0
DAY = 86400.0
RATIO_WINDOW_CALLS = 100
EMERGENCY_CALLBACK_WINDOW = 6 * HOUR

PRE_FEATURES = [
    "cli_trunk_mismatch",
    "short_cli_from_foreign",
    "a_is_landline",
    "connection_type",
    "source_operator",
    "hour_local_b",
    "pair_calls_before",
    "b_called_a_before",
    "number_age_days",
    "calls_1m",
    "calls_1h",
    "calls_24h",
    "short_ratio",
    "unanswered_ratio",
    "ring_drop_ratio",
    "unique_b_ratio_24h",
    "regions_count_1h",
    "regions_entropy_1h",
    "reciprocity",
    "in_out_ratio_24h",
    "imsi_per_imei_24h",
    "gateway_tac",
    "fan_group_size_24h",
    "fan_b_overlap",
    "trunk_distinct_a_1h",
]
POST_ONLY_FEATURES = ["answered", "duration_sec", "short_call", "ring_drop"]
POST_FEATURES = PRE_FEATURES + POST_ONLY_FEATURES
CAT_FEATURES = ["connection_type", "source_operator"]

CONNECTION_TYPES = ("mobile", "landline", "voip", "intl")
OUTCOMES = ("answered", "no_answer", "rejected", "ring_drop")


@dataclass
class Call:
    ts: datetime
    a_number: str
    b_number: str
    ingress_country: str = "RU"
    trunk_id: str = ""
    connection_type: str = "mobile"
    imei: Optional[str] = None
    imsi: Optional[str] = None
    outcome: Optional[str] = None
    duration_sec: Optional[float] = None

    def __post_init__(self) -> None:
        self.a_number = nb.normalize(self.a_number)
        self.b_number = nb.normalize(self.b_number)
        self.ingress_country = (self.ingress_country or "RU").upper()
        if not self.trunk_id:
            self.trunk_id = (
                f"RU-{nb.operator_of(self.a_number)}"
                if self.ingress_country == "RU"
                else f"INTL-{self.ingress_country}"
            )

    @property
    def t(self) -> float:
        return self.ts.timestamp()

    @property
    def completed(self) -> bool:
        return self.outcome is not None

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> "Call":
        ts = row["ts"]
        if not isinstance(ts, datetime):
            ts = datetime.fromisoformat(str(ts))
        def opt(key: str):
            value = row.get(key)
            if value is None or (isinstance(value, float) and math.isnan(value)) or value == "":
                return None
            return value
        duration = opt("duration_sec")
        return cls(
            ts=ts,
            a_number=str(row["a_number"]),
            b_number=str(row["b_number"]),
            ingress_country=str(opt("ingress_country") or "RU"),
            trunk_id=str(opt("trunk_id") or ""),
            connection_type=str(opt("connection_type") or "mobile"),
            imei=None if opt("imei") is None else str(opt("imei")),
            imsi=None if opt("imsi") is None else str(opt("imsi")),
            outcome=opt("outcome"),
            duration_sec=None if duration is None else float(duration),
        )


@dataclass
class _Out:
    t: float
    b: str
    b_region: int
    outcome: str
    duration: float


class _CountedWindow:
    """Скользящее окно событий с подсчётом уникальных значений по ключам."""

    def __init__(self, span: float, keys: tuple[str, ...]):
        self.span = span
        self.events: deque[tuple[float, tuple]] = deque()
        self.counters = {k: Counter() for k in keys}
        self.keys = keys

    def prune(self, now: float) -> None:
        while self.events and self.events[0][0] < now - self.span:
            _, values = self.events.popleft()
            for key, value in zip(self.keys, values):
                c = self.counters[key]
                c[value] -= 1
                if c[value] <= 0:
                    del c[value]

    def add(self, t: float, *values) -> None:
        self.events.append((t, values))
        for key, value in zip(self.keys, values):
            self.counters[key][value] += 1

    def distinct(self, key: str) -> int:
        return len(self.counters[key])

    def __len__(self) -> int:
        return len(self.events)


@dataclass
class FeatureStore:
    """Накопленная история. Важно: `features()` смотрит только на прошлые вызовы,
    текущий ещё не записан — иначе модель «подглядывала» бы в будущее.
    """

    out_events: dict[str, deque] = field(default_factory=lambda: defaultdict(deque))
    in_events: dict[str, deque] = field(default_factory=lambda: defaultdict(deque))
    first_seen: dict[str, float] = field(default_factory=dict)
    dialed: dict[str, set] = field(default_factory=lambda: defaultdict(set))
    callers: dict[str, set] = field(default_factory=lambda: defaultdict(set))
    reciprocal: Counter = field(default_factory=Counter)  # сколько B хоть раз перезвонили A
    pair_dir: Counter = field(default_factory=Counter)  # (A,B) → сколько раз A звонил B
    imei_window: dict[str, _CountedWindow] = field(default_factory=dict)
    trunk_window: dict[str, _CountedWindow] = field(default_factory=dict)
    last_emergency_call: dict[str, float] = field(default_factory=dict)  # кто звонил в 103/112

    def features(self, call: Call, stage: str = "pre") -> dict[str, Any]:
        t = call.t
        a, b = call.a_number, call.b_number
        b_region = nb.region_of(b)
        hour_msk = call.ts.hour + call.ts.minute / 60
        foreign = call.ingress_country != "RU"

        # Окна считаем по истории A до текущего вызова (его ещё нет в out_events).
        out = self.out_events.get(a, deque())
        last_24h = [e for e in out if e.t >= t - DAY]
        last_1h = [e for e in last_24h if e.t >= t - HOUR]
        n_1m = sum(1 for e in last_1h if e.t >= t - MIN)
        # Окно 100 вызовов, а не 10: на коротком окне доля коротких слишком шумная.
        recent = last_24h[-RATIO_WINDOW_CALLS:]

        def ratio(pred) -> float:
            # NaN, а не 0: у нового номера нет истории — это «неизвестно», а не «норма».
            return sum(1 for e in recent if pred(e)) / len(recent) if recent else math.nan

        regions = Counter(e.b_region for e in last_1h)
        if len(last_1h) > 1 and len(regions) > 1:
            # Нормируем на log(N), иначе у кого больше вызовов — выше энтропия «автоматически».
            probs = [c / len(last_1h) for c in regions.values()]
            entropy = -sum(p * math.log(p) for p in probs) / math.log(len(last_1h))
        else:
            entropy = 0.0 if last_1h else math.nan

        incoming = sum(1 for x in self.in_events.get(a, ()) if x >= t - DAY)
        dialed = self.dialed.get(a)

        imei_w = self.imei_window.get(call.imei) if call.imei else None
        if imei_w is not None:
            imei_w.prune(t)
        group = imei_w.distinct("a") if imei_w is not None and len(imei_w) else 0
        # Веер: много A на одном устройстве + почти нет общих B (списки не пересекаются).
        # fan_b_overlap ≈ 0 → идеальный веер; ближе к 1 → общие контакты, как у людей.
        if imei_w is not None and group >= 2:
            fan_overlap = 1.0 - imei_w.distinct("b") / len(imei_w)
        else:
            fan_overlap = math.nan

        # При подмене номера одноразовые — смотрим на шлюз, а не на историю A.
        trunk_w = self.trunk_window.get(call.trunk_id) if foreign else None
        if trunk_w is not None:
            trunk_w.prune(t)

        first = self.first_seen.get(a)
        feats: dict[str, Any] = {
            # Казахстан (+76/+77) тоже +7: number_country отличает его от RU.
            "cli_trunk_mismatch": int(nb.number_country(a) not in ("XX", call.ingress_country)),
            "short_cli_from_foreign": int(
                nb.is_service_number(a) and (foreign or call.connection_type == "voip")
            ),
            "a_is_landline": int(nb.is_landline(a)),
            "connection_type": call.connection_type,
            "source_operator": nb.operator_of(a) if not foreign else f"intl_{call.ingress_country}",
            "hour_local_b": round((hour_msk + nb.tz_offset_msk(b_region)) % 24, 2),
            "pair_calls_before": self.pair_dir[(a, b)] + self.pair_dir[(b, a)],
            "b_called_a_before": int(self.pair_dir[(b, a)] > 0),
            "number_age_days": round(min((t - first) / DAY, 365.0), 3) if first is not None else 0.0,
            "calls_1m": n_1m,
            "calls_1h": len(last_1h),
            "calls_24h": len(last_24h),
            "short_ratio": ratio(lambda e: e.outcome == "answered" and e.duration < 3),
            "unanswered_ratio": ratio(lambda e: e.outcome in ("no_answer", "rejected")),
            "ring_drop_ratio": ratio(lambda e: e.outcome == "ring_drop"),
            # Долю уникальных всегда смотрим вместе с calls_24h: 3 из 3 = 100% — норма.
            "unique_b_ratio_24h": len({e.b for e in last_24h}) / len(last_24h) if last_24h else math.nan,
            "regions_count_1h": len(regions),
            "regions_entropy_1h": entropy,
            "reciprocity": self.reciprocal[a] / len(dialed) if dialed else math.nan,
            "in_out_ratio_24h": incoming / len(last_24h) if last_24h else math.nan,
            "imsi_per_imei_24h": imei_w.distinct("imsi") if imei_w is not None else 0,
            "gateway_tac": int(bool(call.imei) and call.imei[:8] in nb.GSM_GATEWAY_TACS),
            "fan_group_size_24h": group,
            "fan_b_overlap": fan_overlap,
            "trunk_distinct_a_1h": trunk_w.distinct("a") if trunk_w is not None else 0,
        }
        if stage == "post":
            outcome = call.outcome or "answered"
            duration = float(call.duration_sec or 0.0) if outcome == "answered" else 0.0
            feats.update(
                answered=int(outcome == "answered"),
                duration_sec=duration,
                short_call=int(outcome == "answered" and duration < 3),
                ring_drop=int(outcome == "ring_drop"),
            )
        return feats

    def context(self, call: Call) -> dict[str, Any]:
        """Факты только для политики. В модель не идут: иначе она могла бы
        «научиться» не блокировать скорую, а стоп-кран должен быть жёстким правилом.
        """
        last = self.last_emergency_call.get(call.b_number)
        hours = (call.t - last) / HOUR if last is not None else math.inf
        return {
            "b_called_emergency_hours_ago": hours,
            "b_in_emergency_callback_window": hours * HOUR <= EMERGENCY_CALLBACK_WINDOW,
        }

    def update(self, call: Call) -> None:
        """Записать завершённый вызов. Вызывать строго после features()/score()."""
        t = call.t
        a, b = call.a_number, call.b_number
        outcome = call.outcome or "answered"
        duration = float(call.duration_sec or 0.0) if outcome == "answered" else 0.0

        out = self.out_events[a]
        out.append(_Out(t, b, nb.region_of(b), outcome, duration))
        while out and out[0].t < t - DAY:
            out.popleft()
        inc = self.in_events[b]
        inc.append(t)
        while inc and inc[0] < t - DAY:
            inc.popleft()

        self.first_seen.setdefault(a, t)
        self.first_seen.setdefault(b, t)

        if b not in self.dialed[a]:
            self.dialed[a].add(b)
            if b in self.callers[a]:
                self.reciprocal[a] += 1
        if a not in self.callers[b]:
            self.callers[b].add(a)
            if a in self.dialed[b]:
                self.reciprocal[b] += 1
        self.pair_dir[(a, b)] += 1

        if call.imei:
            w = self.imei_window.setdefault(call.imei, _CountedWindow(DAY, ("imsi", "a", "b")))
            w.prune(t)
            w.add(t, call.imsi or a, a, b)
        if call.ingress_country != "RU":
            w = self.trunk_window.setdefault(call.trunk_id, _CountedWindow(HOUR, ("a",)))
            w.prune(t)
            w.add(t, a)

        if nb.is_emergency_short(b):
            self.last_emergency_call[a] = t

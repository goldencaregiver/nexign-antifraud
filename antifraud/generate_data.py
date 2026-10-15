"""Синтетические CDR: обычные абоненты, ловушки на ложные срабатывания и мошенники.

Пишет сырые вызовы в data/calls.csv. Метка `is_fraud` — истина сценария,
`label` — она же с шумом, как в жалобах абонентов. Обучаемся на `label`,
оцениваемся на `is_fraud`.

Ловушки (vacation, bad_signal, sales, ambulance, call_center…) нужны, чтобы
модель не выучила «любой короткий / любой зарубежный = мошенник».
«Хитрые» мошенники (прогрев SIM, подмена IMEI, тихий SIM-банк) — чтобы
пороги из Excel и наивный возраст номера не давали идеальный AUC.

    python -m antifraud.generate_data --seed 42
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import numbering as nb
from . import whitelist as wl

START = datetime(2026, 9, 28)
DAYS = 9
WARMUP_DAYS = 2
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

COLUMNS = [
    "ts", "a_number", "b_number", "ingress_country", "trunk_id", "connection_type",
    "imei", "imsi", "outcome", "duration_sec", "is_fraud", "scenario", "entity",
]


@dataclass
class Subscriber:
    number: str
    region: int
    imei: str
    imsi: str
    rate: float
    contacts: list[str] = field(default_factory=list)
    weights: np.ndarray | None = None
    kind: str = "normal"
    abroad: tuple[int, int, str] | None = None
    join_day: int = 0


class Generator:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.used: set[str] = set(wl.EMERGENCY_SERVICE_NUMBERS) | set(wl.SERVICE_REGISTRY)
        self.rows: list[tuple] = []
        self.prefixes = nb.mobile_prefixes_by_region()
        self.regions = sorted(r for r in self.prefixes if r)
        weights = np.array([12.0 if r == nb.MOSCOW else 6.0 if r == nb.SPB else 1.0 for r in self.regions])
        self.region_p = weights / weights.sum()
        self.normal_tacs = [f"{self.rng.integers(10**7, 10**8 - 1)}" for _ in range(60)]

    # --- номера и устройства -------------------------------------------------

    def region(self) -> int:
        return int(self.rng.choice(self.regions, p=self.region_p))

    def mobile(self, region: int | None = None) -> str:
        region = self.region() if region is None else region
        while True:
            prefix = self.prefixes[region][self.rng.integers(len(self.prefixes[region]))]
            n = prefix + f"{self.rng.integers(0, 10**6):06d}"
            if n not in self.used:
                self.used.add(n)
                return n

    def landline(self, code: str = "495") -> str:
        while True:
            n = f"+7{code}{self.rng.integers(0, 10 ** (10 - len(code))):0{10 - len(code)}d}"
            if n not in self.used:
                self.used.add(n)
                return n

    def imei(self, tac: str | None = None) -> str:
        tac = tac or self.normal_tacs[self.rng.integers(len(self.normal_tacs))]
        return tac + f"{self.rng.integers(0, 10**7):07d}"

    def imsi(self) -> str:
        return "25001" + f"{self.rng.integers(0, 10**10):010d}"

    def at(self, day: int, hour_msk: float) -> datetime:
        return START + timedelta(days=day, hours=float(hour_msk) % 24)

    def local_hour(self, night_share: float = 0.03) -> float:
        if self.rng.random() < night_share:
            return float(self.rng.uniform(0, 6))
        return float(np.clip(self.rng.normal(15, 3.5), 7, 23.9))

    def emit(self, ts, a, b, outcome, duration, is_fraud, scenario, entity, *,
             ingress="RU", trunk="", conn="mobile", imei=None, imsi=None) -> None:
        if outcome != "answered":
            duration = 0.0
        self.rows.append((ts, a, b, ingress, trunk, conn, imei, imsi, outcome,
                          round(float(duration), 1), int(is_fraud), scenario, entity))

    def outcome(self, p_answer: float, p_no: float, p_rej: float) -> str:
        r = self.rng.random()
        if r < p_answer:
            return "answered"
        if r < p_answer + p_no:
            return "no_answer"
        if r < p_answer + p_no + p_rej:
            return "rejected"
        return "ring_drop"

    def talk(self, mean_log: float = 4.2, sigma: float = 1.0, short_p: float = 0.05) -> float:
        if self.rng.random() < short_p:
            return float(self.rng.uniform(0.5, 3))
        return float(np.clip(self.rng.lognormal(mean_log, sigma), 3, 3600))

    # --- обычные абоненты ----------------------------------------------------

    def build_normals(self, n: int) -> list[Subscriber]:
        subs = []
        for _ in range(n):
            region = self.region()
            subs.append(Subscriber(
                number=self.mobile(region), region=region, imei=self.imei(), imsi=self.imsi(),
                rate=float(np.clip(self.rng.lognormal(1.6, 0.6), 1, 40)),
            ))
        by_region: dict[int, list[Subscriber]] = {}
        for s in subs:
            by_region.setdefault(s.region, []).append(s)
        for s in subs:
            k = int(self.rng.integers(4, 26))
            local = [x for x in by_region[s.region] if x is not s]
            picks = []
            for _ in range(k):
                pool = local if local and self.rng.random() < 0.7 else subs
                c = pool[self.rng.integers(len(pool))]
                if c is not s and c.number not in picks:
                    picks.append(c.number)
            picks += [self.mobile(s.region) for _ in range(int(self.rng.integers(0, 5)))]
            s.contacts = picks
            w = 1.0 / np.arange(1, len(picks) + 1) ** 0.9
            s.weights = w / w.sum()
        by_number = {s.number: s for s in subs}
        for s in subs:
            for c in s.contacts:
                other = by_number.get(c)
                if other is not None and s.number not in other.contacts:
                    other.contacts.append(s.number)
                    w = 1.0 / np.arange(1, len(other.contacts) + 1) ** 0.9
                    other.weights = w / w.sum()
        idx = self.rng.permutation(n)
        # Ловушки: выглядят «подозрительно», но легальны.
        for i in idx[: n // 20]:
            start = int(self.rng.integers(WARMUP_DAYS - 1, DAYS - 2))
            subs[i].kind = "vacation"  # зарубежный шлюз, но звонит своим контактам
            subs[i].abroad = (start, start + int(self.rng.integers(2, 4)),
                              str(self.rng.choice(["TR", "AE", "TH", "GE", "EG", "AM"])))
        for i in idx[n // 20: n // 10]:
            subs[i].kind = "bad_signal"  # серии коротких перезвонов — не прозвон базы
        for i in idx[n // 10: n // 10 + n // 20]:
            subs[i].kind = "night_shift"
        for i in idx[n // 10 + n // 20: n // 10 + n // 20 + n // 30]:
            subs[i].kind = "sales"  # много новых номеров, как у мошенника, но с ответами
            subs[i].rate = float(self.rng.uniform(20, 60))
        for i in self.rng.choice(n, size=n // 15, replace=False):
            # Новые абоненты: number_age_days ≈ 0 — нельзя ловить мошенников только по нему.
            subs[i].join_day = int(self.rng.integers(WARMUP_DAYS - 1, DAYS - 1))
        return subs

    def run_normals(self, subs: list[Subscriber]) -> list[tuple[datetime, Subscriber]]:
        emergency_calls = []
        for s in subs:
            offset = nb.tz_offset_msk(s.region)
            for day in range(s.join_day, DAYS):
                abroad = s.abroad if s.abroad and s.abroad[0] <= day < s.abroad[1] else None
                n_calls = self.rng.poisson(s.rate * (0.5 if abroad else 1.0))
                night = 0.25 if s.kind == "night_shift" else 0.03
                to_contacts = 0.97 if abroad else 0.3 if s.kind == "sales" else 0.9
                for _ in range(n_calls):
                    ts = self.at(day, self.local_hour(night) - offset)
                    if self.rng.random() < to_contacts:
                        b = s.contacts[self.rng.choice(len(s.contacts), p=s.weights)]
                    else:
                        b = self.mobile(s.region if self.rng.random() < 0.7 else None)
                    extra = {}
                    if abroad:
                        extra = dict(ingress=abroad[2], trunk=f"INTL-{abroad[2]}-1", conn="intl")
                    if s.kind == "bad_signal" and self.rng.random() < 0.15:
                        for k in range(int(self.rng.integers(2, 5))):
                            self.emit(ts + timedelta(seconds=25 * k), s.number, b, "answered",
                                      self.rng.uniform(0.3, 2.5), 0, "bad_signal", s.number,
                                      imei=s.imei, imsi=s.imsi, **extra)
                        ts += timedelta(seconds=120)
                    answer = (0.6, 0.28, 0.08) if s.kind == "sales" else (0.82, 0.13, 0.04)
                    self.emit(ts, s.number, b, self.outcome(*answer), self.talk(short_p=0.12 if s.kind == "sales" else 0.05),
                              0, s.kind, s.number, imei=s.imei, imsi=s.imsi, **extra)
                if not abroad and self.rng.random() < 0.006:
                    ts = self.at(day, self.local_hour(0.3) - offset)
                    self.emit(ts, s.number, str(self.rng.choice(["103", "112"])), "answered",
                              self.rng.uniform(40, 200), 0, "emergency_request", s.number,
                              imei=s.imei, imsi=s.imsi)
                    emergency_calls.append((ts, s))
        return emergency_calls

    # --- легитимные ловушки --------------------------------------------------

    def run_ambulance(self, emergency_calls, normals) -> None:
        stations = sorted(wl.EMERGENCY_SERVICE_NUMBERS)
        for ts, patient in emergency_calls:
            if self.rng.random() < 0.8:
                station = stations[self.rng.integers(len(stations))]
                t = ts + timedelta(minutes=float(self.rng.uniform(3, 40)))
                for _ in range(int(self.rng.integers(1, 3))):
                    self.emit(t, station, patient.number, self.outcome(0.8, 0.15, 0.03),
                              self.rng.uniform(15, 120), 0, "ambulance", station,
                              conn="landline", trunk="RU-Rostelecom")
                    t += timedelta(minutes=float(self.rng.uniform(1, 8)))
        for station in stations:
            for day in range(DAYS):
                for _ in range(self.rng.poisson(6)):
                    b = normals[self.rng.integers(len(normals))].number
                    self.emit(self.at(day, self.rng.uniform(0, 24)), station, b,
                              self.outcome(0.75, 0.2, 0.03), self.rng.uniform(15, 120), 0,
                              "ambulance", station, conn="landline", trunk="RU-Rostelecom")

    def run_call_centers(self, normals) -> None:
        registry = list(wl.SERVICE_REGISTRY.items())
        unregistered = [(self.landline(str(self.rng.choice(["495", "499", "812"]))), "Небольшая доставка")
                        for _ in range(6)]
        for number, name in registry + unregistered:
            robot = "робот" in name
            scenario = "robot" if robot else "call_center"
            region = nb.region_of(number)
            for day in range(DAYS):
                for _ in range(self.rng.poisson(160 if robot else 220)):
                    if robot:
                        local = [s for s in normals if s.region == region] or normals
                        b = local[self.rng.integers(len(local))].number
                    else:
                        b = (normals[self.rng.integers(len(normals))].number
                             if self.rng.random() < 0.7 else self.mobile())
                    ts = self.at(day, self.rng.uniform(9, 20.5))
                    talk = self.rng.uniform(15, 45) if robot else self.talk(4.4, 0.8, 0.03)
                    self.emit(ts, number, b, self.outcome(0.66, 0.24, 0.08), talk, 0, scenario,
                              number, conn="landline", trunk="RU-Rostelecom")
                    if not robot and self.rng.random() < 0.08:
                        self.emit(ts + timedelta(minutes=float(self.rng.uniform(5, 300))), b, number,
                                  "answered", self.talk(), 0, "normal", b)

    def run_foreign_legit(self, normals) -> None:
        kz = [("+77" + f"{self.rng.integers(0, 10**9):09d}", "KZ") for _ in range(40)]
        other = [(f"+{code}{self.rng.integers(10**8, 10**9)}", c)
                 for code, c in (("90", "TR"), ("44", "GB"), ("49", "DE"), ("374", "AM"), ("995", "GE"))
                 for _ in range(8)]
        for number, country in kz + other:
            friends = [normals[self.rng.integers(len(normals))].number for _ in range(int(self.rng.integers(2, 6)))]
            for day in range(DAYS):
                for _ in range(self.rng.poisson(3)):
                    b = friends[self.rng.integers(len(friends))]
                    self.emit(self.at(day, self.local_hour()), number, b, self.outcome(0.8, 0.15, 0.04),
                              self.talk(), 0, "foreign_legit", number, ingress=country,
                              trunk=f"INTL-{country}-1", conn="intl")
                    if self.rng.random() < 0.5:
                        self.emit(self.at(day, self.local_hour()), b, number, self.outcome(0.8, 0.15, 0.04),
                                  self.talk(), 0, "normal", b)

    # --- мошенники -----------------------------------------------------------

    def fraud_call(self, ts, a, b, scenario, entity, *, answer=0.33, short_share=0.5, **extra) -> None:
        outcome = self.outcome(answer, 0.42, 0.2)
        talk = self.rng.uniform(0.5, 3) if self.rng.random() < short_share else self.talk(4.5, 0.9, 0)
        self.emit(ts, a, b, outcome, talk, 1, scenario, entity, **extra)

    def victim(self, normals) -> str:
        return normals[self.rng.integers(len(normals))].number if self.rng.random() < 0.55 else self.mobile()

    def warm_sims(self, pool, normals, day: int, entity: str) -> None:
        """Прогрев: за несколько дней до обзвона SIM делают пару «живых» вызовов в день.

        Без этого модель ловит мошенников почти только по number_age_days и ломается
        на тихом SIM-банке со старыми номерами.
        """
        numbers = [p[0] for p in pool]
        for d in range(max(0, day - int(self.rng.integers(3, 8))), day):
            for number, imsi, imei in pool:
                for _ in range(int(self.rng.integers(1, 4))):
                    b = (numbers[self.rng.integers(len(numbers))] if self.rng.random() < 0.5
                         else normals[self.rng.integers(len(normals))].number)
                    if b != number:
                        self.emit(self.at(d, self.rng.uniform(9, 21)), number, b, "answered",
                                  self.talk(), 1, "simbank", entity, imei=imei, imsi=imsi)

    def run_simbank(self, normals, n_gateways: int, *, slow: bool) -> None:
        """Агрессивный: SIM живут 1–2 дня. Тихий (slow): старые SIM, мало вызовов.

        Тихий целиком уходит в тест (HELD_OUT_SCENARIOS) — проверка обобщения.
        Часть шлюзов подменяет IMEI под каждую SIM: тогда imsi_per_imei и fan_*
        пропадают, остаются поведение (нет взаимности, много неотвеченных).
        """
        scenario = "simbank_slow" if slow else "simbank"
        for g in range(n_gateways):
            tac = sorted(nb.GSM_GATEWAY_TACS)[g % len(nb.GSM_GATEWAY_TACS)]
            gateway_imei = self.imei(tac)
            # per_sim / shared кодируем в entity — иначе стратификация по подтипу не работает.
            imei_per_sim = slow or g % 5 < 2
            entity = f"gw:{'per_sim' if imei_per_sim else 'shared'}:{gateway_imei}"
            day = 0
            while day < DAYS:
                pool = [(self.mobile(), self.imsi(), self.imei() if imei_per_sim else gateway_imei)
                        for _ in range(int(self.rng.integers(4, 9)))]
                life = DAYS if slow else int(self.rng.integers(1, 3))
                if not slow and self.rng.random() < 0.4:
                    self.warm_sims(pool, normals, day, entity)
                for d in range(day, min(day + life, DAYS)):
                    for number, imsi, imei in pool:
                        per_day = self.rng.integers(8, 25) if slow else self.rng.integers(40, 120)
                        hours = np.sort(self.rng.uniform(9, 21, size=per_day))
                        if not slow:
                            hours[self.rng.random(per_day) < 0.1] += self.rng.uniform(-8, 3)
                        for h in hours:
                            b = self.victim(normals)
                            self.fraud_call(self.at(d, h), number, b, scenario, entity,
                                            answer=0.5 if slow else 0.33,
                                            short_share=0.25 if slow else 0.5,
                                            imei=imei, imsi=imsi)
                            if self.rng.random() < (0.12 if slow else 0.04):
                                self.emit(self.at(d, h + self.rng.uniform(0.1, 3)), b, number,
                                          self.outcome(0.5, 0.4, 0.05), self.talk(), 0, "normal", b)
                day += life

    def run_intl_spoof(self, normals, n_campaigns: int) -> None:
        """Подмена CLI через зарубежный шлюз: городские, 112/8-800, номера скорой и банков.

        Именно этот сценарий даёт детерминированную автоблокировку (hard_spoof).
        """
        stations = sorted(wl.EMERGENCY_SERVICE_NUMBERS)
        registered = sorted(wl.SERVICE_REGISTRY)
        for k in range(n_campaigns):
            country = str(self.rng.choice(["LV", "GB", "DE", "PL", "NL", "EE"]))
            trunk = f"INTL-{country}-{7 + k}"
            entity = f"campaign:{k}"
            start = int(self.rng.integers(WARMUP_DAYS, DAYS - 2))
            for d in range(start, min(start + int(self.rng.integers(2, 4)), DAYS)):
                for h in np.sort(self.rng.uniform(9, 20, size=int(self.rng.integers(250, 600)))):
                    r = self.rng.random()
                    if r < 0.55:
                        cli = self.mobile()
                    elif r < 0.85:
                        cli = self.landline(str(self.rng.choice(["495", "499", "812"])))
                    elif r < 0.9:
                        cli = str(self.rng.choice(["112", "102", "+78005553535", "+78002000900"]))
                    elif r < 0.95:
                        cli = str(self.rng.choice(stations + registered))
                    else:
                        cli = self.mobile()
                    conn = "voip" if self.rng.random() < 0.5 else "intl"
                    for j in range(int(self.rng.integers(1, 3))):
                        self.fraud_call(self.at(d, h + j * 0.02), cli, self.victim(normals), "intl_spoof",
                                        entity, ingress=country, trunk=trunk, conn=conn)


def generate(seed: int = 42, n_normals: int = 2000) -> pd.DataFrame:
    g = Generator(seed)
    normals = g.build_normals(n_normals)
    emergency = g.run_normals(normals)
    g.run_ambulance(emergency, normals)
    g.run_call_centers(normals)
    g.run_foreign_legit(normals)
    g.run_simbank(normals, 20, slow=False)
    g.run_simbank(normals, 6, slow=True)
    g.run_intl_spoof(normals, 8)

    df = pd.DataFrame(g.rows, columns=COLUMNS).sort_values("ts", kind="stable").reset_index(drop=True)
    df.insert(0, "call_id", np.arange(len(df)))

    rng = np.random.default_rng(seed + 1)
    # Шум как ложные жалобы: несколько обычных абонентов помечены мошенниками.
    noisy_entities = set(rng.choice(df.loc[df.scenario == "normal", "entity"].unique(),
                                    size=max(1, n_normals // 200), replace=False))
    df["label"] = df["is_fraud"]
    df.loc[df.entity.isin(noisy_entities), "label"] = 1
    # Warmup нужен, чтобы у абонентов была история; в обучение эти дни не идут.
    df["warmup"] = df.ts < START + timedelta(days=WARMUP_DAYS)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normals", type=int, default=2000)
    args = parser.parse_args()
    df = generate(args.seed, args.normals)
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(DATA_DIR / "calls.csv", index=False)
    print(f"{len(df)} вызовов -> {DATA_DIR / 'calls.csv'}")
    print(df.groupby("scenario").agg(calls=("call_id", "size"), fraud=("is_fraud", "mean")).to_string())


if __name__ == "__main__":
    main()

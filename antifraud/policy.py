"""Что делать с вызовом. Модель даёт риск, а решение принимает политика.

Уровни:
    0 — пропустить;
    1 — метка «возможно мошенничество» у получателя, вызов проходит;
    2 — метка и номер A на проверку / ограничение исходящих;
    3 — автоблокировка. Только до ответа и только при подмене номера,
        которую видно по маршруту. Один поведенческий балл сюда не доводит.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from . import numbering as nb
from . import whitelist as wl
from .features import Call

ACTIONS = {
    0: "Пропустить",
    1: "Метка «возможно мошенничество» у получателя",
    2: "Пометить абонента и отправить номер на проверку",
    3: "Заблокировать вызов",
}


@dataclass
class Thresholds:
    label: float
    review: float
    block: float

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> "Thresholds":
        return cls(label=d["label"], review=d["review"], block=d["block"])


@dataclass
class Decision:
    level: int
    action: str
    autoblock: bool
    reasons: list[str] = field(default_factory=list)
    protected: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "action": self.action,
            "autoblock": self.autoblock,
            "reasons": self.reasons,
            "protected": self.protected,
        }


def autoblock_enabled() -> bool:
    return os.getenv("AUTOBLOCK_ENABLED", "1") not in ("0", "false", "False")


def decide(call: Call, score: float, feats: dict[str, Any], ctx: dict[str, Any],
           thresholds: Thresholds, stage: str) -> Decision:
    """Порядок важен: сначала балл, потом подмена, потом стоп-краны (реестр/скорая).

    Стоп-краны идут последними специально: они могут понизить уровень, который
    выставили модель и правило подмены. Так даже «подозрительный» звонок скорой
    не получает блокировку.
    """
    reasons: list[str] = []
    a = call.a_number
    foreign = call.ingress_country != "RU"
    registry = wl.lookup(a)
    # История пары — главный тормоз ложных блокировок (отпускник, родные).
    no_history = feats.get("pair_calls_before", 0) == 0

    # 1) Балл модели → метка / проверка. До уровня 3 балл сам по себе не доводит.
    if score >= thresholds.review:
        level = 2
    elif score >= thresholds.label:
        level = 1
    else:
        level = 0

    # 2) Подмена маршрута. Городской/служебный номер физически не уезжает в роуминг —
    #    это hard_spoof. Мобильный может быть в отпуске, поэтому нужен ещё высокий балл.
    hard_spoof = foreign and nb.number_country(a) == "RU" and (
        nb.is_landline(a) or nb.is_service_number(a) or registry is not None
    )
    soft_spoof = bool(feats.get("cli_trunk_mismatch")) and no_history

    if hard_spoof:
        what = "номер из реестра" if registry else "городской или служебный номер"
        reasons.append(f"Российский {what} пришёл через зарубежный шлюз {call.trunk_id}: так легально не бывает")
        level = max(level, 2)
        if no_history:
            level = 3
    elif soft_spoof and score >= thresholds.block:
        reasons.append("Российский мобильный номер через зарубежный шлюз, раньше с получателем не общался")
        level = 3

    # 3) Стоп-краны. Скорая звонит с городского номера подстанции, не с «103» —
    #    поэтому реестр и окно обратного вызова важнее защиты по коротким номерам.
    protected = False
    if nb.is_emergency_short(call.b_number):
        return Decision(level=0, action=ACTIONS[0], autoblock=False, protected=True,
                        reasons=["Вызов в экстренную службу: пропускаем всегда"])
    if registry is not None and not foreign:
        kind, name = registry
        protected = True
        if kind == "emergency":
            # Метка «возможно мошенничество» на скорой тоже вредна: пациент может не взять трубку.
            reasons.append(f"{name}: вызов не трогаем")
            level = 0
        else:
            reasons.append(f"Сервисный номер из реестра ({name}): не выше метки")
            level = min(level, 1)

    if ctx.get("b_in_emergency_callback_window"):
        minutes = ctx["b_called_emergency_hours_ago"] * 60
        reasons.append(f"Получатель звонил в 103/112 {minutes:.0f} мин назад: окно обратного вызова, не блокируем")
        protected = True
        level = min(level, 1)

    # 4) Блокировать можно только пока телефон звонит. После вызова — только пометка.
    if stage == "post" and level == 3:
        reasons.append("Вызов уже состоялся: блокировать можно только следующие вызовы с этого маршрута")
        level = 2

    autoblock = level == 3 and autoblock_enabled()
    if level == 3 and not autoblock:
        reasons.append("Автоблокировка выключена настройкой AUTOBLOCK_ENABLED, только метка")
        level = 2

    return Decision(level=level, action=ACTIONS[level], autoblock=autoblock,
                    reasons=reasons, protected=protected)
